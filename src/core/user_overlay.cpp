#include "core/user_overlay.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <functional>
#include <sstream>
#include <system_error>
#include <thread>

#include "core/toml_fields.h"

namespace ajp::core {
namespace {

using fields::refuse;

std::optional<std::int64_t> slot_of(const toml::node& node) {
  const toml::table* table = node.as_table();
  if (table == nullptr) return std::nullopt;
  return (*table)["slot"].value_exact<std::int64_t>();
}

toml::table* find_slot(toml::array& profiles, std::int64_t slot) {
  for (toml::node& node : profiles) {
    if (slot_of(node) == slot) return node.as_table();
  }
  return nullptr;
}

// Textes de la base et de user.toml ; user.toml absent vaut un texte vide.
struct LayerTexts {
  std::string base;
  std::string user;
};

std::expected<LayerTexts, Refusal> read_layers(const std::filesystem::path& base,
                                               const std::filesystem::path& user) {
  auto base_text = read_text_file(base);
  if (!base_text) return refuse(RefusalCode::ProfileUnreadable, base.string());
  auto user_text = read_text_file_if_present(user);
  if (!user_text) return refuse(RefusalCode::ProfileUnreadable, user.string());
  return LayerTexts{.base = std::move(*base_text), .user = std::move(*user_text).value_or("")};
}

// Suffixe du fichier temporaire, propre à cet appel et sans exception possible
// (std::random_device peut en lever) : horloge et identité du fil d'exécution.
std::string temp_suffix() {
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::size_t thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
  return std::format("{:x}-{:x}", ticks, thread);
}

void merge_tables(toml::table& base, const toml::table& user) {
  for (const auto& [key, node] : user) {
    toml::table* base_table = base[key].as_table();
    const toml::table* user_table = node.as_table();
    if (base_table != nullptr && user_table != nullptr) {
      merge_tables(*base_table, *user_table);
    } else {
      base.insert_or_assign(key, node);
    }
  }
}

// Chaque [[profile]] de user.toml complète celui de la base au même slot, ou s'ajoute.
// Un slot répété dans user.toml est refusé, comme dans la base : aucune fusion en silence.
std::expected<void, Refusal> merge_profiles(toml::array& base, const toml::array& user) {
  std::vector<std::int64_t> seen;
  for (std::size_t i = 0; i < user.size(); ++i) {
    const std::string where = std::format("user.toml profile[{}]", i);
    const toml::table* entry = user.get(i)->as_table();
    if (entry == nullptr) return refuse(RefusalCode::FieldType, where);
    auto slot = fields::required<std::int64_t>(*entry, "slot", where);
    if (!slot) return std::unexpected(slot.error());
    if (std::ranges::contains(seen, *slot))
      return refuse(RefusalCode::FieldOutOfRange, where + ".slot (déjà défini)");
    seen.push_back(*slot);
    if (toml::table* target = find_slot(base, *slot)) {
      merge_tables(*target, *entry);
    } else {
      base.push_back(*entry);
    }
  }
  return {};
}

// --- Modifications. ---

std::expected<void, Refusal> check_slot(int slot) {
  if (!limits::kStartProfile.contains(slot)) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("slot = {}", slot));
  }
  return {};
}

toml::array& user_profiles(toml::table& user) {
  if (toml::array* found = user["profile"].as_array()) return *found;
  return *user.insert_or_assign("profile", toml::array{}).first->second.as_array();
}

toml::table& user_profile(toml::table& user, int slot) {
  toml::array& profiles = user_profiles(user);
  if (toml::table* found = find_slot(profiles, slot)) return *found;
  profiles.push_back(toml::table{{"slot", slot}});
  return *profiles.back().as_table();
}

// Le profil tel que format_profile l'écrit : name, hdr et rule, celle-ci toujours présente.
std::expected<toml::table, Refusal> profile_table(int slot, const Profile& profile) {
  ProfileFile file{.start_profile = slot};
  file.profiles[static_cast<std::size_t>(slot - 1)] = profile;
  auto root = fields::parse(format_profile(file), RefusalCode::FieldType, "profil modifié");
  if (!root) return std::unexpected(root.error());
  toml::table table = *(*root)["profile"][0].as_table();
  if (!table.contains("rule")) table.insert("rule", toml::array{});
  return table;
}

// Écrit le champ dans user.toml, ou l'en retire quand la base porte déjà cette valeur.
template <class Field>
std::expected<void, Refusal> edit_profile(toml::table& user, const ProfileFile& base, int slot,
                                          std::string_view key, Field Profile::* member, const Field& value) {
  if (auto r = check_slot(slot); !r) return r;
  const std::optional<Profile>& base_profile = base.profiles[static_cast<std::size_t>(slot - 1)];
  if (base_profile && (*base_profile).*member == value) {
    if (toml::table* entry = find_slot(user_profiles(user), slot)) entry->erase(key);
    return {};
  }
  Profile carrier;
  carrier.*member = value;
  auto table = profile_table(slot, carrier);
  if (!table) return std::unexpected(table.error());
  user_profile(user, slot).insert_or_assign(key, *table->get(key));
  return {};
}

struct EditVisitor {
  toml::table& user;
  const LayeredProfile& base;

  std::expected<void, Refusal> operator()(const edit::StartProfile& e) const {
    if (auto r = check_slot(e.slot); !r) return r;
    if (e.slot == base.profile.start_profile) {
      user.erase("start_profile");
    } else {
      user.insert_or_assign("start_profile", e.slot);
    }
    return {};
  }
  std::expected<void, Refusal> operator()(const edit::ProfileName& e) const {
    return edit_profile(user, base.profile, e.slot, "name", &Profile::name, e.name);
  }
  std::expected<void, Refusal> operator()(const edit::ProfileHdr& e) const {
    return edit_profile(user, base.profile, e.slot, "hdr", &Profile::hdr, e.mode);
  }
  std::expected<void, Refusal> operator()(const edit::ProfileRules& e) const {
    return edit_profile(user, base.profile, e.slot, "rule", &Profile::rules, e.rules);
  }
  std::expected<void, Refusal> operator()(const edit::Shortcut& e) const {
    toml::table& keys = fields::child(user, "keys");
    const std::string_view name = action_name(e.action);
    if (base.keys.chord(e.action) == e.chord) {
      keys.erase(name);
    } else {
      keys.insert_or_assign(name, e.chord ? format_chord(*e.chord) : std::string{});
    }
    return {};
  }
  std::expected<void, Refusal> operator()(const edit::ResetProfile& e) const {
    if (auto r = check_slot(e.slot); !r) return r;
    toml::array& profiles = user_profiles(user);
    for (auto it = profiles.begin(); it != profiles.end(); ++it) {
      if (slot_of(*it) == e.slot) {
        profiles.erase(it);
        break;
      }
    }
    return {};
  }
};

// Retire ce qui ne dit plus rien : profil réduit à son slot (si la base l'a), tables vides.
void prune(toml::table& user, const ProfileFile& base) {
  if (toml::array* profiles = user["profile"].as_array()) {
    for (std::size_t i = profiles->size(); i-- > 0;) {
      const toml::table* entry = profiles->get(i)->as_table();
      const std::optional<std::int64_t> slot = slot_of(*profiles->get(i));
      const bool in_base = slot && limits::kStartProfile.contains(*slot) &&
                           base.profiles[static_cast<std::size_t>(*slot - 1)].has_value();
      if (entry != nullptr && entry->size() == 1 && in_base)
        profiles->erase(profiles->begin() + static_cast<std::ptrdiff_t>(i));
    }
    if (profiles->empty()) user.erase("profile");
  }
  if (const toml::table* keys = user["keys"].as_table(); keys != nullptr && keys->empty()) user.erase("keys");
}

}  // namespace

std::expected<std::string, Refusal> merge_layers(std::string_view base_toml, std::string_view user_toml) {
  auto base = fields::parse(base_toml, RefusalCode::ProfileUnreadable, "profil de base");
  if (!base) return std::unexpected(base.error());
  auto user = fields::parse(user_toml, RefusalCode::ProfileUnreadable, "user.toml");
  if (!user) return std::unexpected(user.error());

  toml::array* base_profiles = (*base)["profile"].as_array();
  const toml::array* user_profiles_node = (*user)["profile"].as_array();
  if (base_profiles != nullptr && user_profiles_node != nullptr) {
    if (auto r = merge_profiles(*base_profiles, *user_profiles_node); !r) return std::unexpected(r.error());
    user->erase("profile");
  }
  merge_tables(*base, *user);
  return fields::format(*base);
}

std::expected<LayeredProfile, Refusal> parse_layered(std::string_view base_toml, std::string_view user_toml) {
  auto merged = merge_layers(base_toml, user_toml);
  if (!merged) return std::unexpected(merged.error());
  auto profile = parse_profile(*merged);
  if (!profile) return std::unexpected(profile.error());
  auto keys = parse_keys(*merged);
  if (!keys) return std::unexpected(keys.error());
  return LayeredProfile{.profile = std::move(*profile), .keys = *keys};
}

std::expected<LayeredProfile, Refusal> load_layered(const std::filesystem::path& base,
                                                    const std::filesystem::path& user) {
  auto texts = read_layers(base, user);
  if (!texts) return std::unexpected(texts.error());
  return parse_layered(texts->base, texts->user);
}

std::expected<std::string, Refusal> apply_edits(std::string_view base_toml, std::string_view user_toml,
                                                std::span<const UserEdit> edits) {
  auto base = parse_layered(base_toml, "");
  if (!base) return std::unexpected(base.error());
  auto user = fields::parse(user_toml, RefusalCode::ProfileUnreadable, "user.toml");
  if (!user) return std::unexpected(user.error());

  const EditVisitor visitor{.user = *user, .base = *base};
  for (const UserEdit& one : edits) {
    if (auto r = std::visit(visitor, one); !r) return std::unexpected(r.error());
  }
  prune(*user, base->profile);
  std::string text = fields::format(*user);
  if (auto layered = parse_layered(base_toml, text); !layered) return std::unexpected(layered.error());
  return text;
}

std::expected<LayeredProfile, Refusal> save_edits(const std::filesystem::path& base,
                                                  const std::filesystem::path& user,
                                                  std::span<const UserEdit> edits) {
  // Un user.toml qui serait la base elle-même (même chemin, lien) la réécrirait.
  if (std::error_code error; std::filesystem::equivalent(base, user, error)) {
    return refuse(RefusalCode::FileUnwritable, std::format("{} : c'est le profil de base", user.string()));
  }
  auto texts = read_layers(base, user);
  if (!texts) return std::unexpected(texts.error());
  auto updated = apply_edits(texts->base, texts->user, edits);
  if (!updated) return std::unexpected(updated.error());
  if (auto written = write_text_atomically(user, *updated); !written) return std::unexpected(written.error());
  return parse_layered(texts->base, *updated);
}

std::expected<std::string, Refusal> read_text_file(const std::filesystem::path& path) {
  // Un dossier s'ouvre sous Linux et se lit vide : seul un fichier ordinaire est lu.
  if (std::error_code error; !std::filesystem::is_regular_file(path, error)) {
    return refuse(RefusalCode::FileUnreadable, path.string());
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) return refuse(RefusalCode::FileUnreadable, path.string());
  std::ostringstream text;
  text << in.rdbuf();
  if (in.bad()) return refuse(RefusalCode::FileUnreadable, path.string());
  return text.str();
}

std::expected<std::optional<std::string>, Refusal> read_text_file_if_present(
    const std::filesystem::path& path) {
  std::error_code error;
  const std::filesystem::file_status status = std::filesystem::status(path, error);
  if (status.type() == std::filesystem::file_type::not_found) return std::optional<std::string>{};
  if (error)
    return refuse(RefusalCode::FileUnreadable, std::format("{} : {}", path.string(), error.message()));
  auto text = read_text_file(path);
  if (!text) return std::unexpected(text.error());
  return std::optional<std::string>{std::move(*text)};
}

std::expected<void, Refusal> write_text_atomically(const std::filesystem::path& path, std::string_view text) {
  std::error_code error;
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), error);
  const std::filesystem::path temp =
      path.parent_path() / std::format(".{}.{}.tmp", path.filename().string(), temp_suffix());
  {
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.close();  // la fermeture vide le tampon : son échec compte aussi
    if (!out) {
      std::filesystem::remove(temp, error);
      return refuse(RefusalCode::FileUnwritable, temp.string());
    }
  }
  std::filesystem::rename(temp, path, error);
  if (error) {
    const std::string why = error.message();
    std::filesystem::remove(temp, error);
    return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", path.string(), why));
  }
  return {};
}

}  // namespace ajp::core
