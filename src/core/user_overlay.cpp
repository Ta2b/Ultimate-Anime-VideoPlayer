#include "core/user_overlay.h"

#include <cstdint>
#include <format>
#include <fstream>
#include <random>
#include <sstream>
#include <system_error>

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
std::expected<void, Refusal> merge_profiles(toml::array& base, const toml::array& user) {
  for (std::size_t i = 0; i < user.size(); ++i) {
    const std::string where = std::format("user.toml profile[{}]", i);
    const toml::table* entry = user.get(i)->as_table();
    if (entry == nullptr) return refuse(RefusalCode::FieldType, where);
    const std::optional<std::int64_t> slot = slot_of(*entry);
    if (!slot) return refuse(RefusalCode::FieldMissing, where + ".slot");
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
  auto base_text = read_text_file(base);
  if (!base_text) return refuse(RefusalCode::ProfileUnreadable, base.string());
  std::string user_text;
  if (std::error_code error; std::filesystem::exists(user, error)) {
    auto text = read_text_file(user);
    if (!text) return refuse(RefusalCode::ProfileUnreadable, user.string());
    user_text = std::move(*text);
  }
  return parse_layered(*base_text, user_text);
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
  auto base_text = read_text_file(base);
  if (!base_text) return refuse(RefusalCode::ProfileUnreadable, base.string());
  std::string user_text;
  if (std::error_code error; std::filesystem::exists(user, error)) {
    auto text = read_text_file(user);
    if (!text) return refuse(RefusalCode::ProfileUnreadable, user.string());
    user_text = std::move(*text);
  }
  auto updated = apply_edits(*base_text, user_text, edits);
  if (!updated) return std::unexpected(updated.error());
  if (auto written = write_text_atomically(user, *updated); !written) return std::unexpected(written.error());
  return parse_layered(*base_text, *updated);
}

std::expected<std::string, Refusal> read_text_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return refuse(RefusalCode::FileUnreadable, path.string());
  std::ostringstream text;
  text << in.rdbuf();
  if (in.bad()) return refuse(RefusalCode::FileUnreadable, path.string());
  return text.str();
}

std::expected<void, Refusal> write_text_atomically(const std::filesystem::path& path, std::string_view text) {
  std::error_code error;
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), error);
  std::random_device entropy;
  const std::filesystem::path temp =
      path.parent_path() / std::format(".{}.{:08x}.tmp", path.filename().string(), entropy());
  {
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.flush();
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
