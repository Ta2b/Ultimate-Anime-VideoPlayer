#include "core/settings.h"

#include <algorithm>
#include <format>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>

#include "core/model_index.h"
#include "core/text_file.h"
#include "core/toml_fields.h"

namespace ajp::core {
namespace {

constexpr std::array<std::string_view, kUsageCount> kUsageNames{"upscale", "interpolate"};

template <class T>
std::expected<void, Refusal> no_duplicate(const std::vector<T>& values, std::string_view where) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      if (values[j] == values[i])
        return refuse(RefusalCode::FieldOutOfRange, std::format("{}[{}] (doublon)", where, i));
    }
  }
  return {};
}

// P2.21 : la lecture est tolérante, comme celle de play.toml. Une valeur fausse (type, nom inconnu, hors
// plage) est écartée seule, et nommée dans skipped ; le reste du fichier vaut.
struct Reader {
  std::vector<std::string>* skipped;
  void skip(std::string where) const {
    if (skipped != nullptr) skipped->push_back(std::move(where));
  }
  // La valeur lue, ou rien (écartée et nommée) si elle est fausse.
  template <class T>
  std::optional<T> take(std::expected<T, Refusal> read) const {
    if (read) return std::move(*read);
    skip(std::move(read.error().detail));
    return std::nullopt;
  }
};

std::optional<std::vector<std::string>> strings(const toml::table& table, std::string_view key,
                                                std::string_view where, const Reader& reader) {
  auto array = reader.take(fields::array(table, key, where));
  if (!array) return std::nullopt;
  std::vector<std::string> values;
  for (std::size_t i = 0; i < (*array)->size(); ++i) {
    if (std::optional<std::string> value = (*array)->get(i)->value_exact<std::string>())
      values.push_back(std::move(*value));
    else
      reader.skip(std::format("{}.{}[{}]", where, key, i));
  }
  return values;
}

std::optional<EngineId> engine(const std::string& name, const std::string& where, const Reader& reader) {
  const std::optional<EngineId> id = parse_engine(name);
  if (!id) reader.skip(std::format("{} = « {} »", where, name));
  return id;
}

std::optional<std::vector<EngineId>> engines(const toml::table& table, std::string_view key,
                                             const Reader& reader) {
  auto names = strings(table, key, "engines", reader);
  if (!names) return std::nullopt;
  std::vector<EngineId> ids;
  for (std::size_t i = 0; i < names->size(); ++i) {
    if (retired_engine((*names)[i])) continue;
    const auto id = engine((*names)[i], std::format("engines.{}[{}]", key, i), reader);
    if (id && !std::ranges::contains(ids, *id)) ids.push_back(*id);
  }
  return ids;
}

void read_engines(const toml::table& root, Settings& settings, const Reader& reader) {
  auto table = reader.take(fields::table(root, "engines", "settings"));
  if (!table) return;
  if ((*table)->contains("chosen")) {
    if (auto chosen = engines(**table, "chosen", reader)) settings.chosen_engines = std::move(*chosen);
  }
  if ((*table)->contains("installed")) {
    if (auto installed = engines(**table, "installed", reader))
      settings.installed_engines = std::move(*installed);
  }
  // Vulkan est livré partout et ne se désinstalle pas (engines.h, Delivery::Bundled).
  for (std::vector<EngineId>* list : {&settings.chosen_engines, &settings.installed_engines})
    if (!std::ranges::contains(*list, EngineId::Vulkan)) list->insert(list->begin(), EngineId::Vulkan);
  auto preferred = reader.take(fields::table(**table, "preferred", "engines"));
  if (!preferred) return;
  if ((*table)->contains("preferred")) settings.preferred = {};
  for (std::size_t i = 0; i < kUsageCount; ++i) {
    auto name = reader.take(fields::optional<std::string>(**preferred, kUsageNames[i], "engines.preferred"));
    if (!name || !*name || retired_engine(**name)) continue;
    const std::string where = std::format("engines.preferred.{}", kUsageNames[i]);
    const auto id = engine(**name, where, reader);
    if (!id) continue;
    if (std::ranges::contains(settings.installed_engines, *id))
      settings.preferred[i] = id;
    else
      reader.skip(where + " : moteur non installé");
  }
}

void read_rest(const toml::table& root, Settings& settings, const Reader& reader) {
  if (auto first_run = reader.take(fields::optional<bool>(root, "first_run_done", "settings")))
    settings.first_run_done = first_run->value_or(false);
  if (auto step = reader.take(fields::optional<std::int64_t>(root, "first_run_step", "settings")))
    settings.first_run_step = static_cast<int>(std::clamp<std::int64_t>(step->value_or(0), 0, 8));
  if (auto registered = reader.take(fields::optional<bool>(root, "open_with_registered", "settings")))
    settings.open_with_registered = registered->value_or(false);

  if (auto playback = reader.take(fields::table(root, "playback", "settings"))) {
    if (auto resume = reader.take(fields::optional<bool>(**playback, "resume_position", "playback")))
      settings.resume_position = resume->value_or(true);
  }
}

toml::array names_of(const std::vector<EngineId>& engines) {
  toml::array names;
  for (const EngineId id : engines) names.push_back(std::string(engine_name(id)));
  return names;
}

}  // namespace

std::string_view usage_name(Usage usage) {
  return kUsageNames[static_cast<std::size_t>(usage)];
}

std::expected<void, Refusal> validate_settings(const Settings& settings) {
  if (auto r = no_duplicate(settings.chosen_engines, "engines.chosen"); !r) return r;
  if (auto r = no_duplicate(settings.installed_engines, "engines.installed"); !r) return r;
  // Vulkan est livré partout et ne se désinstalle pas (engines.h, Delivery::Bundled).
  for (const auto& [where, list] : {std::pair{"engines.chosen", &settings.chosen_engines},
                                    std::pair{"engines.installed", &settings.installed_engines}}) {
    if (!std::ranges::contains(*list, EngineId::Vulkan)) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("{} : vulkan, livré avec le lecteur, y figure", where));
    }
  }
  for (std::size_t i = 0; i < kUsageCount; ++i) {
    const std::optional<EngineId>& id = settings.preferred[i];
    if (!id) continue;
    const std::string where = std::format("engines.preferred.{} = « {} »", kUsageNames[i], engine_name(*id));
    if (!std::ranges::contains(settings.installed_engines, *id)) {
      return refuse(RefusalCode::FieldOutOfRange, where + " : moteur non installé");
    }
  }
  return {};
}

std::expected<Settings, Refusal> parse_settings(std::string_view toml_text,
                                                std::vector<std::string>* skipped) {
  auto root = fields::parse(toml_text, RefusalCode::FileUnreadable, "settings.toml");
  if (!root) return std::unexpected(root.error());
  // Une version plus récente se lit avec ce que celle-ci en connaît (A22-07) : le lecteur ne perd plus ses
  // moteurs choisis pour un numéro ; ses clés inconnues restent à la réécriture. Une version absente ou
  // inférieure à 1 reste refusée.
  const auto version = fields::required<std::int64_t>(*root, "version", "settings.toml");
  if (!version) return std::unexpected(version.error());
  if (*version < kSettingsVersion)
    return std::unexpected(
        refuse(RefusalCode::FormatVersion, std::format("settings.toml version = {}", *version)));
  if (*version > kSettingsVersion && skipped != nullptr)
    skipped->push_back(std::format("version = {} : plus récente que {}, lue avec les clés connues", *version,
                                   kSettingsVersion));
  Settings settings;
  const Reader reader{.skipped = skipped};
  read_engines(*root, settings, reader);
  read_rest(*root, settings, reader);
  if (auto r = validate_settings(settings); !r) return std::unexpected(r.error());  // ne devrait plus arriver
  return settings;
}

std::expected<Settings, Refusal> load_settings(const std::filesystem::path& path,
                                               std::vector<std::string>* skipped) {
  auto text = read_text_file_if_present(path);
  if (!text) return std::unexpected(text.error());
  if (!*text) return Settings{};
  return parse_settings(**text, skipped);
}

std::expected<std::string, Refusal> format_settings(const Settings& settings,
                                                    std::string_view existing_toml) {
  if (auto r = validate_settings(settings); !r) return std::unexpected(r.error());
  auto root = fields::parse(existing_toml, RefusalCode::FileUnreadable, "settings.toml");
  if (!root) return std::unexpected(root.error());

  // Jamais abaissée : la version plus récente, relancée, sait qu'un lecteur plus ancien est passé.
  const std::int64_t on_disk = root->contains("version") ? (*root)["version"].value_or<std::int64_t>(0) : 0;
  root->insert_or_assign("version", std::max<std::int64_t>(kSettingsVersion, on_disk));
  root->insert_or_assign("first_run_done", settings.first_run_done);
  root->insert_or_assign("first_run_step", settings.first_run_step);
  root->insert_or_assign("open_with_registered", settings.open_with_registered);
  toml::table& engines_table = fields::child(*root, "engines");
  engines_table.insert_or_assign("chosen", names_of(settings.chosen_engines));
  engines_table.insert_or_assign("installed", names_of(settings.installed_engines));
  toml::table& preferred = fields::child(engines_table, "preferred");
  preferred.erase("truehdr");  // RTX HDR, retiré
  for (std::size_t i = 0; i < kUsageCount; ++i) {
    if (const std::optional<EngineId>& engine = settings.preferred[i]) {
      preferred.insert_or_assign(kUsageNames[i], std::string(engine_name(*engine)));
    } else {
      preferred.erase(kUsageNames[i]);
    }
  }
  toml::table& playback = fields::child(*root, "playback");
  playback.insert_or_assign("resume_position", settings.resume_position);
  return fields::format(*root);
}

std::expected<void, Refusal> save_settings(const std::filesystem::path& path, const Settings& settings) {
  auto existing = read_text_file_if_present(path);
  if (!existing) return std::unexpected(existing.error());
  auto text = format_settings(settings, existing->value_or(""));
  if (!text) return std::unexpected(text.error());
  return write_text_atomically(path, *text);
}

}  // namespace ajp::core
