#include "core/settings.h"

#include <algorithm>
#include <format>
#include <initializer_list>
#include <utility>

#include "core/model_index.h"
#include "core/toml_fields.h"
#include "core/user_overlay.h"

namespace ajp::core {
namespace {

using fields::refuse;

constexpr std::array<std::string_view, kUsageCount> kUsageNames{"upscale", "interpolate", "truehdr"};

bool is_hdr_sdk(EngineId engine) {
  return engine == EngineId::RtxVideoSdk || engine == EngineId::MaxineVfx;
}

// Étiquette BCP 47 simplifiée : langue de 2 ou 3 lettres minuscules, puis sous-étiquettes alphanumériques.
bool language_tag(std::string_view tag) {
  const auto alnum = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
  };
  const std::size_t dash = tag.find('-');
  const std::string_view language = tag.substr(0, dash);
  if (language.size() < 2 || language.size() > 3) return false;
  if (!std::ranges::all_of(language, [](char c) { return c >= 'a' && c <= 'z'; })) return false;
  if (dash == std::string_view::npos) return true;
  for (std::string_view rest = tag.substr(dash + 1);;) {
    const std::size_t next = rest.find('-');
    const std::string_view part = rest.substr(0, next);
    if (part.size() < 2 || part.size() > 8 || !std::ranges::all_of(part, alnum)) return false;
    if (next == std::string_view::npos) return true;
    rest.remove_prefix(next + 1);
  }
}

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

std::expected<void, Refusal> check_languages(const std::vector<std::string>& tags, std::string_view where) {
  if (tags.size() > limits::kMaxLanguages) return refuse(RefusalCode::FieldOutOfRange, std::string(where));
  for (std::size_t i = 0; i < tags.size(); ++i) {
    if (!language_tag(tags[i])) {
      return refuse(RefusalCode::FieldOutOfRange, std::format("{}[{}] = « {} »", where, i, tags[i]));
    }
  }
  return no_duplicate(tags, where);
}

std::expected<std::vector<std::string>, Refusal> strings(const toml::table& table, std::string_view key,
                                                         std::string_view where) {
  auto array = fields::array(table, key, where);
  if (!array) return std::unexpected(array.error());
  std::vector<std::string> values;
  for (std::size_t i = 0; i < (*array)->size(); ++i) {
    std::optional<std::string> value = (*array)->get(i)->value_exact<std::string>();
    if (!value) return refuse(RefusalCode::FieldType, std::format("{}.{}[{}]", where, key, i));
    values.push_back(std::move(*value));
  }
  return values;
}

std::expected<EngineId, Refusal> engine(const std::string& name, const std::string& where) {
  const std::optional<EngineId> id = parse_engine(name);
  if (!id) return refuse(RefusalCode::FieldOutOfRange, std::format("{} = « {} »", where, name));
  return *id;
}

std::expected<std::vector<EngineId>, Refusal> engines(const toml::table& table, std::string_view key) {
  auto names = strings(table, key, "engines");
  if (!names) return std::unexpected(names.error());
  std::vector<EngineId> ids;
  for (std::size_t i = 0; i < names->size(); ++i) {
    auto id = engine((*names)[i], std::format("engines.{}[{}]", key, i));
    if (!id) return std::unexpected(id.error());
    ids.push_back(*id);
  }
  return ids;
}

std::expected<void, Refusal> read_engines(const toml::table& root, Settings& settings) {
  auto table = fields::table(root, "engines", "settings");
  if (!table) return std::unexpected(table.error());
  if ((*table)->contains("chosen")) {
    auto chosen = engines(**table, "chosen");
    if (!chosen) return std::unexpected(chosen.error());
    settings.chosen_engines = std::move(*chosen);
  }
  if ((*table)->contains("installed")) {
    auto installed = engines(**table, "installed");
    if (!installed) return std::unexpected(installed.error());
    settings.installed_engines = std::move(*installed);
  }
  auto preferred = fields::table(**table, "preferred", "engines");
  if (!preferred) return std::unexpected(preferred.error());
  if ((*table)->contains("preferred")) settings.preferred = {};
  for (std::size_t i = 0; i < kUsageCount; ++i) {
    auto name = fields::optional<std::string>(**preferred, kUsageNames[i], "engines.preferred");
    if (!name) return std::unexpected(name.error());
    if (!*name) continue;
    auto id = engine(**name, std::format("engines.preferred.{}", kUsageNames[i]));
    if (!id) return std::unexpected(id.error());
    settings.preferred[i] = *id;
  }
  return {};
}

std::expected<void, Refusal> read_rest(const toml::table& root, Settings& settings) {
  auto first_run = fields::optional<bool>(root, "first_run_done", "settings");
  if (!first_run) return std::unexpected(first_run.error());
  settings.first_run_done = first_run->value_or(false);
  auto step = fields::optional<std::int64_t>(root, "first_run_step", "settings");
  if (!step) return std::unexpected(step.error());
  settings.first_run_step = static_cast<int>(std::clamp<std::int64_t>(step->value_or(0), 0, 8));
  auto registered = fields::optional<bool>(root, "open_with_registered", "settings");
  if (!registered) return std::unexpected(registered.error());
  settings.open_with_registered = registered->value_or(false);
  const auto choice = [&](const char* key, std::string& into, std::initializer_list<std::string_view> allowed)
      -> std::expected<void, Refusal> {
    auto value = fields::optional<std::string>(root, key, "settings");
    if (!value) return std::unexpected(value.error());
    if (!*value) return {};
    if (std::ranges::find(allowed, std::string_view(**value)) == allowed.end())
      return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, std::format("settings.{} : « {} » inconnu", key, **value)});
    into = **value;
    return {};
  };
  if (auto r = choice("update_mode", settings.update_mode, {"notify", "auto", "never"}); !r) return r;
  if (auto r = choice("update_channel", settings.update_channel, {"stable", "preview"}); !r) return r;

  auto languages = fields::table(root, "languages", "settings");
  if (!languages) return std::unexpected(languages.error());
  auto audio = strings(**languages, "audio", "languages");
  if (!audio) return std::unexpected(audio.error());
  settings.audio_languages = std::move(*audio);
  auto subtitles = strings(**languages, "subtitles", "languages");
  if (!subtitles) return std::unexpected(subtitles.error());
  settings.subtitle_languages = std::move(*subtitles);

  auto playback = fields::table(root, "playback", "settings");
  if (!playback) return std::unexpected(playback.error());
  auto resume = fields::optional<bool>(**playback, "resume_position", "playback");
  if (!resume) return std::unexpected(resume.error());
  settings.resume_position = resume->value_or(true);
  auto delay = fields::optional<std::int64_t>(**playback, "audio_delay_ms", "playback");
  if (!delay) return std::unexpected(delay.error());
  settings.audio_delay_ms = delay->value_or(0);
  return {};
}

toml::array names_of(const std::vector<EngineId>& engines) {
  toml::array names;
  for (const EngineId id : engines) names.push_back(std::string(engine_name(id)));
  return names;
}

toml::array array_of(const std::vector<std::string>& values) {
  toml::array array;
  for (const std::string& value : values) array.push_back(value);
  return array;
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
    if (is_hdr_sdk(*id) != (static_cast<Usage>(i) == Usage::TrueHdr)) {
      return refuse(RefusalCode::FieldOutOfRange, where + " : moteur d'un autre usage");
    }
  }
  if (auto r = check_languages(settings.audio_languages, "languages.audio"); !r) return r;
  if (auto r = check_languages(settings.subtitle_languages, "languages.subtitles"); !r) return r;
  if (!limits::kAudioDelayMs.contains(settings.audio_delay_ms)) {
    return refuse(RefusalCode::FieldOutOfRange, "playback.audio_delay_ms");
  }
  return {};
}

std::expected<Settings, Refusal> parse_settings(std::string_view toml_text) {
  auto root = fields::parse(toml_text, RefusalCode::FileUnreadable, "settings.toml");
  if (!root) return std::unexpected(root.error());
  if (auto version = fields::check_version(*root, kSettingsVersion, "settings.toml"); !version) {
    return std::unexpected(version.error());
  }
  Settings settings;
  if (auto r = read_engines(*root, settings); !r) return std::unexpected(r.error());
  if (auto r = read_rest(*root, settings); !r) return std::unexpected(r.error());
  if (auto r = validate_settings(settings); !r) return std::unexpected(r.error());
  return settings;
}

std::expected<Settings, Refusal> load_settings(const std::filesystem::path& path) {
  auto text = read_text_file_if_present(path);
  if (!text) return std::unexpected(text.error());
  if (!*text) return Settings{};
  return parse_settings(**text);
}

std::expected<std::string, Refusal> format_settings(const Settings& settings,
                                                    std::string_view existing_toml) {
  if (auto r = validate_settings(settings); !r) return std::unexpected(r.error());
  auto root = fields::parse(existing_toml, RefusalCode::FileUnreadable, "settings.toml");
  if (!root) return std::unexpected(root.error());

  root->insert_or_assign("version", kSettingsVersion);
  root->insert_or_assign("first_run_done", settings.first_run_done);
  root->insert_or_assign("first_run_step", settings.first_run_step);
  root->insert_or_assign("open_with_registered", settings.open_with_registered);
  root->insert_or_assign("update_mode", settings.update_mode);
  root->insert_or_assign("update_channel", settings.update_channel);
  toml::table& engines_table = fields::child(*root, "engines");
  engines_table.insert_or_assign("chosen", names_of(settings.chosen_engines));
  engines_table.insert_or_assign("installed", names_of(settings.installed_engines));
  toml::table& preferred = fields::child(engines_table, "preferred");
  for (std::size_t i = 0; i < kUsageCount; ++i) {
    if (const std::optional<EngineId>& engine = settings.preferred[i]) {
      preferred.insert_or_assign(kUsageNames[i], std::string(engine_name(*engine)));
    } else {
      preferred.erase(kUsageNames[i]);
    }
  }
  toml::table& languages = fields::child(*root, "languages");
  languages.insert_or_assign("audio", array_of(settings.audio_languages));
  languages.insert_or_assign("subtitles", array_of(settings.subtitle_languages));
  toml::table& playback = fields::child(*root, "playback");
  playback.insert_or_assign("resume_position", settings.resume_position);
  playback.insert_or_assign("audio_delay_ms", settings.audio_delay_ms);
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
