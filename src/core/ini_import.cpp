// Fichier, profils et bornes ; les étapes sont dans ini_import_steps.cpp.
// Repris, pour la sémantique seulement (le code est réécrit) :
//   animejanai-inference-rtx@118c272 src/aji_conf.cpp:85-205 : lecture INI, bornes des chaînes ;
//   mpv-animejanai-hdr AnimeJaNaiPlatform/PlatformServices.cs:345-452 (Read, Migrate, Normalize) ;
//   manager-v2 ViewModels/MainWindowViewModel.cs:923-960 (default_slot, rife_*, config_version).
// Comportement fixé par tests/unit/core/ini_import_test.cpp.
#include "core/ini_import.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <set>
#include <utility>

#include "core/ini_import_internal.h"
#include "core/user_overlay.h"

namespace ajp::core {
namespace ini {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::string lower(std::string_view text) {
  std::string out(text);
  for (char& c : out) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return out;
}

std::string_view trim(std::string_view text) {
  const std::size_t first = text.find_first_not_of(" \t\r");
  if (first == std::string_view::npos) return {};
  return text.substr(first, text.find_last_not_of(" \t\r") - first + 1);
}

std::optional<std::int64_t> parse_integer(std::string_view text) {
  if (text.empty() || !std::ranges::all_of(text, [](char c) { return c >= '0' && c <= '9'; }))
    return std::nullopt;
  std::int64_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) return std::nullopt;
  return value;
}

std::optional<double> parse_real(std::string_view text) {
  double value = 0.0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value)) return std::nullopt;
  return value;
}

IniValue* Keys::find(std::string_view rest) const {
  const auto it = section->find(name(rest));
  if (it == section->end()) return nullptr;
  it->second.used = true;
  return &it->second;
}

int Keys::line_of(std::string_view rest) const {
  const IniValue* value = find(rest);
  return value == nullptr ? 0 : value->line;
}

std::string Keys::text(std::string_view rest, std::string_view fallback) const {
  const IniValue* value = find(rest);
  return value == nullptr ? std::string(fallback) : value->text;
}

std::unexpected<Refusal> Keys::bad(RefusalCode code, std::string_view rest, std::string_view why) const {
  const IniValue* value = find(rest);
  return refuse(code, std::format("ligne {} : {} = « {} » : {}", value == nullptr ? 0 : value->line,
                                  name(rest), value == nullptr ? "" : value->text, why));
}

std::expected<bool, Refusal> Keys::flag(std::string_view rest, bool fallback) const {
  const IniValue* value = find(rest);
  if (value == nullptr) return fallback;
  const std::string answer = lower(value->text);
  if (answer != "yes" && answer != "no") return bad(RefusalCode::FieldType, rest, "yes ou no attendu");
  return answer == "yes";
}

std::expected<std::int64_t, Refusal> Keys::integer(std::string_view rest, std::int64_t fallback) const {
  const IniValue* value = find(rest);
  if (value == nullptr) return fallback;
  const std::optional<std::int64_t> parsed = parse_integer(value->text);
  if (!parsed) return bad(RefusalCode::FieldType, rest, "entier attendu");
  return *parsed;
}

std::expected<double, Refusal> Keys::real(std::string_view rest, double fallback) const {
  const IniValue* value = find(rest);
  if (value == nullptr || value->text.empty()) return fallback;
  const std::optional<double> parsed = parse_real(value->text);
  if (!parsed) return bad(RefusalCode::FieldType, rest, "nombre attendu");
  return *parsed;
}

bool Keys::has_prefix(std::string_view more) const {
  const std::string start = name(more);
  const auto it = section->lower_bound(start);
  return it != section->end() && it->first.starts_with(start);
}

void Keys::consume(std::string_view more) const {
  const std::string start = name(more);
  for (auto it = section->lower_bound(start); it != section->end() && it->first.starts_with(start); ++it) {
    it->second.used = true;
  }
}

int Keys::first_line() const {
  int line = 0;
  for (auto it = section->lower_bound(prefix); it != section->end() && it->first.starts_with(prefix); ++it) {
    if (line == 0 || it->second.line < line) line = it->second.line;
  }
  return line;
}

namespace {

// --- Lecture INI, comme aji_conf.cpp:85-110 : sections et clés en minuscules, valeurs rognées. ---

struct IniFile {
  std::map<std::string, IniSection> sections;
  std::map<std::string, int> lines;  // ligne de l'en-tête de chaque section
};

std::expected<IniFile, Refusal> read_ini(std::string_view text, std::vector<ImportNote>& notes) {
  if (text.starts_with("\xEF\xBB\xBF")) text.remove_prefix(3);
  IniFile ini;
  std::string section;
  int number = 0;
  for (std::size_t start = 0; start <= text.size();) {
    const std::size_t end = std::min(text.find('\n', start), text.size());
    const std::string_view line = trim(text.substr(start, end - start));
    start = end + 1;
    ++number;
    if (line.empty() || line.front() == ';' || line.front() == '#') continue;
    if (line.front() == '[' && line.back() == ']') {
      section = lower(trim(line.substr(1, line.size() - 2)));
      ini.sections[section];
      ini.lines.try_emplace(section, number);
      continue;
    }
    const std::size_t equals = line.find('=');
    const std::string key = equals == std::string_view::npos ? "" : lower(trim(line.substr(0, equals)));
    if (key.empty())
      return refuse(RefusalCode::FieldType, std::format("ligne {} : ni [section] ni clé = valeur", number));
    if (section.empty())
      return refuse(RefusalCode::FieldType, std::format("ligne {} : clé hors de toute section", number));
    const bool inserted =
        ini.sections[section]
            .insert_or_assign(key,
                              IniValue{.text = std::string(trim(line.substr(equals + 1))), .line = number})
            .second;
    if (!inserted)
      notes.push_back({number, std::format("[{}] {} répétée : la dernière valeur l'emporte", section, key)});
  }
  return ini;
}

// --- Bornes d'une chaîne. ---

// « 1920x1080 » : nombre de pixels. Vide, 0 ou inf dans max_resolution : sans borne.
std::expected<std::int64_t, Refusal> pixels(const Keys& chain, std::string_view rest, bool upper) {
  const IniValue* value = chain.find(rest);
  if (value == nullptr || value->text.empty()) return upper ? limits::kPixels.hi : limits::kPixels.lo;
  const std::size_t x = value->text.find('x');
  if (x == std::string::npos) return chain.bad(RefusalCode::FieldType, rest, "largeurxhauteur attendu");
  const std::string_view text = value->text;
  const std::array sides{lower(text.substr(0, x)), lower(text.substr(x + 1))};
  std::int64_t product = 1;
  for (const std::string& side : sides) {
    const std::optional<std::int64_t> length = parse_integer(side);
    if (side == "inf" || (length && *length == 0 && upper)) {
      if (upper) return limits::kPixels.hi;
      return chain.bad(RefusalCode::FieldOutOfRange, rest, "une borne basse est finie");
    }
    if (!length) return chain.bad(RefusalCode::FieldType, rest, "largeurxhauteur attendu");
    if (*length > limits::kPixels.hi)
      return chain.bad(RefusalCode::FieldOutOfRange, rest, "au plus 16384×16384");
    product *= *length;
  }
  if (product > limits::kPixels.hi)
    return chain.bad(RefusalCode::FieldOutOfRange, rest, "au plus 16384×16384");
  return product;
}

// Vide : sans borne ; max_fps 0 ou inf : sans borne, comme dans le fork.
std::expected<double, Refusal> fps(const Keys& chain, std::string_view rest, bool upper) {
  const IniValue* value = chain.find(rest);
  if (value == nullptr || value->text.empty()) return upper ? limits::kFps.hi : limits::kFps.lo;
  if (lower(value->text) == "inf") {
    if (upper) return limits::kFps.hi;
    return chain.bad(RefusalCode::FieldOutOfRange, rest, "une borne basse est finie");
  }
  const std::optional<double> parsed = parse_real(value->text);
  if (!parsed) return chain.bad(RefusalCode::FieldType, rest, "nombre attendu");
  if (upper && *parsed == 0.0) return limits::kFps.hi;
  if (!limits::kFps.contains(*parsed))
    return chain.bad(RefusalCode::FieldOutOfRange, rest, "de 0 à 1000 fps");
  return *parsed;
}

std::expected<Rule, Refusal> import_chain(const Keys& chain, Importer& im, std::vector<HdrMode>& hdr) {
  auto min_pixels = pixels(chain, "min_resolution", false);
  auto max_pixels = pixels(chain, "max_resolution", true);
  auto min_fps = fps(chain, "min_fps", false);
  auto max_fps = fps(chain, "max_fps", true);
  if (!min_pixels) return std::unexpected(min_pixels.error());
  if (!max_pixels) return std::unexpected(max_pixels.error());
  if (!min_fps) return std::unexpected(min_fps.error());
  if (!max_fps) return std::unexpected(max_fps.error());
  Rule rule{.min_pixels = *min_pixels, .max_pixels = *max_pixels, .min_fps = *min_fps, .max_fps = *max_fps};

  const bool ordered = chain.section->contains(chain.name("pipeline_version"));
  if (!ordered && chain.has_prefix("stage_")) {
    return refuse(RefusalCode::FieldMissing, std::format("ligne {} : {} : des étapes sans pipeline_version",
                                                         chain.first_line(), chain.name("")));
  }
  auto steps = ordered ? stage_steps(chain, im, hdr) : legacy_steps(chain, im);
  if (!steps) return std::unexpected(steps.error());
  rule.steps = std::move(*steps);

  const std::string where = std::format("ligne {} : {}", chain.first_line(), chain.name(""));
  if (rule.steps.size() > limits::kMaxSteps) return refuse(RefusalCode::TooManySteps, where);
  double product = 1.0;
  for (const Step& step : rule.steps) {
    if (step.kind != StepKind::Downscale) product *= step.factor.value();
  }
  if (product > limits::kMaxFactorProduct) return refuse(RefusalCode::FactorProductTooLarge, where);
  return rule;
}

std::expected<Profile, Refusal> import_slot(IniSection& section, Importer& im) {
  const Keys top{&section, ""};
  // Sans profile_name, le fork et le gestionnaire affichent « New Profile » (aji_conf.h:41).
  Profile profile{.name = top.text("profile_name", "New Profile")};
  section.erase("hdr_stage_controlled");  // marqueur retiré par le gestionnaire v4

  std::set<std::int64_t> chains;
  for (const auto& [key, value] : section) {
    if (!key.starts_with("chain_")) continue;
    const std::size_t end = key.find('_', 6);
    if (auto number = parse_integer(std::string_view(key).substr(6, end - 6)); number && *number > 0) {
      chains.insert(*number);
    }
  }
  std::vector<HdrMode> hdr;  // au plus un : la deuxième étape HDR est refusée avec sa ligne
  for (const std::int64_t number : chains) {
    auto rule = import_chain(top.sub(std::format("chain_{}_", number)), im, hdr);
    if (!rule) return std::unexpected(rule.error());
    profile.rules.push_back(std::move(*rule));
  }
  if (!hdr.empty()) profile.hdr = hdr.front();
  return profile;
}

void note_unused(const IniSection& section, std::string_view name, Importer& im) {
  for (const auto& [key, value] : section) {
    if (!value.used) im.note(value.line, std::format("[{}] {} : sans équivalent en v5, ignorée", name, key));
  }
}

// [global] : version du fichier, profil de départ et réglages RIFE partagés.
std::expected<std::pair<std::int64_t, std::int64_t>, Refusal> import_global(IniSection& global,
                                                                            Importer& im) {
  const Keys g{&global, ""};
  auto version = g.integer("config_version", 1);
  if (!version) return std::unexpected(version.error());
  if (*version < 1) return g.bad(RefusalCode::FieldOutOfRange, "config_version", "version de 1 à 4");
  if (*version > kIniNewestVersion) {
    return g.bad(RefusalCode::FormatVersion, "config_version", "écrit par un gestionnaire plus récent");
  }
  auto default_slot = g.integer("default_slot", 0);
  if (!default_slot) return std::unexpected(default_slot.error());
  const std::string engine = lower(g.text("rife_engine", "fast"));
  if (engine != "quality" && engine != "old" && engine != "fast" && engine != "new") {
    return g.bad(RefusalCode::FieldOutOfRange, "rife_engine", "quality ou fast attendu");
  }
  im.policy = engine == "quality" || engine == "old" ? Policy::Quality : Policy::Fast;
  auto drawings = g.flag("rife_drawings", true);
  if (!drawings) return std::unexpected(drawings.error());
  im.drawings = *drawings;
  if (const IniValue* backend = g.find("backend")) {
    im.backend = lower(backend->text);
    im.note(backend->line, "[global] backend : le moteur se choisit dans Paramètres > Moteurs");
  }
  note_unused(global, "global", im);
  return std::pair{*version, *default_slot};
}

std::expected<ImportResult, Refusal> import(std::string_view ini_text) {
  Importer im;
  auto ini = read_ini(ini_text, im.notes);
  if (!ini) return std::unexpected(ini.error());
  IniSection& global = ini->sections["global"];
  auto header = import_global(global, im);
  if (!header) return std::unexpected(header.error());
  const auto [version, default_slot] = *header;

  ProfileFile file;
  for (auto& [name, section] : ini->sections) {
    if (name == "global") continue;
    const int line = ini->lines[name];
    if (!name.starts_with("slot_")) {
      im.note(line, std::format("[{}] : section inconnue, ignorée", name));
      continue;
    }
    const std::optional<std::int64_t> slot = parse_integer(std::string_view(name).substr(5));
    if (slot && *slot >= 1000) {
      im.note(line, std::format("[{}] : ancien profil intégré, remplacé par les profils v5", name));
      continue;
    }
    if (!slot || !limits::kStartProfile.contains(*slot)) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("ligne {} : [{}] : neuf profils, de 1 à 9", line, name));
    }
    auto profile = import_slot(section, im);
    if (!profile) return std::unexpected(profile.error());
    file.profiles[static_cast<std::size_t>(*slot - 1)] = std::move(*profile);
    note_unused(section, name, im);
  }

  // v5 démarre toujours sur un profil ; « 0 » (traitement coupé) devient le premier profil défini.
  const auto first = std::ranges::find_if(file.profiles, [](const auto& p) { return p.has_value(); });
  const int first_slot =
      first == file.profiles.end() ? 1 : static_cast<int>(first - file.profiles.begin()) + 1;
  const bool usable = limits::kStartProfile.contains(default_slot) &&
                      file.profiles[static_cast<std::size_t>(default_slot - 1)].has_value();
  file.start_profile = usable ? static_cast<int>(default_slot) : first_slot;
  if (!usable && first != file.profiles.end()) {
    im.note(
        Keys{&global, ""}.line_of("default_slot"),
        std::format("default_slot = {} : le lecteur démarre sur le profil {} ; Ctrl+0 coupe le traitement",
                    default_slot, first_slot));
  }

  std::ranges::stable_sort(im.notes, {}, &ImportNote::line);
  std::string toml =
      std::format("# Importé d'animejanai.conf (format {}).\n", version) + format_profile(file);
  auto reread = parse_profile(toml);
  if (!reread) return refuse(reread.error().code, "TOML v5 produit : " + reread.error().detail);
  if (*reread != file)
    return refuse(RefusalCode::ProfileUnreadable, "TOML v5 produit : relecture différente");
  return ImportResult{.profile = std::move(file),
                      .toml = std::move(toml),
                      .source_version = static_cast<int>(version),
                      .notes = std::move(im.notes)};
}

}  // namespace
}  // namespace ini

std::expected<ImportResult, Refusal> import_ini(std::string_view ini_text) {
  return ini::import(ini_text);
}

std::expected<ImportResult, Refusal> import_ini_file(const std::filesystem::path& path) {
  auto text = read_text_file(path);
  if (!text) return std::unexpected(text.error());
  return import_ini(*text);
}

}  // namespace ajp::core
