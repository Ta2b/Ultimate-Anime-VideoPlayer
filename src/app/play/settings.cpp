#include "app/play/settings.h"

#include <array>
#include <cmath>
#include <format>
#include <fstream>
#include <optional>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++/toml.hpp>

namespace ajp::app::play {
namespace {

constexpr std::int64_t kSettingsVersion = 1;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

struct DownscaleName {
  Downscale filter;
  std::string_view key;
  std::string_view label;
};
constexpr std::array<DownscaleName, kDownscaleCount> kDownscales{{
    {Downscale::Area, "area", "Area"},
    {Downscale::Bilinear, "bilinear", "Bilinear"},
    {Downscale::Bicubic, "bicubic", "Bicubic"},
    {Downscale::Mitchell, "mitchell", "Mitchell"},
    {Downscale::CatmullRom, "catmull-rom", "Catmull-Rom"},
}};

template <class T>
std::expected<void, Refusal> check(std::string_view key, T value, core::Range<T> range) {
  if (range.contains(value)) return {};
  return refuse(RefusalCode::FieldOutOfRange,
                std::format("{} : {} hors de {} à {}", key, value, range.lo, range.hi));
}

// Lit table[key] s'il existe, du type de out (entier, booléen, réel, texte) ; FieldType sinon.
template <class T>
std::expected<void, Refusal> read(const toml::table& table, std::string_view section, std::string_view key,
                                  T& out) {
  const toml::node* node = table.get(key);
  if (node == nullptr) return {};
  if constexpr (std::is_same_v<T, bool>) {
    if (auto v = node->value_exact<bool>()) {
      out = *v;
      return {};
    }
  } else if constexpr (std::is_same_v<T, int>) {
    if (auto v = node->value_exact<std::int64_t>(); v && *v >= INT32_MIN && *v <= INT32_MAX) {
      out = static_cast<int>(*v);
      return {};
    }
  } else if constexpr (std::is_same_v<T, float>) {
    if (auto v = node->value<double>()) {  // « 1 » se lit comme 1.0
      out = static_cast<float>(*v);
      return {};
    }
  } else {
    if (auto v = node->value_exact<std::string>()) {
      out = *v;
      return {};
    }
  }
  return refuse(RefusalCode::FieldType, std::format("{}.{}", section, key));
}

const toml::table* section(const toml::table& root, std::string_view name) {
  const toml::node* node = root.get(name);
  return node != nullptr ? node->as_table() : nullptr;
}

}  // namespace

std::string_view name(Downscale filter) {
  return kDownscales[static_cast<std::size_t>(filter)].key;
}

std::string_view label(Downscale filter) {
  return kDownscales[static_cast<std::size_t>(filter)].label;
}

std::expected<PlaySettings, Refusal> validate(const PlaySettings& s) {
  const std::array<std::expected<void, Refusal>, 7> checks{
      check("image.work_height", s.work_height, limits::kWorkHeight),
      check("interpolation.rife", s.rife_factor, limits::kRifeFactor),
      check("rtx_hdr.contrast", s.true_hdr_contrast, limits::kTrueHdrContrast),
      check("rtx_hdr.saturation", s.true_hdr_saturation, limits::kTrueHdrSaturation),
      check("rtx_hdr.middle_gray", s.true_hdr_gray, limits::kTrueHdrGray),
      check("audio.delay_ms", s.audio_delay_ms, limits::kAudioDelayMs),
      check("audio.volume", s.volume, limits::kVolume),
  };
  for (const auto& checked : checks) {
    if (!checked) return std::unexpected(checked.error());
  }
  if (s.true_hdr_peak != 0) {
    if (auto peak = check("rtx_hdr.peak", s.true_hdr_peak, limits::kTrueHdrPeak); !peak)
      return std::unexpected(peak.error());
  }
  if (s.model.find('/') != std::string::npos || s.model.find('\\') != std::string::npos)
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("image.model : « {} » n'est pas un nom de fichier", s.model));
  return s;
}

std::expected<PlaySettings, Refusal> parse_settings(std::string_view toml_text) {
  toml::parse_result parsed = toml::parse(toml_text);
  if (!parsed) {
    return refuse(RefusalCode::ProfileUnreadable,
                  std::format("play.toml ligne {} : {}", parsed.error().source().begin.line,
                              parsed.error().description()));
  }
  const toml::table& root = parsed.table();
  PlaySettings s;
  std::vector<std::expected<void, Refusal>> reads;
  if (const toml::table* image = section(root, "image")) {
    std::string downscale(name(s.downscale));
    reads.push_back(read(*image, "image", "upscale", s.upscale));
    reads.push_back(read(*image, "image", "model", s.model));
    reads.push_back(read(*image, "image", "work_height", s.work_height));
    reads.push_back(read(*image, "image", "downscale", downscale));
    bool known = false;
    for (const DownscaleName& entry : kDownscales) {
      if (entry.key == downscale) {
        s.downscale = entry.filter;
        known = true;
      }
    }
    if (!known)
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("image.downscale : « {} »", downscale)));
  }
  if (const toml::table* interpolation = section(root, "interpolation"))
    reads.push_back(read(*interpolation, "interpolation", "rife", s.rife_factor));
  if (const toml::table* hdr = section(root, "rtx_hdr")) {
    reads.push_back(read(*hdr, "rtx_hdr", "enabled", s.true_hdr));
    reads.push_back(read(*hdr, "rtx_hdr", "contrast", s.true_hdr_contrast));
    reads.push_back(read(*hdr, "rtx_hdr", "saturation", s.true_hdr_saturation));
    reads.push_back(read(*hdr, "rtx_hdr", "middle_gray", s.true_hdr_gray));
    reads.push_back(read(*hdr, "rtx_hdr", "peak", s.true_hdr_peak));
    reads.push_back(read(*hdr, "rtx_hdr", "debanding", s.true_hdr_debanding));
  }
  if (const toml::table* audio = section(root, "audio")) {
    reads.push_back(read(*audio, "audio", "delay_ms", s.audio_delay_ms));
    reads.push_back(read(*audio, "audio", "volume", s.volume));
  }
  if (const toml::table* stats = section(root, "stats")) {
    reads.push_back(read(*stats, "stats", "all", s.stats.all));
    reads.push_back(read(*stats, "stats", "chain", s.stats.chain));
    reads.push_back(read(*stats, "stats", "frames", s.stats.frames));
    reads.push_back(read(*stats, "stats", "rife", s.stats.rife));
    reads.push_back(read(*stats, "stats", "tracks", s.stats.tracks));
    reads.push_back(read(*stats, "stats", "output", s.stats.output));
    reads.push_back(read(*stats, "stats", "position", s.stats.position));
    reads.push_back(read(*stats, "stats", "memory", s.stats.memory));
  }
  for (const auto& done : reads) {
    if (!done) return std::unexpected(done.error());
  }
  return validate(s);
}

std::string format_settings(const PlaySettings& s) {
  toml::table root;
  root.insert("version", kSettingsVersion);
  root.insert("image", toml::table{{"upscale", s.upscale},
                                   {"model", s.model},
                                   {"work_height", s.work_height},
                                   {"downscale", std::string(name(s.downscale))}});
  root.insert("interpolation", toml::table{{"rife", s.rife_factor}});
  root.insert("rtx_hdr", toml::table{{"enabled", s.true_hdr},
                                     {"contrast", s.true_hdr_contrast},
                                     {"saturation", s.true_hdr_saturation},
                                     {"middle_gray", s.true_hdr_gray},
                                     {"peak", s.true_hdr_peak},
                                     {"debanding", s.true_hdr_debanding}});
  // Le volume au centième : 0.85 et non 0.8500000238418579.
  root.insert("audio", toml::table{{"delay_ms", s.audio_delay_ms},
                                   {"volume", std::round(static_cast<double>(s.volume) * 100.0) / 100.0}});
  root.insert("stats", toml::table{{"all", s.stats.all},
                                   {"chain", s.stats.chain},
                                   {"frames", s.stats.frames},
                                   {"rife", s.stats.rife},
                                   {"tracks", s.stats.tracks},
                                   {"output", s.stats.output},
                                   {"position", s.stats.position},
                                   {"memory", s.stats.memory}});
  std::ostringstream out;
  out << "# Réglages de « ajp play », écrits par le menu des paramètres.\n" << root << '\n';
  return out.str();
}

std::expected<PlaySettings, Refusal> load_settings(const std::filesystem::path& path) {
  std::error_code error;
  if (!std::filesystem::exists(path, error)) return PlaySettings{};
  std::ifstream file(path, std::ios::binary);
  if (!file) return refuse(RefusalCode::FileUnreadable, path.string());
  std::ostringstream text;
  text << file.rdbuf();
  auto settings = parse_settings(text.str());
  if (!settings)
    return refuse(settings.error().code, std::format("{} : {}", path.string(), settings.error().detail));
  return settings;
}

std::expected<void, Refusal> save_settings(const std::filesystem::path& path, const PlaySettings& settings) {
  if (auto valid = validate(settings); !valid) return std::unexpected(valid.error());
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  std::filesystem::path temp = path;
  temp += ".tmp";
  {
    std::ofstream file(temp, std::ios::binary | std::ios::trunc);
    file << format_settings(settings);
    if (!file.flush()) return refuse(RefusalCode::FileUnwritable, temp.string());
  }
  std::filesystem::rename(temp, path, error);
  if (error)
    return refuse(RefusalCode::FileUnwritable, std::format("{} : {}", path.string(), error.message()));
  return {};
}

PlaySettings settings_for(const PlaySettings& saved, const PlayOptions& options) {
  PlaySettings s = saved;
  if (options.was_given("--no-upscale")) s.upscale = false;
  if (options.was_given("--model") && options.model) s.model = options.model->filename().string();
  if (options.was_given("--work-height")) s.work_height = options.work_height;
  if (options.was_given("--rife")) s.rife_factor = options.rife_factor;
  if (options.was_given("--truehdr")) s.true_hdr = true;
  if (options.was_given("--truehdr-contrast")) s.true_hdr_contrast = options.true_hdr_contrast;
  if (options.was_given("--truehdr-saturation")) s.true_hdr_saturation = options.true_hdr_saturation;
  if (options.was_given("--truehdr-gray")) s.true_hdr_gray = options.true_hdr_gray;
  if (options.was_given("--truehdr-peak")) s.true_hdr_peak = options.true_hdr_peak;
  return s;
}

PlaySettings settings_to_save(const PlaySettings& running, const PlaySettings& from_file,
                              const PlaySettings& started, const PlayOptions& options) {
  PlaySettings out = running;
  const auto keep = [&](std::string_view option, auto member) {
    if (options.was_given(option) && running.*member == started.*member) out.*member = from_file.*member;
  };
  keep("--no-upscale", &PlaySettings::upscale);
  keep("--model", &PlaySettings::model);
  keep("--work-height", &PlaySettings::work_height);
  keep("--rife", &PlaySettings::rife_factor);
  keep("--truehdr", &PlaySettings::true_hdr);
  keep("--truehdr-contrast", &PlaySettings::true_hdr_contrast);
  keep("--truehdr-saturation", &PlaySettings::true_hdr_saturation);
  keep("--truehdr-gray", &PlaySettings::true_hdr_gray);
  keep("--truehdr-peak", &PlaySettings::true_hdr_peak);
  return out;
}

PlayOptions options_for(const PlayOptions& options, const PlaySettings& s) {
  PlayOptions out = options;
  out.upscale = s.upscale;
  // --model donne un chemin complet : il reste tel quel ; sinon le nom choisi dans le dossier des
  // modèles (le lecteur le cherche à côté du modèle par défaut).
  if (!options.was_given("--model")) out.model.reset();
  out.work_height = s.work_height;
  out.rife_factor = s.rife_factor;
  out.true_hdr = s.true_hdr;
  out.true_hdr_contrast = s.true_hdr_contrast;
  out.true_hdr_saturation = s.true_hdr_saturation;
  out.true_hdr_gray = s.true_hdr_gray;
  out.true_hdr_peak = s.true_hdr_peak;
  return out;
}

PlaySettings live_part(const PlaySettings& running, const PlaySettings& wanted, bool rife_loaded) {
  PlaySettings s = running;
  s.downscale = wanted.downscale;
  if (rife_loaded || wanted.rife_factor < 2) s.rife_factor = wanted.rife_factor;
  s.true_hdr_contrast = wanted.true_hdr_contrast;
  s.true_hdr_saturation = wanted.true_hdr_saturation;
  s.true_hdr_gray = wanted.true_hdr_gray;
  s.true_hdr_peak = wanted.true_hdr_peak;
  s.true_hdr_debanding = wanted.true_hdr_debanding;
  s.audio_delay_ms = wanted.audio_delay_ms;
  s.volume = wanted.volume;
  s.stats = wanted.stats;
  return s;
}

Change change_between(const PlaySettings& running, const PlaySettings& wanted, bool rife_loaded) {
  std::vector<std::string_view> reopen;
  if (running.upscale != wanted.upscale || running.model != wanted.model ||
      running.work_height != wanted.work_height)
    reopen.emplace_back("le moteur SPAN est rechargé (construit la première fois pour cette taille)");
  if (running.true_hdr != wanted.true_hdr)
    reopen.emplace_back(wanted.true_hdr ? "RTX HDR s'allume" : "RTX HDR s'éteint");
  if (wanted.rife_factor >= 2 && running.rife_factor < 2 && !rife_loaded)
    reopen.emplace_back("le moteur RIFE est chargé (construit la première fois pour cette taille)");
  if (!reopen.empty()) {
    std::string reason;
    for (const std::string_view part : reopen)
      reason += std::format("{}{}", reason.empty() ? "" : " ; ", part);
    return Change{.cost = ChangeCost::Reopen, .reason = reason + " ; la lecture reprend au même endroit"};
  }
  if (running == wanted) return {};
  return Change{.cost = ChangeCost::Live};
}

}  // namespace ajp::app::play
