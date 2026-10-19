#include "app/play/settings.h"

#include <array>
#include <cctype>
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
// Dans l'ordre des valeurs de core::DownscaleFilter.
constexpr std::array<DownscaleName, kDownscaleCount> kDownscales{{
    {Downscale::Area, "area", "Area"},
    {Downscale::Spline36, "spline36", "Spline36"},
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
  } else if constexpr (std::is_same_v<T, double>) {
    if (auto v = node->value<double>()) {
      out = *v;
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

bool is_noise(std::string_view token) {
  // b5f48, SPANF3, unshuffle, fp16, 1x3xHxW, dyn-HW, op21, dynamo : la fabrique du modèle.
  if (token.starts_with("SPAN") || token == "unshuffle" || token == "fp16" || token == "dynamo" ||
      token.starts_with("dyn-") || token.starts_with("1x3x"))
    return true;
  if (token.size() >= 3 && token.starts_with("op") && std::isdigit(static_cast<unsigned char>(token[2])))
    return true;
  if (token.size() >= 4 && token[0] == 'b' && std::isdigit(static_cast<unsigned char>(token[1])) &&
      token.find('f') != std::string_view::npos)
    return true;
  return false;
}

}  // namespace

std::string model_label(const std::string& file) {
  std::string_view name = file;
  if (name.ends_with(".onnx")) name.remove_suffix(5);
  if (name.starts_with("2x_")) name.remove_prefix(3);
  if (name.starts_with("AnimeJaNai_")) name.remove_prefix(11);
  std::string out;
  while (!name.empty()) {
    const std::size_t cut = name.find('_');
    const std::string_view token = name.substr(0, cut);
    if (!is_noise(token)) {
      std::string word(token);
      // « V3.1Sharp1 » : « V3.1 Sharp1 ».
      if (const std::size_t sharp = word.find("Sharp"); sharp != std::string::npos && sharp > 0)
        word.insert(sharp, " ");
      out += out.empty() ? word : " " + word;
    }
    if (cut == std::string_view::npos) break;
    name.remove_prefix(cut + 1);
  }
  return out.empty() ? file : out;
}

std::string_view name(OledBlack mode) {
  switch (mode) {
    case OledBlack::Auto:
      return "auto";
    case OledBlack::On:
      return "on";
    case OledBlack::Off:
      return "off";
  }
  return "auto";
}

std::string_view label(OledBlack mode) {
  switch (mode) {
    case OledBlack::Auto:
      return "Auto (selon l'écran)";
    case OledBlack::On:
      return "Activé";
    case OledBlack::Off:
      return "Désactivé";
  }
  return "Auto (selon l'écran)";
}

render::GradeSettings with_preset(render::GradePreset preset, const render::GradeSettings& current) {
  render::GradeSettings out = render::grade_preset(preset, current);
  out.gamut_p3 = current.gamut_p3;
  return out;
}

std::string_view name(Downscale filter) {
  return kDownscales[static_cast<std::size_t>(filter)].key;
}

std::string_view label(Downscale filter) {
  return kDownscales[static_cast<std::size_t>(filter)].label;
}

std::expected<PlaySettings, Refusal> validate(const PlaySettings& s) {
  const std::array<std::expected<void, Refusal>, 8> checks{
      check("image.work_height", s.work_height, limits::kWorkHeight),
      check("interpolation.rife", s.rife_factor, limits::kRifeFactor),
      check("rtx_hdr.contrast", s.true_hdr_contrast, limits::kTrueHdrContrast),
      check("rtx_hdr.saturation", s.true_hdr_saturation, limits::kTrueHdrSaturation),
      check("rtx_hdr.middle_gray", s.true_hdr_gray, limits::kTrueHdrGray),
      check("audio.delay_ms", s.audio_delay_ms, limits::kAudioDelayMs),
      check("audio.volume", s.volume, limits::kVolume),
      check("subtitles.scale", s.subtitle_scale, limits::kSubtitleScale),
  };
  for (const auto& checked : checks) {
    if (!checked) return std::unexpected(checked.error());
  }
  if (s.true_hdr_peak != 0) {
    if (auto peak = check("rtx_hdr.peak", s.true_hdr_peak, limits::kTrueHdrPeak); !peak)
      return std::unexpected(peak.error());
  }
  if (auto grade = render::check_grade_settings(s.grade_values); !grade)
    return std::unexpected(grade.error());
  if (auto steps = check("grade.oled_black_steps", s.oled_black_steps, render::grade_limits::kBlackSteps);
      !steps)
    return std::unexpected(steps.error());
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
  if (const toml::table* interpolation = section(root, "interpolation")) {
    reads.push_back(read(*interpolation, "interpolation", "rife", s.rife_factor));
    reads.push_back(read(*interpolation, "interpolation", "smooth_motion", s.smooth_motion));
  }
  if (const toml::table* hdr = section(root, "rtx_hdr")) {
    reads.push_back(read(*hdr, "rtx_hdr", "enabled", s.true_hdr));
    reads.push_back(read(*hdr, "rtx_hdr", "contrast", s.true_hdr_contrast));
    reads.push_back(read(*hdr, "rtx_hdr", "saturation", s.true_hdr_saturation));
    reads.push_back(read(*hdr, "rtx_hdr", "middle_gray", s.true_hdr_gray));
    reads.push_back(read(*hdr, "rtx_hdr", "peak", s.true_hdr_peak));
    reads.push_back(read(*hdr, "rtx_hdr", "debanding", s.true_hdr_debanding));
    reads.push_back(read(*hdr, "rtx_hdr", "native", s.hdr_native));
  }
  if (const toml::table* audio = section(root, "audio")) {
    reads.push_back(read(*audio, "audio", "delay_ms", s.audio_delay_ms));
    reads.push_back(read(*audio, "audio", "volume", s.volume));
  }
  if (const toml::table* subtitles = section(root, "subtitles"))
    reads.push_back(read(*subtitles, "subtitles", "scale", s.subtitle_scale));
  if (const toml::table* oled = section(root, "oled")) {
    reads.push_back(read(*oled, "oled", "care", s.oled_care));
    reads.push_back(read(*oled, "oled", "steady_cadence", s.steady_cadence));
  }
  if (const toml::table* grade = section(root, "grade")) {
    std::string preset(render::preset_key(s.grade_preset));
    render::GradeSettings& v = s.grade_values;
    reads.push_back(read(*grade, "grade", "enabled", s.grade));
    reads.push_back(read(*grade, "grade", "preset", preset));
    reads.push_back(read(*grade, "grade", "shadows", v.shadows));
    reads.push_back(read(*grade, "grade", "highlights", v.highlights));
    reads.push_back(read(*grade, "grade", "highlights_rtx", v.highlights_rtx));
    reads.push_back(read(*grade, "grade", "vibrance", v.vibrance));
    reads.push_back(read(*grade, "grade", "vibrance_rtx", v.vibrance_rtx));
    reads.push_back(read(*grade, "grade", "area_rtx", v.area_rtx));
    reads.push_back(read(*grade, "grade", "guard", v.guard));
    reads.push_back(read(*grade, "grade", "expand", v.expand));
    reads.push_back(read(*grade, "grade", "gamut_p3", v.gamut_p3));
    std::string oled(name(s.oled_black));
    reads.push_back(read(*grade, "grade", "oled_black", oled));
    reads.push_back(read(*grade, "grade", "oled_black_steps", s.oled_black_steps));
    bool oled_known = false;
    for (int i = 0; i < kOledBlackCount; ++i) {
      if (name(static_cast<OledBlack>(i)) == oled) {
        s.oled_black = static_cast<OledBlack>(i);
        oled_known = true;
      }
    }
    if (!oled_known)
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("grade.oled_black : « {} »", oled)));
    bool known = false;
    for (int i = 0; i < render::kGradePresetCount; ++i) {
      const auto candidate = static_cast<render::GradePreset>(i);
      if (render::preset_key(candidate) == preset) {
        s.grade_preset = candidate;
        known = true;
      }
    }
    if (!known)
      reads.emplace_back(refuse(RefusalCode::FieldOutOfRange, std::format("grade.preset : « {} »", preset)));
    s.grade_values = with_preset(s.grade_preset, v);
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
  root.insert("interpolation", toml::table{{"rife", s.rife_factor}, {"smooth_motion", s.smooth_motion}});
  root.insert("rtx_hdr", toml::table{{"enabled", s.true_hdr},
                                     {"contrast", s.true_hdr_contrast},
                                     {"saturation", s.true_hdr_saturation},
                                     {"middle_gray", s.true_hdr_gray},
                                     {"peak", s.true_hdr_peak},
                                     {"debanding", s.true_hdr_debanding},
                                     {"native", s.hdr_native}});
  // Le volume au centième : 0.85 et non 0.8500000238418579.
  root.insert("audio", toml::table{{"delay_ms", s.audio_delay_ms},
                                   {"volume", std::round(static_cast<double>(s.volume) * 100.0) / 100.0}});
  root.insert("subtitles",
              toml::table{{"scale", std::round(static_cast<double>(s.subtitle_scale) * 100.0) / 100.0}});
  // Les réglages du grade au centième, comme le menu les montre.
  const auto hundredth = [](double v) { return std::round(v * 100.0) / 100.0; };
  const render::GradeSettings& g = s.grade_values;
  root.insert("grade", toml::table{{"enabled", s.grade},
                                   {"preset", std::string(render::preset_key(s.grade_preset))},
                                   {"shadows", hundredth(g.shadows)},
                                   {"highlights", hundredth(g.highlights)},
                                   {"highlights_rtx", hundredth(g.highlights_rtx)},
                                   {"vibrance", hundredth(g.vibrance)},
                                   {"vibrance_rtx", hundredth(g.vibrance_rtx)},
                                   {"area_rtx", hundredth(g.area_rtx)},
                                   {"guard", hundredth(g.guard)},
                                   {"expand", hundredth(g.expand)},
                                   {"gamut_p3", g.gamut_p3},
                                   {"oled_black", std::string(name(s.oled_black))},
                                   {"oled_black_steps", hundredth(s.oled_black_steps)}});
  root.insert("oled", toml::table{{"care", s.oled_care}, {"steady_cadence", s.steady_cadence}});
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
  s.hdr_native = wanted.hdr_native;
  s.audio_delay_ms = wanted.audio_delay_ms;
  s.volume = wanted.volume;
  s.subtitle_scale = wanted.subtitle_scale;
  s.grade = wanted.grade;
  s.grade_preset = wanted.grade_preset;
  s.grade_values = wanted.grade_values;
  s.oled_black = wanted.oled_black;
  s.oled_black_steps = wanted.oled_black_steps;
  s.smooth_motion = wanted.smooth_motion;  // gardé pour le prochain lancement
  s.oled_care = wanted.oled_care;
  s.steady_cadence = wanted.steady_cadence;
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
