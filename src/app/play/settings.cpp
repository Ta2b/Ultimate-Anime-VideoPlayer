#include "app/play/settings.h"

#include <algorithm>
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

namespace ajp::app::play {
namespace {

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

// Des balises de langue : lettres et tirets (« fre », « en-US »), kMaxLanguages au plus.
std::expected<void, Refusal> check_languages(std::string_view key,
                                             const std::vector<std::string>& languages) {
  if (languages.size() > limits::kMaxLanguages)
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("{} : {} langues, {} au plus", key, languages.size(), limits::kMaxLanguages));
  for (const std::string& tag : languages) {
    const bool letters = !tag.empty() && tag.size() <= 16 && std::ranges::all_of(tag, [](char c) {
      return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '-';
    });
    if (!letters)
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("{} : « {} » n'est pas une balise de langue", key, tag));
  }
  return {};
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

std::string rife_label(const std::string& file) {
  std::string_view name = file;
  if (name.ends_with(".onnx")) name.remove_suffix(5);
  if (!name.starts_with("rife_v")) return file;
  name.remove_prefix(6);
  std::string out = "RIFE ";
  for (const char c : name) out += c == '_' ? ' ' : c;
  return out;
}

std::string_view hdr_engine_key(const PlaySettings& s) {
  return s.true_hdr ? "rtx" : (s.hdr_native ? "adaptive" : "sdr");
}

std::string_view name(MotionEngine engine) {
  switch (engine) {
    case MotionEngine::Rife:
      return "rife";
    case MotionEngine::Gmfss:
      return "gmfss";
    case MotionEngine::GmfssUnion:
      return "gmfss_union";
    case MotionEngine::DistilDrba:
      return "distil_drba";
  }
  return "rife";
}

std::string_view label(MotionEngine engine) {
  switch (engine) {
    case MotionEngine::Rife:
      return "RIFE";
    case MotionEngine::Gmfss:
      return "GMFSS Fortuna (le plus fin, le plus lourd)";
    case MotionEngine::GmfssUnion:
      return "GMFSS Fortuna Union (GMFSS mêlé à RIFE)";
    case MotionEngine::DistilDrba:
      return "DistilDRBA (poses gardées par le réseau)";
  }
  return "RIFE";
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
  if (!valid_factor(factor_of(s))) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("interpolation.rife : {}/{} hors de 0 à {} (entier ou moitié)", s.rife_factor,
                              s.rife_factor_den, limits::kRifeFactor.hi));
  }
  const std::array<std::expected<void, Refusal>, 8> checks{
      check("image.work_height", s.work_height, limits::kWorkHeight),
      check("hdr.contrast", s.true_hdr_contrast, limits::kTrueHdrContrast),
      check("hdr.saturation", s.true_hdr_saturation, limits::kTrueHdrSaturation),
      check("hdr.middle_gray", s.true_hdr_gray, limits::kTrueHdrGray),
      check("audio.delay_ms", s.audio_delay_ms, limits::kAudioDelayMs),
      check("sync.video_latency_ms", s.video_latency_ms, limits::kVideoLatencyMs),
      check("audio.volume", s.volume, limits::kVolume),
      check("subtitles.scale", s.subtitle_scale, limits::kSubtitleScale),
  };
  for (const auto& checked : checks) {
    if (!checked) return std::unexpected(checked.error());
  }
  if (auto ok = check_languages("audio.languages", s.audio_languages); !ok)
    return std::unexpected(ok.error());
  if (auto ok = check_languages("subtitles.languages", s.subtitle_languages); !ok)
    return std::unexpected(ok.error());
  if (s.true_hdr_peak != 0) {
    if (auto peak = check("hdr.peak", s.true_hdr_peak, limits::kTrueHdrPeak); !peak)
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

PlaySettings settings_for(const PlaySettings& saved, const PlayOptions& options) {
  PlaySettings s = saved;
  if (options.was_given("--no-upscale")) s.upscale = false;
  if (options.was_given("--model") && options.model) s.model = options.model->filename().string();
  if (options.was_given("--work-height")) s.work_height = options.work_height;
  if (options.was_given("--rife")) {
    s.rife_factor = options.rife_factor;
    s.rife_factor_den = options.rife_factor_den;
  }
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
  keep("--rife", &PlaySettings::rife_factor_den);
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
  out.rife_factor_den = s.rife_factor_den;
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
  if (rife_loaded || wanted.rife_factor < 2) {
    s.rife_factor = wanted.rife_factor;
    s.rife_factor_den = wanted.rife_factor_den;
  }
  if (wanted.rife_factor < 2) {  // sans interpolation, rien à recharger
    s.motion_engine = wanted.motion_engine;
    s.rife_model = wanted.rife_model;
    s.warp_after_span = wanted.warp_after_span;
  }
  // Le modèle RIFE ne sert pas à un autre moteur du mouvement : le changer ne recharge rien.
  if (running.motion_engine == wanted.motion_engine && wanted.motion_engine != MotionEngine::Rife)
    s.rife_model = wanted.rife_model;
  s.true_hdr_contrast = wanted.true_hdr_contrast;
  s.true_hdr_saturation = wanted.true_hdr_saturation;
  s.true_hdr_gray = wanted.true_hdr_gray;
  s.true_hdr_peak = wanted.true_hdr_peak;
  s.true_hdr_debanding = wanted.true_hdr_debanding;
  s.hdr_native = wanted.hdr_native;
  s.audio_delay_ms = wanted.audio_delay_ms;
  s.display_sync = wanted.display_sync;
  s.video_latency_ms = wanted.video_latency_ms;
  s.volume = wanted.volume;
  s.subtitle_scale = wanted.subtitle_scale;
  s.audio_languages = wanted.audio_languages;  // au prochain fichier
  s.subtitle_languages = wanted.subtitle_languages;
  s.grade = wanted.grade;
  s.grade_preset = wanted.grade_preset;
  s.grade_values = wanted.grade_values;
  s.oled_black = wanted.oled_black;
  s.oled_black_steps = wanted.oled_black_steps;
  s.keep_poses = wanted.keep_poses;
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
    reopen.emplace_back(
        "le moteur d'interpolation est chargé (construit la première fois pour cette taille)");
  if (wanted.rife_factor >= 2 && running.motion_engine != wanted.motion_engine)
    reopen.emplace_back("le moteur du mouvement change");
  if (wanted.rife_factor >= 2 && wanted.motion_engine == MotionEngine::Rife &&
      running.rife_model != wanted.rife_model)
    reopen.emplace_back("le modèle RIFE est chargé (construit la première fois pour cette taille)");
  if (wanted.rife_factor >= 2 && running.warp_after_span != wanted.warp_after_span)
    reopen.emplace_back(wanted.warp_after_span ? "le warp passe après SPAN" : "le warp repasse avant SPAN");
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