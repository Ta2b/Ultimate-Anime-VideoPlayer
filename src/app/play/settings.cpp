#include "app/play/settings.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <format>
#include <fstream>
#include <optional>
#include <span>
#include <sstream>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "app/play/anime4k_choice.h"
#include "app/play/chain_shape.h"
#include "core/utf8_path.h"
#include "render/cube_lut.h"

namespace ajp::app::play {
namespace {

// Le refus d'une clé du fichier (P2.11) : la clé, « audio.delay_ms », dans Refusal::key, que la lecture
// tolérante écarte sans relire le texte ; le détail commence par elle.
std::unexpected<Refusal> refuse_key(RefusalCode code, std::string key, std::string_view rest = {}) {
  std::string detail = key;
  detail += rest;
  return std::unexpected(Refusal{.code = code, .detail = std::move(detail), .key = std::move(key)});
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
    {Downscale::Hermite, "hermite", "Hermite"},
}};

template <class T>
std::expected<void, Refusal> check(std::string_view key, T value, core::Range<T> range) {
  if (range.contains(value)) return {};
  return refuse_key(RefusalCode::FieldOutOfRange, std::string(key),
                    std::format(" : {} hors de {} à {}", value, range.lo, range.hi));
}

// Des balises de langue : lettres et tirets (« fre », « en-US »).
std::expected<void, Refusal> check_languages(std::string_view key,
                                             const std::vector<std::string>& languages) {
  for (const std::string& tag : languages) {
    const bool letters = !tag.empty() && tag.size() <= 16 && std::ranges::all_of(tag, [](char c) {
      return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '-';
    });
    if (!letters)
      return refuse_key(RefusalCode::FieldOutOfRange, std::string(key),
                        std::format(" : « {} » n'est pas une balise de langue", tag));
  }
  return {};
}

bool is_noise(std::string_view token) {
  // b5f48, SPANF3, unshuffle, fp16, 1x3xHxW, dyn-HW, op21, dynamo : la fabrique du modèle.
  if (token.starts_with("SPAN") || token == "unshuffle" || token == "fp16" || token == "rgb" ||
      token == "dynamo" || token.starts_with("dyn-") || token.starts_with("1x3x"))
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
  if (const auto size = anime4k_size(file)) return anime4k_label(*size);
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

std::string_view name(MotionEngine engine) {
  switch (engine) {
    case MotionEngine::Rife:
      return "rife";
    case MotionEngine::DistilDrba:
      return "distil_drba";
    case MotionEngine::Apple:
      return "apple";
  }
  return "rife";
}

std::string_view label(MotionEngine engine) {
  switch (engine) {
    case MotionEngine::Rife:
      return "RIFE";
    case MotionEngine::DistilDrba:
      return "DistilDRBA";
    case MotionEngine::Apple:
      return "Apple (VideoToolbox, macOS 15.4)";
  }
  return "RIFE";
}

std::string_view name(SpanOrder order) {
  switch (order) {
    case SpanOrder::Auto:
      return "auto";
    case SpanOrder::Before:
      return "before";
    case SpanOrder::After:
      return "after";
  }
  return "auto";
}

std::string_view label(SpanOrder order) {
  switch (order) {
    case SpanOrder::Auto:
      return i18n::tr("Auto (after SPAN above 1080p or from ×3)");
    case SpanOrder::Before:
      return i18n::tr("Before SPAN (best quality)");
    case SpanOrder::After:
      return i18n::tr("After SPAN (fastest)");
  }
  return i18n::tr("Auto (after SPAN above 1080p or from ×3)");
}

std::string_view name(FinalScaler scaler) {
  switch (scaler) {
    case FinalScaler::Default:
      return "default";
    case FinalScaler::EwaLanczos:
      return "ewa_lanczos";
    case FinalScaler::EwaLanczosSharp:
      return "ewa_lanczos_sharp";
  }
  return "default";
}

std::string_view label(FinalScaler scaler) {
  switch (scaler) {
    case FinalScaler::Default:
      return i18n::tr("libplacebo default");
    case FinalScaler::EwaLanczos:
      return i18n::tr("EWA Lanczos");
    case FinalScaler::EwaLanczosSharp:
      return i18n::tr("EWA Lanczos sharp");
  }
  return i18n::tr("libplacebo default");
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
      return i18n::tr("Auto (from the display)");
    case OledBlack::On:
      return i18n::tr("On");
    case OledBlack::Off:
      return i18n::tr("Off");
  }
  return i18n::tr("Auto (from the display)");
}

std::string_view name(Downscale filter) {
  return kDownscales[static_cast<std::size_t>(filter)].key;
}

std::string_view label(Downscale filter) {
  return kDownscales[static_cast<std::size_t>(filter)].label;
}

std::string_view badge_text(Badge badge) {
  switch (badge) {
    case Badge::Recommended:
      return i18n::tr("Recommended");
    case Badge::Fastest:
      return i18n::tr("Fastest");
    case Badge::Finest:
      return i18n::tr("Finest");
    case Badge::Experimental:
      return i18n::tr("Experimental");
  }
  return {};
}

std::optional<InterpLevel> interp_level(MotionEngine engine, std::string_view rife_model) {
  for (const InterpChoice& choice : kInterpLevels) {
    if (choice.engine != engine) continue;
    if (engine == MotionEngine::Rife && choice.rife_model != rife_model) continue;
    return choice.level;
  }
  return std::nullopt;
}

std::optional<Badge> badge_of(Downscale filter) {
  switch (filter) {
    case Downscale::Hermite:
      return Badge::Recommended;
    case Downscale::Bilinear:
      return Badge::Fastest;
    case Downscale::Spline36:
      return Badge::Finest;
    default:
      return std::nullopt;
  }
}

std::string resolution_text(int height) {
  return std::format("{}p", height);
}

std::expected<PlaySettings, Refusal> validate(const PlaySettings& s) {
  if (!valid_factor(factor_of(s))) {
    return refuse_key(RefusalCode::FieldOutOfRange, "interpolation.rife",
                      std::format(" : {}/{} hors de 0 à {} (entier ou moitié)", s.rife_factor,
                                  s.rife_factor_den, limits::kRifeFactor.hi));
  }
  const std::array<std::expected<void, Refusal>, 6> checks{
      check("image.work_height", s.work_height, limits::kWorkHeight),
      check("audio.delay_ms", s.audio_delay_ms, limits::kAudioDelayMs),
      check("sync.video_latency_ms", s.video_latency_ms, limits::kVideoLatencyMs),
      check("audio.volume", s.volume, limits::kVolume),
      check("subtitles.scale", s.subtitle_scale, limits::kSubtitleScale),
      check("picture.sharpness", s.sharpness, limits::kSharpness),
  };
  for (const auto& checked : checks) {
    if (!checked) return std::unexpected(checked.error());
  }
  if (s.auto_cap != 0 && (s.auto_cap < kAutoCapMin || s.auto_cap > kAutoCapMax))
    return refuse_key(
        RefusalCode::FieldOutOfRange, "interpolation.auto_cap",
        std::format(" : {} hors de {} à {} (0 : l'écran)", s.auto_cap, kAutoCapMin, kAutoCapMax));
  if (s.language_presets.size() > limits::kLanguagePresets)
    return refuse_key(
        RefusalCode::FieldOutOfRange, "languages.presets",
        std::format(" : {} préréglages, {} au plus", s.language_presets.size(), limits::kLanguagePresets));
  for (const media::LanguagePreset& preset : s.language_presets) {
    if (auto ok = check_languages(
            "languages.audio", preset.audio.empty() ? std::vector<std::string>{} : std::vector{preset.audio});
        !ok)
      return std::unexpected(ok.error());
    if (auto ok =
            check_languages("languages.subtitles", preset.subtitles.empty() ? std::vector<std::string>{}
                                                                            : std::vector{preset.subtitles});
        !ok)
      return std::unexpected(ok.error());
  }
  if (s.true_hdr_peak != 0) {
    if (auto peak = check("hdr.peak", s.true_hdr_peak, limits::kTrueHdrPeak); !peak)
      return std::unexpected(peak.error());
  }
  if (auto grade = render::check_grade_settings(s.grade_values); !grade)
    return std::unexpected(grade.error());
  if (auto steps = check("grade.oled_black_steps", s.oled_black_steps, render::grade_limits::kBlackSteps);
      !steps)
    return std::unexpected(steps.error());
  constexpr core::Range<int> kLookSlider{-100, 100};
  for (const auto& [field, value, range] : {std::tuple{"grade.look_contrast", s.look_contrast, kLookSlider},
                                            {"grade.look_highlight", s.look_highlight, kLookSlider},
                                            {"grade.surface_white", s.surface_white, kSurfaceWhite}}) {
    if (auto in = check(field, value, range); !in) return std::unexpected(in.error());
  }
  for (const auto& [field, value] : {std::pair{"grade.anime_contrast", s.grade_values.anime_contrast},
                                     std::pair{"grade.anime_saturation", s.grade_values.anime_saturation},
                                     std::pair{"grade.anime_highlights", s.grade_values.anime_highlights}}) {
    if (auto in = check(field, value, core::Range<int>{0, 200}); !in) return std::unexpected(in.error());
  }
  // Un dossier d'installation sans retour à la ligne ni octet nul, d'une longueur de chemin (A22-09).
  if (s.engines_dir.size() > kMaxEnginesDirBytes ||
      s.engines_dir.find_first_of(std::string_view("\0\n\r", 3)) != std::string::npos)
    return refuse_key(RefusalCode::FieldOutOfRange, "engines.install_dir", " : pas un chemin de dossier");
  if (s.model.find('/') != std::string::npos || s.model.find('\\') != std::string::npos)
    return refuse_key(RefusalCode::FieldOutOfRange, "image.model",
                      std::format(" : « {} » n'est pas un nom de fichier", s.model));
  return s;
}

PlaySettings settings_for(const PlaySettings& saved, const PlayOptions& options) {
  PlaySettings s = saved;
  if (options.mute) s.volume = 0.0F;
  if (options.was_given("--no-upscale")) s.upscale = false;
  if (options.was_given("--no-restore-lines")) s.restore_lines = false;
  if (options.was_given("--model") && options.model) s.model = core::to_utf8(options.model->filename());
  if (options.was_given("--work-height")) s.work_height = options.work_height;
  // A18-17 : le RIFE chargé est celui que le menu montre et que --bench-out enregistre.
  if (options.was_given("--rife-model") && options.rife_model) {
    s.rife_model = core::to_utf8(options.rife_model->filename());
    s.motion_engine = MotionEngine::Rife;
  }
  if (options.motion)  // A18-09
    s.motion_engine =
        *options.motion == MotionChoice::DistilDrba ? MotionEngine::DistilDrba : MotionEngine::Rife;
  if (options.was_given("--rife")) {
    s.rife_factor = options.rife_factor;
    s.rife_factor_den = options.rife_factor_den;
    s.rife_auto = false;  // un multiplicateur donné à la main coupe le mode auto, comme dans le menu
  }
  if (options.was_given("--truehdr-peak")) s.true_hdr_peak = options.true_hdr_peak;
  if (options.was_given("--anime-contrast")) s.grade_values.anime_contrast = options.anime_contrast;
  if (options.was_given("--anime-saturation")) s.grade_values.anime_saturation = options.anime_saturation;
  if (options.was_given("--anime-highlights")) s.grade_values.anime_highlights = options.anime_highlights;
  return s;
}

PlaySettings settings_to_save(const PlaySettings& running, const PlaySettings& from_file,
                              const PlaySettings& started, const PlayOptions& options) {
  PlaySettings out = running;
  const auto keep = [&](std::string_view option, auto member) {
    if (options.was_given(option) && running.*member == started.*member) out.*member = from_file.*member;
  };
  keep("--mute", &PlaySettings::volume);
  keep("--no-upscale", &PlaySettings::upscale);
  keep("--no-restore-lines", &PlaySettings::restore_lines);
  keep("--model", &PlaySettings::model);
  keep("--work-height", &PlaySettings::work_height);
  keep("--rife-model", &PlaySettings::rife_model);
  keep("--rife-model", &PlaySettings::motion_engine);
  keep("--motion", &PlaySettings::motion_engine);
  keep("--rife", &PlaySettings::rife_factor);
  keep("--rife", &PlaySettings::rife_factor_den);
  keep("--rife", &PlaySettings::rife_auto);  // l'option la coupe pour cette lecture seulement (A22-05)
  keep("--truehdr-peak", &PlaySettings::true_hdr_peak);
  // Les peps de l'anime vivent dans grade_values : une option donnée à la main n'est pas enregistrée.
  const auto keep_grade = [&](std::string_view option, int render::GradeSettings::* member) {
    if (options.was_given(option) && running.grade_values.*member == started.grade_values.*member)
      out.grade_values.*member = from_file.grade_values.*member;
  };
  keep_grade("--anime-contrast", &render::GradeSettings::anime_contrast);
  keep_grade("--anime-saturation", &render::GradeSettings::anime_saturation);
  keep_grade("--anime-highlights", &render::GradeSettings::anime_highlights);
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
  out.true_hdr_peak = s.true_hdr_peak;
  return out;
}

PlaySettings live_part(const PlaySettings& running, const PlaySettings& wanted, bool rife_loaded,
                       const OpenedChain* opened) {
  PlaySettings s = running;
  s.preset = wanted.preset;
  s.priority = wanted.priority;
  s.downscale = wanted.downscale;
  s.rife_auto = wanted.rife_auto;
  s.auto_cap = wanted.auto_cap;
  s.auto_free = wanted.auto_free;
  if (!wanted.upscale) s.span_precision = wanted.span_precision;  // sans SPAN, rien à recharger
  if (rife_loaded || wanted.rife_factor < 2) {
    s.rife_factor = wanted.rife_factor;
    s.rife_factor_den = wanted.rife_factor_den;
  }
  if (wanted.rife_factor < 2) {  // sans interpolation, rien à recharger
    s.motion_engine = wanted.motion_engine;
    s.rife_model = wanted.rife_model;
    s.span_order = wanted.span_order;
    s.motion_precision = wanted.motion_precision;
    s.rife_fused =
        wanted.rife_fused;  // sans interpolation, rien à recharger : le réglage s'enregistre (A22-01)
    s.drba_fused = wanted.drba_fused;
  }
  // Le modèle RIFE ne sert pas à un autre moteur du mouvement : le changer ne recharge rien.
  if (running.motion_engine == wanted.motion_engine && wanted.motion_engine != MotionEngine::Rife)
    s.rife_model = wanted.rife_model;
  s.true_hdr_peak = wanted.true_hdr_peak;  // hdr_native reste celui de la sortie (il suit le système)
  s.final_scaler = wanted.final_scaler;
  s.deband = wanted.deband;
  s.sharpness = wanted.sharpness;
  s.autoplay_next = wanted.autoplay_next;
  s.engines_dir = wanted.engines_dir;
  s.gpu =
      wanted.gpu;  // vaut au prochain lancement (le panneau le dit) : il s'enregistre tout de suite (A22-01)
  s.update_mode = wanted.update_mode;
  s.update_frequency = wanted.update_frequency;
  s.language = wanted.language;
  s.audio_delay_ms = wanted.audio_delay_ms;
  s.display_sync = wanted.display_sync;
  s.video_latency_ms = wanted.video_latency_ms;
  s.volume = wanted.volume;
  s.subtitle_scale = wanted.subtitle_scale;
  s.subtitle_delay_ms = wanted.subtitle_delay_ms;
  s.language_presets = wanted.language_presets;  // au prochain fichier
  s.grade = wanted.grade;
  s.grade_values = wanted.grade_values;
  s.look_contrast = wanted.look_contrast;
  s.look_highlight = wanted.look_highlight;
  s.surface_white = wanted.surface_white;
  s.flash_guard = wanted.flash_guard;
  s.oled_black = wanted.oled_black;
  s.oled_black_steps = wanted.oled_black_steps;
  s.keep_poses = wanted.keep_poses;
  s.smooth_motion = wanted.smooth_motion;  // gardé pour le prochain lancement
  s.oled_care = wanted.oled_care;
  s.steady_cadence = wanted.steady_cadence;
  s.stats = wanted.stats;
  // Ce qui demande un autre anneau ou un autre ordre du warp attend la réouverture, d'un bloc : la cadence
  // en cours reste celle pour laquelle la chaîne a été ouverte.
  if (opened != nullptr && !fits(*opened, s)) {
    s.rife_auto = running.rife_auto;
    s.auto_cap = running.auto_cap;
    s.auto_free = running.auto_free;
    s.rife_factor = running.rife_factor;
    s.rife_factor_den = running.rife_factor_den;
  }
  return s;
}

PlaySettings follow_running(const PlaySettings& draft, const PlaySettings& seen, const PlaySettings& running,
                            bool rife_loaded) {
  PlaySettings s = live_part(draft, running, rife_loaded);
  // Les réglages que live_part ne prend que selon l'état de running (interpolation coupée, SPAN éteint).
  const auto keep = [&](auto member) {
    if (running.*member == seen.*member) s.*member = draft.*member;
  };
  keep(&PlaySettings::rife_factor);
  keep(&PlaySettings::rife_factor_den);
  keep(&PlaySettings::rife_auto);
  keep(&PlaySettings::auto_cap);
  keep(&PlaySettings::auto_free);
  keep(&PlaySettings::motion_engine);
  keep(&PlaySettings::rife_model);
  keep(&PlaySettings::span_order);
  keep(&PlaySettings::motion_precision);
  keep(&PlaySettings::rife_fused);
  keep(&PlaySettings::drba_fused);
  keep(&PlaySettings::span_precision);
  return s;
}

namespace {

// La forme tirée à l'ouverture (A08-03, A22-04) : ce que wanted demande et que la chaîne ouverte n'a pas.
// L'ordre du warp n'est dit que si le réglage span_order ne le dit pas déjà. opened nullptr : aucune chaîne.
void add_shape_reasons(std::vector<std::string>& reopen, const PlaySettings& running,
                       const PlaySettings& wanted, const OpenedChain* opened) {
  if (opened == nullptr || fits(*opened, wanted)) return;
  const ChainShape shape = chain_shape(wanted, *opened);
  if (shape.ring > opened->shape.ring)
    reopen.emplace_back(i18n::tr("the frame ring grows for this interpolation"));
  if (shape.warp_after_span != opened->shape.warp_after_span && running.span_order == wanted.span_order)
    reopen.emplace_back(shape.warp_after_span ? i18n::tr("warp runs after SPAN")
                                              : i18n::tr("warp runs before SPAN"));
}

// Ce que recharge un agrandissement changé : SPAN, ou Anime4K, qui n'a rien à construire.
std::string upscale_reason(const PlaySettings& wanted) {
  if (wanted.upscale && anime4k_size(wanted.model)) return i18n::tr("the upscaler changes to Anime4K");
  return i18n::tr("SPAN reloads (built the first time for this size)");
}

// La réouverture et ses raisons, jointes pour l'encadré du panneau.
Change reopen_for(const std::vector<std::string>& reasons) {
  std::string reason;
  for (const std::string& part : reasons) reason += std::format("{}{}", reason.empty() ? "" : " ; ", part);
  return Change{.cost = ChangeCost::Reopen,
                .reason = reason + i18n::tr(" ; playback resumes at the same spot")};
}

}  // namespace

Change change_between(const PlaySettings& running, const PlaySettings& wanted, bool rife_loaded,
                      const OpenedChain* opened) {
  std::vector<std::string> reopen;
  if (running.upscale != wanted.upscale || running.model != wanted.model ||
      running.work_height != wanted.work_height || running.restore_lines != wanted.restore_lines)
    reopen.emplace_back(upscale_reason(wanted));
  if (running.audio_device != wanted.audio_device) reopen.emplace_back(i18n::tr("the output device changes"));
  if (wanted.rife_factor >= 2 && running.rife_factor < 2 && !rife_loaded)
    reopen.emplace_back(i18n::tr("the interpolation engine loads (built the first time for this size)"));
  if (wanted.rife_factor >= 2 && running.motion_engine != wanted.motion_engine)
    reopen.emplace_back(i18n::tr("the interpolation engine changes"));
  if (wanted.rife_factor >= 2 && wanted.motion_engine == MotionEngine::Rife &&
      running.rife_fused != wanted.rife_fused)
    reopen.emplace_back(wanted.rife_fused
                            ? i18n::tr("fused RIFE (six engines, built the first time for this size)")
                            : i18n::tr("whole RIFE instead of fused RIFE"));
  if (wanted.rife_factor >= 2 && wanted.motion_engine == MotionEngine::DistilDrba &&
      running.drba_fused != wanted.drba_fused)
    reopen.emplace_back(wanted.drba_fused
                            ? i18n::tr("fused DistilDRBA (four engines, built the first time for this size)")
                            : i18n::tr("whole DistilDRBA instead of fused DistilDRBA"));
  if (wanted.rife_factor >= 2 && wanted.motion_engine == MotionEngine::Rife &&
      running.rife_model != wanted.rife_model)
    reopen.emplace_back(i18n::tr("the RIFE model loads (built the first time for this size)"));
  if (wanted.rife_factor >= 2 && running.span_order != wanted.span_order) {
    reopen.emplace_back(wanted.span_order == SpanOrder::Before  ? i18n::tr("warp runs before SPAN")
                        : wanted.span_order == SpanOrder::After ? i18n::tr("warp runs after SPAN")
                                                                : i18n::tr("warp order follows the output"));
  }
  if (wanted.upscale && running.span_precision != wanted.span_precision) {
    reopen.emplace_back(i18n::tr_format("SPAN switches to {} (built the first time for this size)",
                                        precision_text(wanted.span_precision)));
  }
  if (wanted.rife_factor >= 2 && running.motion_engine == wanted.motion_engine &&
      running.motion_precision != wanted.motion_precision) {
    reopen.emplace_back(i18n::tr_format("{} switches to {} (built the first time for this size)",
                                        label(wanted.motion_engine),
                                        precision_text(wanted.motion_precision)));
  }
  add_shape_reasons(reopen, running, wanted, opened);
  if (!reopen.empty()) return reopen_for(reopen);
  if (running == wanted) return {};
  return Change{.cost = ChangeCost::Live};
}

std::optional<std::int64_t> audio_fallback_delay_us(const PlaySettings& settings) {
  if (settings.audio_device.empty()) return std::nullopt;
  const auto known = settings.audio_device_delays_ms.find("");
  return std::int64_t{known != settings.audio_device_delays_ms.end() ? known->second : 0} * 1000;
}

}  // namespace ajp::app::play
