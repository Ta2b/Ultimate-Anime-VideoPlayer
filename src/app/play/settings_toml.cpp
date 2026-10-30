// Le fichier play.toml des réglages : lecture (parse_settings, load_settings), écriture
// (format_settings, save_settings). Les valeurs lues passent par validate() (settings.cpp).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "app/play/auto_config.h"
#include "app/play/settings.h"
#include "core/log.h"
#include "core/user_overlay.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++-ajp/toml.hpp>

namespace ajp::app::play {
namespace {

constexpr std::int64_t kSettingsVersion = 1;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
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

// Un cran de précision (precision.h) : sa clé, ou FieldOutOfRange pour un autre texte.
std::expected<void, Refusal> read_precision(const toml::table& table, std::string_view section,
                                            std::string_view key, PrecisionTier& out) {
  std::string text(name(out));
  if (auto ok = read(table, section, key, text); !ok) return ok;
  for (int i = 0; i < kPrecisionTierCount; ++i) {
    if (name(static_cast<PrecisionTier>(i)) == text) {
      out = static_cast<PrecisionTier>(i);
      return {};
    }
  }
  return refuse(RefusalCode::FieldOutOfRange, std::format("{}.{} : « {} »", section, key, text));
}

// L'ancienne liste de langues : un tableau de balises (« ["fre", "eng"] ») ; FieldType sinon.
std::expected<void, Refusal> read_legacy_languages(const toml::table& table, std::string_view section,
                                                   std::vector<std::string>& out) {
  const toml::node* node = table.get("languages");
  if (node == nullptr) return {};
  const toml::array* list = node->as_array();
  if (list == nullptr) return refuse(RefusalCode::FieldType, std::format("{}.languages", section));
  std::vector<std::string> languages;
  for (const toml::node& item : *list) {
    const std::optional<std::string> tag = item.value_exact<std::string>();
    if (!tag) return refuse(RefusalCode::FieldType, std::format("{}.languages", section));
    languages.push_back(*tag);
  }
  out = std::move(languages);
  return {};
}

// interpolation.rife : un entier (×3), ou un réel en moitié (×1.5).
std::expected<void, Refusal> read_factor(const toml::table& table, PlaySettings& s) {
  const toml::node* node = table.get("rife");
  if (node == nullptr) return {};
  if (auto whole = node->value_exact<std::int64_t>(); whole && *whole >= INT32_MIN && *whole <= INT32_MAX) {
    s.rife_factor = static_cast<int>(*whole);
    s.rife_factor_den = 1;
    return {};
  }
  if (auto real = node->value_exact<double>()) {
    const double halves = *real * 2.0;
    if (halves != std::floor(halves) || halves < 0.0 || halves > 1000.0) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("interpolation.rife : {} n'est pas une moitié", *real));
    }
    const int count = static_cast<int>(halves);
    s.rife_factor = count % 2 == 0 ? count / 2 : count;
    s.rife_factor_den = count % 2 == 0 ? 1 : 2;
    return {};
  }
  return refuse(RefusalCode::FieldType, "interpolation.rife");
}

const toml::table* section(const toml::table& root, std::string_view name) {
  const toml::node* node = root.get(name);
  return node != nullptr ? node->as_table() : nullptr;
}

}  // namespace

namespace {

std::expected<PlaySettings, Refusal> settings_from_table(const toml::table& root);

}  // namespace

std::expected<PlaySettings, Refusal> parse_settings(std::string_view toml_text) {
  toml::parse_result parsed = toml::parse(toml_text);
  if (!parsed) {
    return refuse(RefusalCode::ProfileUnreadable,
                  std::format("play.toml ligne {} : {}", parsed.error().source().begin.line,
                              parsed.error().description()));
  }
  return settings_from_table(parsed.table());
}

namespace {

std::expected<PlaySettings, Refusal> settings_from_table(const toml::table& root) {
  PlaySettings s;
  std::vector<std::string> legacy_audio;
  std::vector<std::string> legacy_subtitles;
  std::vector<std::expected<void, Refusal>> reads;
  if (const toml::table* automatic = section(root, "auto")) {
    std::string preset(key(s.preset));
    std::string priority(key(s.priority));
    reads.push_back(read(*automatic, "auto", "preset", preset));
    reads.push_back(read(*automatic, "auto", "priority", priority));
    if (const auto found = preset_from_key(preset)) {
      s.preset = *found;
    } else {
      reads.emplace_back(refuse(RefusalCode::FieldOutOfRange, std::format("auto.preset : « {} »", preset)));
    }
    if (const auto found = priority_from_key(priority)) {
      s.priority = *found;
    } else {
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("auto.priority : « {} »", priority)));
    }
  }
  if (const toml::table* image = section(root, "image")) {
    std::string downscale(name(s.downscale));
    reads.push_back(read(*image, "image", "upscale", s.upscale));
    reads.push_back(read(*image, "image", "model", s.model));
    reads.push_back(read(*image, "image", "work_height", s.work_height));
    reads.push_back(read(*image, "image", "downscale", downscale));
    reads.push_back(read_precision(*image, "image", "span_precision", s.span_precision));
    bool known = false;
    for (int i = 0; i < kDownscaleCount; ++i) {
      const auto filter = static_cast<Downscale>(i);
      if (name(filter) == downscale) {
        s.downscale = filter;
        known = true;
      }
    }
    if (!known)
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("image.downscale : « {} »", downscale)));
  }
  if (const toml::table* interpolation = section(root, "interpolation")) {
    reads.push_back(read_factor(*interpolation, s));
    reads.push_back(read(*interpolation, "interpolation", "smooth_motion", s.smooth_motion));
    reads.push_back(read(*interpolation, "interpolation", "model", s.rife_model));
    reads.push_back(read(*interpolation, "interpolation", "keep_poses", s.keep_poses));
    reads.push_back(read(*interpolation, "interpolation", "rife_fused", s.rife_fused));
    reads.push_back(read(*interpolation, "interpolation", "drba_fused", s.drba_fused));
    reads.push_back(read(*interpolation, "interpolation", "gmfss_fused", s.gmfss_fused));
    // gmfss_quality (faux : l'ancien Performance, devenu Équilibre ; vrai : Qualité), puis les crans.
    bool legacy_quality = false;
    reads.push_back(read(*interpolation, "interpolation", "gmfss_quality", legacy_quality));
    if (legacy_quality)
      s.gmfss_precision = PrecisionTier::Quality;
    else if (interpolation->contains("gmfss_quality"))
      s.gmfss_precision = PrecisionTier::Balanced;  // faux : l'ancien Performance, FP16
    reads.push_back(read_precision(*interpolation, "interpolation", "gmfss_precision", s.gmfss_precision));
    reads.push_back(read_precision(*interpolation, "interpolation", "motion_precision", s.motion_precision));
    reads.push_back(read(*interpolation, "interpolation", "auto", s.rife_auto));
    reads.push_back(read(*interpolation, "interpolation", "auto_cap", s.auto_cap));
    std::string auto_mode = s.auto_free ? "libre" : "recommande";
    reads.push_back(read(*interpolation, "interpolation", "auto_mode", auto_mode));
    if (auto_mode == "libre" || auto_mode == "recommande") {
      s.auto_free = auto_mode == "libre";
    } else {
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange,
                 std::format("interpolation.auto_mode : « {} » (recommande ou libre)", auto_mode)));
    }
    // span_order ; les fichiers d'avant portent warp_after_span, qui ne compte que s'il est vrai (faux
    // était le défaut, écrit partout : il laisse Auto).
    bool legacy_after = false;
    reads.push_back(read(*interpolation, "interpolation", "warp_after_span", legacy_after));
    if (legacy_after) s.span_order = SpanOrder::After;
    std::string order(name(s.span_order));
    reads.push_back(read(*interpolation, "interpolation", "span_order", order));
    bool order_known = false;
    for (int i = 0; i < kSpanOrderCount; ++i) {
      if (name(static_cast<SpanOrder>(i)) == order) {
        s.span_order = static_cast<SpanOrder>(i);
        order_known = true;
      }
    }
    if (!order_known)
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("interpolation.span_order : « {} »", order)));
    std::string engine(name(s.motion_engine));
    reads.push_back(read(*interpolation, "interpolation", "engine", engine));
    // Le flux optique matériel des RTX a été retiré : RIFE le remplace, sans plainte.
    bool engine_known = engine == "optical_flow";
    if (engine_known) s.motion_engine = MotionEngine::Rife;
    for (int i = 0; i < kMotionEngineCount; ++i) {
      if (name(static_cast<MotionEngine>(i)) == engine) {
        s.motion_engine = static_cast<MotionEngine>(i);
        engine_known = true;
      }
    }
    if (!engine_known)
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("interpolation.engine : « {} »", engine)));
  }
  // Le mode HDR de l'application : son moteur (RTX HDR sur NVIDIA, HDR adaptatif partout, ou SDR)
  // et les réglages qu'ils partagent. Les fichiers d'avant (section rtx_hdr) se lisent encore.
  if (const toml::table* hdr = section(root, "hdr")) {
    std::string engine(hdr_engine_key(s));
    reads.push_back(read(*hdr, "hdr", "engine", engine));
    if (engine == "rtx" || engine == "adaptive" || engine == "sdr") {
      s.true_hdr = engine == "rtx";
      s.hdr_native = engine == "adaptive";
    } else {
      reads.emplace_back(refuse(RefusalCode::FieldOutOfRange, std::format("hdr.engine : « {} »", engine)));
    }
    reads.push_back(read(*hdr, "hdr", "contrast", s.true_hdr_contrast));
    reads.push_back(read(*hdr, "hdr", "saturation", s.true_hdr_saturation));
    reads.push_back(read(*hdr, "hdr", "middle_gray", s.true_hdr_gray));
    reads.push_back(read(*hdr, "hdr", "peak", s.true_hdr_peak));
    reads.push_back(read(*hdr, "hdr", "debanding", s.true_hdr_debanding));
  } else if (const toml::table* rtx = section(root, "rtx_hdr")) {
    reads.push_back(read(*rtx, "rtx_hdr", "enabled", s.true_hdr));
    reads.push_back(read(*rtx, "rtx_hdr", "contrast", s.true_hdr_contrast));
    reads.push_back(read(*rtx, "rtx_hdr", "saturation", s.true_hdr_saturation));
    reads.push_back(read(*rtx, "rtx_hdr", "middle_gray", s.true_hdr_gray));
    reads.push_back(read(*rtx, "rtx_hdr", "peak", s.true_hdr_peak));
    reads.push_back(read(*rtx, "rtx_hdr", "debanding", s.true_hdr_debanding));
    reads.push_back(read(*rtx, "rtx_hdr", "native", s.hdr_native));
  }
  if (const toml::table* audio = section(root, "audio")) {
    reads.push_back(read(*audio, "audio", "delay_ms", s.audio_delay_ms));
    reads.push_back(read(*audio, "audio", "volume", s.volume));
    reads.push_back(read(*audio, "audio", "device", s.audio_device));
    reads.push_back(read(*audio, "audio", "listening", s.listening));
    reads.push_back(read(*audio, "audio", "faithful", s.audio_faithful));
    reads.push_back(read(*audio, "audio", "profile", s.audio_profile));
    if (s.audio_profile != "custom" && ::ajp::audio::find_preset(s.audio_profile) == nullptr)
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("audio.profile : « {} »", s.audio_profile)));
    if (const toml::table* custom = audio->get_as<toml::table>("custom")) {
      std::array<float, ::ajp::audio::kMixTuningFields> fields = ::ajp::audio::to_array(s.audio_custom);
      constexpr std::array<const char*, ::ajp::audio::kMixTuningFields> kNames{"crossfeed_gain",
                                                                               "crossfeed_cutoff_hz",
                                                                               "crossfeed_delay_ms",
                                                                               "head_radius_m",
                                                                               "head_shadow",
                                                                               "surround_gain",
                                                                               "center_lift",
                                                                               "lfe_gain",
                                                                               "downmix_scale",
                                                                               "distance",
                                                                               "room",
                                                                               "eq_bass_db",
                                                                               "eq_voice_db",
                                                                               "eq_treble_db",
                                                                               "night",
                                                                               "virtual_speakers"};
      for (std::size_t i = 0; i < fields.size(); ++i)
        reads.push_back(read(*custom, "audio.custom", kNames[i], fields[i]));
      s.audio_custom = ::ajp::audio::tuning_from(fields);
    }
    if (!listening_index(s.listening))
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("audio.listening : « {} »", s.listening)));
    if (const toml::table* devices = audio->get_as<toml::table>("devices")) {
      for (const auto& [name, node] : *devices) {
        const auto own = node.value_exact<std::string>();
        if (!own || !listening_index(*own)) {
          reads.emplace_back(
              refuse(RefusalCode::FieldOutOfRange, std::format("audio.devices.{}", std::string(name.str()))));
          continue;
        }
        s.device_listening.emplace_back(std::string(name.str()), *own);
      }
    }
    if (const toml::table* profiles = audio->get_as<toml::table>("device_profiles")) {
      for (const auto& [name, node] : *profiles) {
        const auto key = node.value_exact<std::string>();
        if (!key || (*key != "custom" && ::ajp::audio::find_preset(*key) == nullptr)) {
          reads.emplace_back(refuse(RefusalCode::FieldOutOfRange,
                                    std::format("audio.device_profiles.{}", std::string(name.str()))));
          continue;
        }
        s.device_profile.emplace_back(std::string(name.str()), *key);
      }
    }
    // « Son anime » : [audio.anime] (niveau, réglages du niveau « custom ») et [audio.device_anime].
    if (const toml::table* anime = audio->get_as<toml::table>("anime")) {
      reads.push_back(read(*anime, "audio.anime", "level", s.anime_sound));
      if (s.anime_sound != "custom" && ::ajp::audio::find_anime_level(s.anime_sound) == nullptr)
        reads.emplace_back(
            refuse(RefusalCode::FieldOutOfRange, std::format("audio.anime.level : « {} »", s.anime_sound)));
      if (const toml::table* custom = anime->get_as<toml::table>("custom")) {
        // Une valeur hors plage est refusée en nommant la clé, jamais corrigée en silence.
        ::ajp::audio::AnimeTuning& c = s.anime_custom;
        const auto ranged = [&](const char* key, float& value, float lo, float hi) {
          reads.push_back(read(*custom, "audio.anime.custom", key, value));
          if (!(value >= lo && value <= hi))
            reads.emplace_back(
                refuse(RefusalCode::FieldOutOfRange,
                       std::format("audio.anime.custom.{} : {} hors de {} à {}", key, value, lo, hi)));
        };
        ranged("width", c.width, 0.0F, 1.0F);
        ranged("ambience", c.ambience, 0.0F, 1.0F);
        ranged("voice_db", c.voice_db, 0.0F, 4.0F);
        ranged("presence_db", c.presence_db, 0.0F, 4.0F);
        ranged("voice_forward", c.voice_forward, 0.0F, 1.0F);
        ranged("bass", c.bass, 0.0F, 1.0F);
        reads.push_back(read(*custom, "audio.anime.custom", "loudness", c.loudness));
        ranged("target_lufs", c.target_lufs, -23.0F, -16.0F);
        ranged("compress", c.compress, 0.0F, 1.0F);
        ranged("ceiling_db", c.ceiling_db, -3.0F, -0.5F);
        c.enabled = true;
      }
    }
    if (const toml::table* devices = audio->get_as<toml::table>("device_anime")) {
      for (const auto& [name, node] : *devices) {
        const auto key = node.value_exact<std::string>();
        if (!key || (*key != "custom" && ::ajp::audio::find_anime_level(*key) == nullptr)) {
          reads.emplace_back(refuse(RefusalCode::FieldOutOfRange,
                                    std::format("audio.device_anime.{}", std::string(name.str()))));
          continue;
        }
        s.device_anime.emplace_back(std::string(name.str()), *key);
      }
    }
    reads.push_back(read(*audio, "audio", "hrir_path", s.hrir_path));
    if (const toml::table* hrirs = audio->get_as<toml::table>("device_hrir")) {
      for (const auto& [name, node] : *hrirs) {
        const auto path = node.value_exact<std::string>();
        if (!path) {
          reads.emplace_back(
              refuse(RefusalCode::FieldType, std::format("audio.device_hrir.{}", std::string(name.str()))));
          continue;
        }
        s.device_hrir.emplace_back(std::string(name.str()), *path);
      }
    }
    reads.push_back(read_legacy_languages(*audio, "audio", legacy_audio));
  }
  if (const toml::table* sync = section(root, "sync")) {
    reads.push_back(read(*sync, "sync", "display", s.display_sync));
    reads.push_back(read(*sync, "sync", "video_latency_ms", s.video_latency_ms));
  }
  if (const toml::table* picture = section(root, "picture")) {
    reads.push_back(read(*picture, "picture", "deband", s.deband));
    std::string scaler(name(s.final_scaler));
    reads.push_back(read(*picture, "picture", "scaler", scaler));
    bool scaler_known = false;
    for (int i = 0; i < kFinalScalerCount; ++i) {
      if (name(static_cast<FinalScaler>(i)) == scaler) {
        s.final_scaler = static_cast<FinalScaler>(i);
        scaler_known = true;
      }
    }
    if (!scaler_known)
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("picture.scaler : « {} »", scaler)));
  }
  if (const toml::table* interface = section(root, "interface")) {
    reads.push_back(read(*interface, "interface", "language", s.language));
    if (s.language != "en" && s.language != "fr")
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("interface.language : « {} »", s.language)));
  }
  if (const toml::table* engines = section(root, "engines"))
    reads.push_back(read(*engines, "engines", "install_dir", s.engines_dir));
  if (const toml::table* updates = section(root, "updates")) {
    reads.push_back(read(*updates, "updates", "mode", s.update_mode));
    reads.push_back(read(*updates, "updates", "channel", s.update_channel));
    reads.push_back(read(*updates, "updates", "frequency", s.update_frequency));
    const auto one_of = [](std::string_view value, std::initializer_list<std::string_view> allowed) {
      return std::ranges::find(allowed, value) != allowed.end();
    };
    if (!one_of(s.update_mode, {"notify", "auto", "never"}))
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("updates.mode : « {} »", s.update_mode)));
    if (!one_of(s.update_channel, {"stable", "preview"}))
      reads.emplace_back(
          refuse(RefusalCode::FieldOutOfRange, std::format("updates.channel : « {} »", s.update_channel)));
    if (!one_of(s.update_frequency, {"launch", "daily", "weekly"}))
      reads.emplace_back(refuse(RefusalCode::FieldOutOfRange,
                                std::format("updates.frequency : « {} »", s.update_frequency)));
  }
  if (const toml::table* span = section(root, "span_prepare"))
    reads.push_back(read(*span, "span_prepare", "background", s.span_prepare_background));
  if (const toml::table* player = section(root, "default_player"))
    reads.push_back(read(*player, "default_player", "dismissed", s.default_player_dismissed));
  if (const toml::table* playlist = section(root, "playlist"))
    reads.push_back(read(*playlist, "playlist", "autoplay_next", s.autoplay_next));
  if (const toml::table* subtitles = section(root, "subtitles")) {
    reads.push_back(read(*subtitles, "subtitles", "scale", s.subtitle_scale));
    reads.push_back(read_legacy_languages(*subtitles, "subtitles", legacy_subtitles));
  }
  if (const toml::table* languages = section(root, "languages")) {
    reads.push_back(read(*languages, "languages", "audio_1", s.language_presets[0].audio));
    reads.push_back(read(*languages, "languages", "subtitles_1", s.language_presets[0].subtitles));
    reads.push_back(read(*languages, "languages", "audio_2", s.language_presets[1].audio));
    reads.push_back(read(*languages, "languages", "subtitles_2", s.language_presets[1].subtitles));
  } else if (!legacy_audio.empty() || !legacy_subtitles.empty()) {
    // Le format d'avant : deux listes, converties en deux préréglages (rang 1, puis rang 2).
    const auto rank = [](const std::vector<std::string>& list, std::size_t i) {
      return i < list.size() ? list[i] : std::string{};
    };
    s.language_presets = {
        {rank(legacy_audio, 0), rank(legacy_subtitles, 0)},
        {legacy_audio.size() > 1 ? legacy_audio[1] : rank(legacy_audio, 0), rank(legacy_subtitles, 1)}};
  }
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
    // La plage de la rc.3 allait à 1,2 ; la course est revenue à 0,8 (l'effet, lui, est plus fort).
    if (v.vibrance <= 1.2) v.vibrance = std::min(v.vibrance, render::grade_limits::kVibrance.hi);
    if (v.vibrance_rtx <= 1.2) v.vibrance_rtx = std::min(v.vibrance_rtx, render::grade_limits::kVibrance.hi);
    reads.push_back(read(*grade, "grade", "area_rtx", v.area_rtx));
    reads.push_back(read(*grade, "grade", "guard", v.guard));
    reads.push_back(read(*grade, "grade", "expand", v.expand));
    reads.push_back(read(*grade, "grade", "gamut_p3", v.gamut_p3));
    reads.push_back(read(*grade, "grade", "look", s.look));
    reads.push_back(read(*grade, "grade", "look_mix", s.look_mix));
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
  // Les rc.3 laissaient aller contraste et saturation à 400 : un play.toml écrit alors se relit au maximum.
  constexpr int kRc3Max = 400;
  if (s.true_hdr_contrast <= kRc3Max)
    s.true_hdr_contrast = std::min(s.true_hdr_contrast, limits::kTrueHdrContrast.hi);
  if (s.true_hdr_saturation <= kRc3Max)
    s.true_hdr_saturation = std::min(s.true_hdr_saturation, limits::kTrueHdrSaturation.hi);
  return validate(s);
}

}  // namespace

std::string format_settings(const PlaySettings& s) {
  toml::table root;
  root.insert("version", kSettingsVersion);
  root.insert("auto", toml::table{{"preset", std::string(key(s.preset))},
                                  {"priority", std::string(key(s.priority))}});
  root.insert("image", toml::table{{"upscale", s.upscale},
                                   {"model", s.model},
                                   {"work_height", s.work_height},
                                   {"downscale", std::string(name(s.downscale))},
                                   {"span_precision", std::string(name(s.span_precision))}});
  toml::table interpolation{{"engine", std::string(name(s.motion_engine))},
                            {"model", s.rife_model},
                            {"keep_poses", s.keep_poses},
                            {"rife_fused", s.rife_fused},
                            {"drba_fused", s.drba_fused},
                            {"gmfss_fused", s.gmfss_fused},
                            {"gmfss_precision", std::string(name(s.gmfss_precision))},
                            {"motion_precision", std::string(name(s.motion_precision))},
                            {"auto", s.rife_auto},
                            {"auto_cap", s.auto_cap},
                            {"auto_mode", std::string(s.auto_free ? "libre" : "recommande")},
                            {"span_order", std::string(name(s.span_order))},
                            {"smooth_motion", s.smooth_motion}};
  if (s.rife_factor_den == 2) {
    interpolation.insert("rife", static_cast<double>(s.rife_factor) / 2.0);
  } else {
    interpolation.insert("rife", s.rife_factor);
  }
  root.insert("interpolation", std::move(interpolation));
  root.insert("hdr", toml::table{{"engine", std::string(hdr_engine_key(s))},
                                 {"contrast", s.true_hdr_contrast},
                                 {"saturation", s.true_hdr_saturation},
                                 {"middle_gray", s.true_hdr_gray},
                                 {"peak", s.true_hdr_peak},
                                 {"debanding", s.true_hdr_debanding}});
  // Le volume au centième : 0.85 et non 0.8500000238418579.
  {
    toml::table audio{
        {"delay_ms", s.audio_delay_ms}, {"volume", std::round(static_cast<double>(s.volume) * 100.0) / 100.0},
        {"device", s.audio_device},     {"listening", s.listening},
        {"faithful", s.audio_faithful}, {"profile", s.audio_profile}};
    {
      const auto fields = ::ajp::audio::to_array(s.audio_custom);
      const auto r4 = [](float v) { return std::round(static_cast<double>(v) * 10000.0) / 10000.0; };
      toml::table custom{{"crossfeed_gain", r4(fields[0])},
                         {"crossfeed_cutoff_hz", r4(fields[1])},
                         {"crossfeed_delay_ms", r4(fields[2])},
                         {"head_radius_m", r4(fields[3])},
                         {"head_shadow", r4(fields[4])},
                         {"surround_gain", r4(fields[5])},
                         {"center_lift", r4(fields[6])},
                         {"lfe_gain", r4(fields[7])},
                         {"downmix_scale", r4(fields[8])},
                         {"distance", r4(fields[9])},
                         {"room", r4(fields[10])},
                         {"eq_bass_db", r4(fields[11])},
                         {"eq_voice_db", r4(fields[12])},
                         {"eq_treble_db", r4(fields[13])},
                         {"night", r4(fields[14])},
                         {"virtual_speakers", r4(fields[15])}};
      audio.insert_or_assign("custom", std::move(custom));
    }
    {
      const auto& c = s.anime_custom;
      const auto r4 = [](float v) { return std::round(static_cast<double>(v) * 10000.0) / 10000.0; };
      audio.insert_or_assign("anime",
                             toml::table{{"level", s.anime_sound},
                                         {"custom", toml::table{{"width", r4(c.width)},
                                                                {"ambience", r4(c.ambience)},
                                                                {"voice_db", r4(c.voice_db)},
                                                                {"presence_db", r4(c.presence_db)},
                                                                {"voice_forward", r4(c.voice_forward)},
                                                                {"bass", r4(c.bass)},
                                                                {"loudness", c.loudness},
                                                                {"target_lufs", r4(c.target_lufs)},
                                                                {"compress", r4(c.compress)},
                                                                {"ceiling_db", r4(c.ceiling_db)}}}});
    }
    if (!s.device_anime.empty()) {
      toml::table anime_devices;
      for (const auto& [name, key] : s.device_anime) anime_devices.insert_or_assign(name, key);
      audio.insert("device_anime", std::move(anime_devices));
    }
    if (!s.device_listening.empty()) {
      toml::table devices;
      for (const auto& [name, own] : s.device_listening) devices.insert_or_assign(name, own);
      audio.insert("devices", std::move(devices));
    }
    if (!s.device_profile.empty()) {
      toml::table profiles;
      for (const auto& [name, key] : s.device_profile) profiles.insert_or_assign(name, key);
      audio.insert("device_profiles", std::move(profiles));
    }
    if (!s.hrir_path.empty()) audio.insert_or_assign("hrir_path", s.hrir_path);
    if (!s.device_hrir.empty()) {
      toml::table hrirs;
      for (const auto& [name, path] : s.device_hrir) hrirs.insert_or_assign(name, path);
      audio.insert("device_hrir", std::move(hrirs));
    }
    root.insert("audio", std::move(audio));
  }
  {
    const media::LanguagePreset first =
        !s.language_presets.empty() ? s.language_presets[0] : media::LanguagePreset{};
    const media::LanguagePreset second =
        s.language_presets.size() > 1 ? s.language_presets[1] : media::LanguagePreset{};
    root.insert("languages", toml::table{{"audio_1", first.audio},
                                         {"subtitles_1", first.subtitles},
                                         {"audio_2", second.audio},
                                         {"subtitles_2", second.subtitles}});
  }
  root.insert("sync", toml::table{{"display", s.display_sync}, {"video_latency_ms", s.video_latency_ms}});
  root.insert("picture", toml::table{{"deband", s.deband}, {"scaler", std::string(name(s.final_scaler))}});
  root.insert("playlist", toml::table{{"autoplay_next", s.autoplay_next}});
  root.insert("interface", toml::table{{"language", s.language}});
  root.insert("default_player", toml::table{{"dismissed", s.default_player_dismissed}});
  root.insert("span_prepare", toml::table{{"background", s.span_prepare_background}});
  if (!s.engines_dir.empty()) root.insert("engines", toml::table{{"install_dir", s.engines_dir}});
  root.insert(
      "updates",
      toml::table{{"mode", s.update_mode}, {"channel", s.update_channel}, {"frequency", s.update_frequency}});
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
                                   {"look", s.look},
                                   {"look_mix", hundredth(s.look_mix)},
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

std::expected<PlaySettings, Refusal> parse_settings_lenient(std::string_view toml_text,
                                                            std::vector<std::string>* dropped) {
  toml::parse_result parsed = toml::parse(toml_text);
  if (!parsed) {
    return refuse(RefusalCode::ProfileUnreadable,
                  std::format("play.toml ligne {} : {}", parsed.error().source().begin.line,
                              parsed.error().description()));
  }
  toml::table root = std::move(parsed).table();
  // Une valeur fautive est écartée seule (le défaut la remplace) : le reste du fichier reste en vigueur.
  constexpr int kMaxDropped = 64;
  for (int attempt = 0; attempt <= kMaxDropped; ++attempt) {
    auto settings = settings_from_table(root);
    if (settings) return settings;
    const std::string& detail = settings.error().detail;
    const std::size_t dot = detail.find('.');
    if (dot == std::string::npos || dot == 0) return settings;
    const std::size_t found = detail.find_first_of(" :", dot);
    const std::size_t end = found == std::string::npos ? detail.size() : found;
    if (end <= dot + 1) return settings;
    const std::string section_name = detail.substr(0, dot);
    const std::string key_name = detail.substr(dot + 1, end - dot - 1);
    toml::node* node = root.get(section_name);
    toml::table* table = node != nullptr ? node->as_table() : nullptr;
    if (table == nullptr || !table->erase(key_name)) return settings;
    if (dropped != nullptr) dropped->push_back(detail);
  }
  return settings_from_table(root);
}

std::expected<PlaySettings, Refusal> load_settings(const std::filesystem::path& path) {
  std::error_code error;
  if (!std::filesystem::exists(path, error)) return PlaySettings{};
  std::ifstream file(path, std::ios::binary);
  if (!file) return refuse(RefusalCode::FileUnreadable, path.string());
  std::ostringstream text;
  text << file.rdbuf();
  std::vector<std::string> dropped;
  auto settings = parse_settings_lenient(text.str(), &dropped);
  for (const std::string& why : dropped)
    log_channel("app")->warn("{} : valeur écartée, défaut gardé ({})", path.string(), why);
  if (!settings)
    return refuse(settings.error().code, std::format("{} : {}", path.string(), settings.error().detail));
  return settings;
}

std::expected<void, Refusal> save_settings(const std::filesystem::path& path, const PlaySettings& settings) {
  if (auto valid = validate(settings); !valid) return std::unexpected(valid.error());
  return core::write_text_atomically(path, format_settings(settings));
}

}  // namespace ajp::app::play
