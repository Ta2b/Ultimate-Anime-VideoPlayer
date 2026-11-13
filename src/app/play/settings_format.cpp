// L'écriture de play.toml : format_settings met les réglages en forme. La lecture et l'accès au fichier sont
// dans settings_toml.cpp.
#include <cmath>
#include <sstream>
#include <string>

#include "app/play/auto_config.h"
#include "app/play/settings.h"
#include "app/play/settings_toml_lenient.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++-ajp/toml.hpp>

namespace ajp::app::play {

std::string format_settings(const PlaySettings& s) {
  toml::table root;
  root.insert("version", kSettingsVersion);
  root.insert("auto", toml::table{{"preset", std::string(key(s.preset))},
                                  {"priority", std::string(key(s.priority))}});
  root.insert("image", toml::table{{"upscale", s.upscale},
                                   {"restore_lines", s.restore_lines},
                                   {"model", s.model},
                                   {"work_height", s.work_height},
                                   {"downscale", std::string(name(s.downscale))},
                                   {"span_precision", std::string(name(s.span_precision))}});
  toml::table interpolation{{"engine", std::string(name(s.motion_engine))},
                            {"model", s.rife_model},
                            {"keep_poses", s.keep_poses},
                            {"smooth_drawings", s.smooth_drawings},
                            {"pair_scale", s.pair_scale},
                            {"strong_middle", s.strong_middle},
                            {"rife_fused", s.rife_fused},
                            {"cubic_warp", s.cubic_warp},
                            {"drba_fused", s.drba_fused},
                            {"motion_precision", std::string(name(s.motion_precision))},
                            {"auto", s.rife_auto},
                            {"auto_cap", s.auto_cap},
                            {"auto_mode", std::string(s.auto_free ? "libre" : "recommande")},
                            {"auto_off", s.auto_off},
                            {"span_order", std::string(name(s.span_order))},
                            {"smooth_motion", s.smooth_motion}};
  if (s.rife_factor_den == 2) {
    interpolation.insert("rife", static_cast<double>(s.rife_factor) / 2.0);
  } else {
    interpolation.insert("rife", s.rife_factor);
  }
  root.insert("interpolation", std::move(interpolation));
  {
    toml::table hdr{{"peak", s.true_hdr_peak}};
    root.insert("hdr", std::move(hdr));
  }
  // Le volume au centième : 0.85 et non 0.8500000238418579.
  {
    toml::table audio{{"delay_ms", s.audio_delay_ms},
                      {"volume", std::round(static_cast<double>(s.volume) * 100.0) / 100.0},
                      {"device", s.audio_device}};
    toml::table delays;
    for (const auto& [device, ms] : s.audio_device_delays_ms) {
      if (device != s.audio_device) delays.insert(device, ms);
    }
    if (!delays.empty()) audio.insert("delays", std::move(delays));
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
  root.insert("picture", toml::table{{"deband", s.deband},
                                     {"scaler", std::string(name(s.final_scaler))},
                                     {"sharpness", s.sharpness}});
  root.insert("playlist", toml::table{{"autoplay_next", s.autoplay_next}});
  root.insert("interface", toml::table{{"language", s.language}});
  if (!s.engines_dir.empty() || !s.gpu.empty() || !s.fast_precision) {
    toml::table engines;
    if (!s.engines_dir.empty()) engines.insert("install_dir", s.engines_dir);
    if (!s.gpu.empty()) engines.insert("gpu", s.gpu);
    if (!s.fast_precision) engines.insert("fast_precision", false);
    root.insert("engines", std::move(engines));
  }
  root.insert("updates", toml::table{{"mode", std::string(name(s.update_mode))},
                                     {"frequency", std::string(name(s.update_frequency))}});
  root.insert("subtitles",
              toml::table{{"scale", std::round(static_cast<double>(s.subtitle_scale) * 100.0) / 100.0}});
  // Les réglages du grade au centième, comme le menu les montre.
  const auto hundredth = [](double v) { return std::round(v * 100.0) / 100.0; };
  const render::GradeSettings& g = s.grade_values;
  root.insert("grade", toml::table{{"enabled", s.grade},
                                   {"shadows", hundredth(g.shadows)},
                                   {"highlights_rtx", hundredth(g.highlights_rtx)},
                                   {"area_rtx", hundredth(g.area_rtx)},
                                   {"look_contrast", s.look_contrast},
                                   {"look_highlight", s.look_highlight},
                                   {"flash_guard", s.flash_guard},
                                   {"look_table", s.look_table},
                                   {"grade_before_warp", s.grade_before_warp},
                                   {"face_skin", s.face_skin},
                                   {"surface_white", s.surface_white},
                                   {"anime_contrast", g.anime_contrast},
                                   {"anime_saturation", g.anime_saturation},
                                   {"anime_highlights", g.anime_highlights},
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

}  // namespace ajp::app::play
