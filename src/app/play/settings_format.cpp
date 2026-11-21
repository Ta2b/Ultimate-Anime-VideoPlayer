// L'écriture de play.toml : format_settings met les réglages en forme. La lecture et l'accès au fichier sont
// dans settings_toml.cpp.
#include <cmath>
#include <format>
#include <sstream>
#include <string>

#include "app/play/auto_config.h"
#include "app/play/settings.h"
#include "app/play/settings_toml_lenient.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++-kirari/toml.hpp>

namespace kirari::app::play {

namespace {

// La table [image] : l'agrandissement et la stabilisation temporelle.
toml::table image_table(const PlaySettings& s) {
  return toml::table{{"upscale", s.upscale},
                     {"restore_lines", s.restore_lines},
                     {"model", s.model},
                     {"work_height", s.work_height},
                     {"downscale", std::string(name(s.downscale))},
                     {"span_precision", std::string(name(s.span_precision))},
                     {"temporal_fix", std::string(name(s.temporal_fix))},
                     {"temporal_fix_full", s.temporal_fix_full}};
}

toml::table experimental_table(const ExperimentalSettings& s) {
  return toml::table{{"sharp1_winograd", s.sharp1_winograd},
                     {"drba_native_tail", s.drba_native_tail},
                     {"gmfss_folded_tail", s.gmfss_folded_tail},
                     {"gmflow_fast", s.gmflow_fast},
                     {"gmfss_direct_splats", s.gmfss_direct_splats},
                     {"fused_layers", s.fused_layers},
                     {"nvof_layer_guard", s.nvof_layer_guard},
                     {"span_fp8_native_blocks", s.span_fp8_native_blocks},
                     {"effect_glow", s.effect_glow},
                     {"guided_glows", s.guided_glows},
                     {"map_effects", s.map_effects}};
}

}  // namespace

std::string format_settings(const PlaySettings& s) {
  toml::table root;
  root.insert("version", kSettingsVersion);
  root.insert("auto", toml::table{{"preset", std::string(key(s.preset))},
                                  {"priority", std::string(key(s.priority))}});
  root.insert("image", image_table(s));
  root.insert("experimental", experimental_table(s.experimental));
  toml::table interpolation{{"engine", std::string(name(s.motion_engine))},
                            {"model", s.rife_model},
                            {"keep_poses", s.keep_poses},
                            {"smooth_drawings", s.smooth_drawings},
                            {"layers", s.layers},
                            {"background_plate", s.background_plate},
                            {"pair_scale", s.pair_scale},
                            {"strong_middle", s.strong_middle},
                            {"motion_720p", s.motion_720p},
                            {"gmfss_middle", s.gmfss_middle},
                            {"gmfss_motion", s.gmfss_motion},
                            {"gmfss_half_flow", s.gmfss_half_flow},
                            {"gmfss_compose", s.gmfss_compose},
                            {"gmfss_every_instant", s.gmfss_every_instant},
                            {"gmfss_light", s.gmfss_light},
                            {"gmfss_key_floor", s.gmfss_key_floor},
                            {"rife_fused", s.rife_fused},
                            {"cubic_warp", s.cubic_warp},
                            {"pan_lock", s.pan_lock},
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
    toml::table hdr{{"peak", s.true_hdr_peak}, {"output", std::string(name(s.hdr_mode))}};
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
    toml::table languages{{"own", s.own_language},
                          {"version", std::string(version_name(s.language_version))}};
    for (std::size_t i = 0; i < s.language_presets.size(); ++i) {
      languages.insert(std::format("audio_{}", i + 1), s.language_presets[i].audio);
      languages.insert(std::format("subtitles_{}", i + 1), s.language_presets[i].subtitles);
    }
    root.insert("languages", std::move(languages));
  }
  root.insert("sync", toml::table{{"display", s.display_sync}, {"video_latency_ms", s.video_latency_ms}});
  root.insert("picture", toml::table{{"deband", s.deband},
                                     {"scaler", std::string(name(s.final_scaler))},
                                     {"sharpness", s.sharpness}});
  root.insert("playlist", toml::table{{"autoplay_next", s.autoplay_next}});
  toml::table interface{{"language", s.language}};
  if (s.ui_scale != 0) interface.insert("ui_scale", s.ui_scale);
  root.insert("interface", std::move(interface));
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
                                   {"face_skin", s.face_skin},
                                   {"surface_level", s.surface_white},
                                   {"anime_contrast", g.anime_contrast},
                                   {"anime_saturation", g.anime_saturation},
                                   {"anime_highlights", g.anime_highlights},
                                   {"oled_black", std::string(name(s.oled_black))},
                                   {"oled_black_steps", hundredth(s.oled_black_steps)}});
  root.insert("oled", toml::table{{"care", s.oled_care}, {"steady_cadence", s.steady_cadence}});
  root.insert("stats", toml::table{{"all", s.stats.all},
                                   {"chain", s.stats.chain},
                                   {"frames", s.stats.frames},
                                   {"fps", s.stats.fps},
                                   {"rife", s.stats.rife},
                                   {"tracks", s.stats.tracks},
                                   {"output", s.stats.output},
                                   {"position", s.stats.position},
                                   {"memory", s.stats.memory}});
  std::ostringstream out;
  out << "# Réglages de « kirari play », écrits par le menu des paramètres.\n" << root << '\n';
  return out.str();
}

}  // namespace kirari::app::play
