#include <array>
#include <string_view>

#include "app/play/settings.h"
#include "core/utf8_path.h"

namespace kirari::app::play {
namespace {

struct ExperimentalFlag {
  bool PlayOptions::* option;
  bool ExperimentalSettings::* setting;
  bool value = true;
};

constexpr std::array kExperimentalFlags{
    ExperimentalFlag{&PlayOptions::gmflow_fast, &ExperimentalSettings::gmflow_fast},
    ExperimentalFlag{&PlayOptions::fused_layers, &ExperimentalSettings::fused_layers},
    ExperimentalFlag{&PlayOptions::nvof_layer_guard, &ExperimentalSettings::nvof_layer_guard},
    ExperimentalFlag{&PlayOptions::span_fp8_native_blocks, &ExperimentalSettings::span_fp8_native_blocks},
    ExperimentalFlag{&PlayOptions::effect_highlights, &ExperimentalSettings::effect_glow, false},
    ExperimentalFlag{&PlayOptions::map_effects, &ExperimentalSettings::map_effects},
};

}  // namespace

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
  // A18-09 ; --motion=gmfss : GMFSS union seul, avec ou sans --gmfss-native (la comparaison des deux).
  if (options.motion) {
    constexpr std::array kEngines{MotionEngine::Rife, MotionEngine::DistilDrba, MotionEngine::Gmfss};
    s.motion_engine = kEngines[static_cast<std::size_t>(*options.motion)];
  }
  if (options.was_given("--rife")) {
    s.rife_factor = options.rife_factor;
    s.rife_factor_den = options.rife_factor_den;
    s.rife_auto = false;  // un multiplicateur donné à la main coupe le mode auto, comme dans le menu
  }
  if (options.was_given("--truehdr-peak")) s.true_hdr_peak = options.true_hdr_peak;
  if (options.smooth_drawings) s.smooth_drawings = true;
  for (const auto& flag : kExperimentalFlags)
    if (options.*flag.option == flag.value) s.experimental.*flag.setting = flag.value;
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
  // --smooth-drawings est un drapeau (absent de `given`) : l'essai de la ligne de commande ne s'enregistre
  // pas.
  if (options.smooth_drawings && running.smooth_drawings == started.smooth_drawings)
    out.smooth_drawings = from_file.smooth_drawings;
  for (const auto& flag : kExperimentalFlags)
    if (options.*flag.option == flag.value &&
        running.experimental.*flag.setting == started.experimental.*flag.setting)
      out.experimental.*flag.setting = from_file.experimental.*flag.setting;
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

}  // namespace kirari::app::play
