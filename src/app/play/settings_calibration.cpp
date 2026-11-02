#include "app/play/settings_calibration.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#include <imgui.h>

#include "app/play/calibration_images.h"
#include "app/play/grade_learning.h"
#include "app/play/i18n.h"
#include "app/play/panel_style.h"
#include "core/messages.h"
#include "render/grade_params.h"

namespace ajp::app::play {

namespace {

bool wizard_slider(const char* label, double& value, core::Range<double> range) {
  auto shown = static_cast<float>(value);
  if (!ImGui::SliderFloat(label, &shown, static_cast<float>(range.lo), static_cast<float>(range.hi), "%.2f",
                          ImGuiSliderFlags_AlwaysClamp))
    return false;
  value = std::clamp(std::round(static_cast<double>(shown) * 100.0) / 100.0, range.lo, range.hi);
  return true;
}

// Le plan montré : sa navigation, ce qu'il faut y voir, et la vidéo de calibration tenue en pause sur lui.
void shot_view(const SettingsChoices& choices, GradeWizard& wizard, float scale, float panel_left,
               std::vector<Input>& actions) {
  const CalibrationImage image =
      calibration_shot(kCalibrationOrder.at(static_cast<std::size_t>(wizard.image)));
  const bool bonus = wizard.image >= kCalibrationRequired;
  if (!bonus) {
    panel::note(i18n::tr_format("Shot {}/{} · {}", wizard.image + 1, kCalibrationRequired,
                                i18n::tr_view(image.title)));
  } else {
    panel::note(i18n::tr_format("Bonus shot {}/{} · {}", wizard.image - kCalibrationRequired + 1,
                                kCalibrationImages - kCalibrationRequired, i18n::tr_view(image.title)));
  }
  panel::tip(std::string(i18n::tr_view(image.guidance)));
  const std::filesystem::path clip = calibration_clip();
  wizard.showing = !clip.empty() && choices.source_file == clip;
  if (!wizard.showing) {
    panel::note(i18n::tr("Opening the calibration video…"));
  } else {
    // Seul ce plan se montre : la barre de lecture, les chapitres ou les flèches qui mènent ailleurs dans la
    // vidéo y ramènent.
    const std::int64_t begin = image.id * kCalibrationSegmentUs;
    const bool outside = choices.position_us < begin || choices.position_us >= begin + kCalibrationSegmentUs;
    const double now = ImGui::GetTime();
    if (wizard.sought != image.id || (outside && now - wizard.sought_at > 0.6)) {
      actions.emplace_back(Command{cmd::Seek{.target_us = image.at_us, .exact = true}});
      wizard.sought = image.id;
      wizard.sought_at = now;
    }
    if (!choices.paused) actions.emplace_back(Command{cmd::Pause{}});  // un plan fixe, pas la suite
  }
  const ImGuiIO& io = ImGui::GetIO();
  const float pad = 18.0F * scale;
  float w = std::max(panel_left - 2.0F * pad, 160.0F * scale);
  float h = w * 9.0F / 16.0F;
  const float room_h = io.DisplaySize.y - 2.0F * pad - 120.0F * scale;
  if (h > room_h) {
    h = std::max(room_h, 90.0F * scale);
    w = h * 16.0F / 9.0F;
  }
  const float x0 = pad;
  const float y0 = std::max(pad, (io.DisplaySize.y - 120.0F * scale - h) * 0.5F);
  wizard.shot = CalibrationShot{.x0 = static_cast<int>(x0),
                                .y0 = static_cast<int>(y0),
                                .x1 = static_cast<int>(x0 + w),
                                .y1 = static_cast<int>(y0 + h)};
}

// La calibration fermée : les profils de grade calibrés de l'écran (choisir celui en service, en supprimer),
// et le départ d'une calibration, pour un nouveau profil ou pour recalibrer celui en service.
void calibration_start(PlaySettings& d, const SettingsChoices& choices, GradeWizard& wizard,
                       std::vector<Input>& actions) {
  render::GradeSettings& g = d.grade_values;
  const std::string& key = choices.display_key;
  // Demandée par l'assistant de premier démarrage : un nouveau profil, sans attendre le bouton.
  const bool asked = std::exchange(wizard.autostart, false);
  panel::note(i18n::tr(
      "Sets the adaptive grade on this display from five required anime shots (night, half-light, "
      "daylight, sky, bright lamps) and thirteen bonus ones: on each one, set the deep blacks, the colors "
      "and "
      "the "
      "middle gray for "
      "that scene. The grade then fits its curve to all of them at once, each scene at its own brightness. "
      "The result is kept for this display."));
  const std::vector<std::string> names = grade_profiles(d, key);
  const PlaySettings::DisplayGrade* active = calibrated_grade(d, key);
  if (!names.empty()) {
    const std::string shown = active != nullptr ? active->name : std::string(i18n::tr("None"));
    if (ImGui::BeginCombo(i18n::tr("Grade profile"), shown.c_str())) {
      for (const std::string& name : names) {
        if (ImGui::Selectable(name.c_str(), active != nullptr && active->name == name)) {
          d.grade = true;
          g.adaptive = true;
          use_calibrated_grade(d, key, name);
        }
      }
      ImGui::EndCombo();
    }
    if (active != nullptr && ImGui::Button(i18n::tr("Delete this profile"))) {
      const std::string gone = active->name;
      forget_calibrated_grade(d, key, gone);
      d = with_calibrated_grade(std::move(d), key);
      active = calibrated_grade(d, key);
    }
  }
  const std::filesystem::path clip = calibration_clip();
  std::error_code missing;
  if (clip.empty() || !std::filesystem::is_regular_file(clip, missing)) {
    panel::note(
        i18n::tr("The calibration video is not shipped with the player: put your calibration.mkv "
                 "in this folder."));
    const std::u8string folder = clip.parent_path().u8string();
    panel::note(std::string(folder.begin(), folder.end()));
    return;
  }
  if (key.empty()) {
    panel::note(i18n::tr("This display does not give its identity (EDID): the calibration cannot be kept."));
    return;
  }
  if (wizard.name[0] == '\0') {
    const std::string next = next_grade_profile_name(d, key);
    std::snprintf(wizard.name.data(), wizard.name.size(), "%s", next.c_str());
  }
  ImGui::InputText(i18n::tr("New profile name"), wizard.name.data(), wizard.name.size());
  std::string wanted(wizard.name.data());
  // « active » et « gray » sont des clés de play.toml sous l'écran, pas des noms de profil.
  if (wanted == "active" || wanted == "gray") wanted += " ";
  const bool fresh = (ImGui::Button(i18n::tr("Calibrate a new profile")) || asked) && !wanted.empty();
  bool again = false;
  if (active != nullptr) {
    ImGui::SameLine();
    again = ImGui::Button(i18n::tr("Recalibrate this profile"));
  }
  if (!fresh && !again) return;
  wizard.profile = again ? active->name : wanted;
  wizard.name[0] = '\0';
  wizard.before = d;
  d.grade = true;
  g.adaptive = true;
  wizard.step = 1;
  wizard.image = 0;
  wizard.samples = {};
  // La vidéo de calibration, en pause sur le premier plan ; à la fin, la vidéo d'avant à sa position.
  const bool already = choices.source_file == clip;
  wizard.on_clip = true;
  wizard.return_file = already ? std::filesystem::path() : choices.source_file;
  wizard.return_us = already ? 0 : choices.position_us;
  wizard.return_paused = choices.paused;
  // La calibration dite en cours dès maintenant : la vidéo de calibration rouvre la lecture, qui coupe le
  // grade adaptatif d'un écran sans profil hors calibration (with_calibrated_grade).
  g.adaptive_force = render::grade_limits::kCalibratingLive;
  const CalibrationImage first = calibration_shot(0);
  wizard.sought = first.id;
  actions.emplace_back(PlayFile{.file = clip, .start_us = first.at_us, .paused = true});
}
}  // namespace

void settle_calibration(GradeWizard& wizard, std::vector<Input>& actions) {
  if (wizard.step > 0 || !wizard.on_clip) return;
  wizard.on_clip = false;
  wizard.showing = false;
  wizard.sought = -1;
  actions.emplace_back(
      PlayFile{.file = wizard.return_file, .start_us = wizard.return_us, .paused = wizard.return_paused});
}

void grade_calibration(PlaySettings& d, const SettingsChoices& choices, GradeWizard& wizard, float scale,
                       float panel_left, std::vector<Input>& actions) {
  using namespace render::grade_limits;
  render::GradeSettings& g = d.grade_values;
  panel::section(i18n::tr("Adaptive grade calibration"));
  if (!choices.hdr_output) {
    wizard.step = 0;
    g.adaptive_force = -1.0;
    panel::note(i18n::tr("The display receives no HDR: the calibration needs an HDR output."));
    panel::end_group();
    return;
  }
  if (wizard.step == 0) {
    g.adaptive_force = -1.0;
    calibration_start(d, choices, wizard, actions);
    panel::end_group();
    return;
  }
  // Chaque plan à sa vraie place : la scène n'est pas forcée. La vidéo de calibration rouvre la lecture avec
  // les réglages d'avant : sur un écran sans profil, le grade adaptatif y est coupé (with_calibrated_grade),
  // et le panneau reprend ces réglages ; il reste allumé à chaque plan.
  g.adaptive_force = kCalibratingLive;
  g.adaptive = true;
  panel::note(i18n::tr(
      "Set each shot so it looks right, as it will be played in your episodes. Each one counts at its "
      "own brightness; Finish fits the grade to all of them."));
  shot_view(choices, wizard, scale, panel_left, actions);
  const AdaptiveScene& scene = choices.adaptive_scene;
  const CalibrationImage image =
      calibration_shot(kCalibrationOrder.at(static_cast<std::size_t>(wizard.image)));
  // La scène du plan se mesure une fois le plan affiché (la pré-analyse des plans, chain_shots.h).
  const bool settled = wizard.showing && scene.live && wizard.sought == image.id &&
                       ImGui::GetTime() - wizard.sought_at > 0.8 &&
                       choices.position_us >= image.id * kCalibrationSegmentUs &&
                       choices.position_us < (image.id + 1) * kCalibrationSegmentUs;
  if (!settled) {
    panel::note(i18n::tr("Measuring the shot…"));
  } else {
    GradeCurve curve = curve_of(d);
    const SceneLook look = look_at(curve, scene.t, scene.glow, scene.tv);
    bool changed = false;
    double shadows = look.shadows;
    if (wizard_slider(i18n::tr("Deep blacks, this scene"), shadows, kShadows)) {
      correct_scene(curve, scene.t, scene.glow, scene.tv, kShadowsFamily, shadows);
      changed = true;
    }
    double colors = look.colors;
    if (wizard_slider(i18n::tr("Colors, this scene"), colors, kVibrance)) {
      correct_scene(curve, scene.t, scene.glow, scene.tv, kColorsFamily, colors);
      changed = true;
    }
    auto gray = static_cast<float>(std::round(look.gray));
    if (ImGui::SliderFloat(i18n::tr("Middle gray, this scene"), &gray, static_cast<float>(kMiddleGray.lo),
                           static_cast<float>(kMiddleGray.hi), "%.0f", ImGuiSliderFlags_AlwaysClamp)) {
      correct_scene(curve, scene.t, scene.glow, scene.tv, kGrayFamily, std::round(static_cast<double>(gray)));
      changed = true;
    }
    if (changed) set_curve(d, curve);
    // Ce que l'utilisateur voit sur ce plan, retenu tant qu'il le regarde : la calibration en tire la courbe.
    wizard.samples.at(static_cast<std::size_t>(image.id)) =
        LearnSample{.unix_s = static_cast<std::int64_t>(std::time(nullptr)),
                    .kind = LearnSample::Kind::Calibration,
                    .touched = kAllFamilies,
                    .scene = LearnScene{.x = scene.x, .glow_seen = scene.glow_seen, .tv = scene.tv},
                    .look = look_at(curve, scene.t, scene.glow, scene.tv)};
  }
  const int last = kCalibrationImages - 1;
  ImGui::BeginDisabled(wizard.image == 0);
  if (ImGui::Button(i18n::tr("Previous shot"))) --wizard.image;
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(wizard.image == last);
  // Après le dernier plan obligatoire, la suite est facultative : les plans bonus précisent la courbe.
  if (ImGui::Button(wizard.image == kCalibrationRequired - 1 ? i18n::tr("Go further: bonus shots")
                                                             : i18n::tr("Next shot")))
    ++wizard.image;
  ImGui::EndDisabled();
  std::vector<LearnSample> kept;
  int required = 0;
  for (int i = 0; i < kCalibrationImages; ++i) {
    const auto& sample =
        wizard.samples.at(static_cast<std::size_t>(kCalibrationOrder.at(static_cast<std::size_t>(i))));
    if (!sample) continue;
    kept.push_back(*sample);
    if (i < kCalibrationRequired) ++required;
  }
  panel::note(i18n::tr_format("{} of {} required shots set, {} bonus.", required, kCalibrationRequired,
                              static_cast<int>(kept.size()) - required));
  ImGui::BeginDisabled(required < kCalibrationRequired);
  const bool finish = ImGui::Button(i18n::tr("Finish"));
  ImGui::EndDisabled();
  if (finish) {
    // La courbe qui rend le mieux tous les plans à la fois, seuils compris ; sans elle (refus), celle réglée.
    if (auto learned = learn_grade(kept, curve_of(d), static_cast<std::int64_t>(std::time(nullptr))))
      set_curve(d, learned->curve);
    wizard.step = 0;
    g.adaptive_force = -1.0;
    keep_calibrated_grade(d, choices.display_key, wizard.profile);
  }
  ImGui::SameLine();
  if (ImGui::Button(i18n::tr("Cancel"))) {
    wizard.step = 0;
    d = wizard.before;
    d.grade_values.adaptive_force = -1.0;
  }
  panel::end_group();
}

void scene_correction(PlaySettings& d, const SettingsChoices& choices, std::vector<Input>& actions) {
  using namespace render::grade_limits;
  const AdaptiveScene& now = choices.adaptive_scene;
  if (!now.live || !d.grade_values.adaptive ||
      render::grade_limits::calibrating(d.grade_values.adaptive_force))
    return;
  ImGui::SeparatorText(i18n::tr("This scene"));
  const double dark = (1.0 - now.glow) * (1.0 - now.t);
  const double bright = (1.0 - now.glow) * now.t;
  panel::note(i18n::tr_format(
      "The scene on screen is {:.0f} % dark, {:.0f} % bright and {:.0f} % glow: a change here "
      "goes to those scene kinds in that share, and is remembered for learning.",
      100.0 * dark, 100.0 * bright, 100.0 * now.glow));
  GradeCurve curve = curve_of(d);
  const SceneLook look = look_at(curve, now.t, now.glow, now.tv);
  bool changed = false;
  double shadows = look.shadows;
  if (wizard_slider(i18n::tr("Deep blacks, this scene"), shadows, kShadows)) {
    correct_scene(curve, now.t, now.glow, now.tv, kShadowsFamily, shadows);
    changed = true;
  }
  ImGui::BeginDisabled(!choices.hdr_output);
  double colors = look.colors;
  if (wizard_slider(i18n::tr("Colors, this scene"), colors, kVibrance)) {
    correct_scene(curve, now.t, now.glow, now.tv, kColorsFamily, colors);
    changed = true;
  }
  ImGui::EndDisabled();
  auto gray = static_cast<float>(std::round(look.gray));
  if (ImGui::SliderFloat(i18n::tr("Middle gray, this scene"), &gray, static_cast<float>(kMiddleGray.lo),
                         static_cast<float>(kMiddleGray.hi), "%.0f", ImGuiSliderFlags_AlwaysClamp)) {
    correct_scene(curve, now.t, now.glow, now.tv, kGrayFamily, std::round(static_cast<double>(gray)));
    changed = true;
  }
  if (changed) set_curve(d, curve);
  ImGui::BeginDisabled(now.corrections < 2);
  if (ImGui::Button(i18n::tr("Learn from my corrections"))) actions.emplace_back(LearnGrade{});
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(now.corrections + now.approvals == 0);
  if (ImGui::Button(i18n::tr("Forget my corrections"))) actions.emplace_back(ForgetGradeLearning{});
  ImGui::EndDisabled();
  panel::note(i18n::tr_format(
      "{} corrections and {} approved scenes kept. Learning fits the scene thresholds and the dark, glow, "
      "bright and blue-sky values to all of them at once, recent ones first; what they do not contradict "
      "stays as calibrated.",
      now.corrections, now.approvals));
}

}  // namespace ajp::app::play
