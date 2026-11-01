#include "app/play/settings_calibration.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

// L'étape de la calibration en cours (wizard.step de 1 à 3).
CalibrationStep step_of(const GradeWizard& wizard) {
  return static_cast<CalibrationStep>(std::clamp(wizard.step, 1, kCalibrationSteps) - 1);
}

// Le plan d'exemple de l'étape : sa navigation, ce qu'il faut y voir, la place de la vidéo à gauche du
// panneau (16:9 dans la place libre), et la vidéo de calibration tenue en pause sur ce plan.
void example_image(const SettingsChoices& choices, GradeWizard& wizard, float scale, float panel_left,
                   std::vector<Input>& actions) {
  const CalibrationStep step = step_of(wizard);
  const int count = calibration_count(step);
  const CalibrationImage image = calibration_image(step, wizard.image);
  panel::note(i18n::tr_format("Example {}/{} · {}", wizard.image + 1, count, i18n::tr_view(image.title)));
  if (ImGui::Button(i18n::tr("Previous example"))) wizard.image = (wizard.image + count - 1) % count;
  ImGui::SameLine();
  if (ImGui::Button(i18n::tr("Next example"))) wizard.image = (wizard.image + 1) % count;
  panel::tip(std::string(i18n::tr_view(image.guidance)));
  const std::filesystem::path clip = calibration_clip();
  wizard.showing = !clip.empty() && choices.source_file == clip;
  if (!wizard.showing) {
    panel::note(i18n::tr("Opening the calibration video…"));
  } else {
    const CalibrationImage shown = calibration_image(step, wizard.image);
    // Chaque étape ne montre que ses plans : la barre de lecture, les chapitres ou les flèches qui mènent
    // ailleurs dans la vidéo y ramènent.
    const std::int64_t begin = shown.id * kCalibrationSegmentUs;
    const bool outside = choices.position_us < begin || choices.position_us >= begin + kCalibrationSegmentUs;
    const double now = ImGui::GetTime();
    if (wizard.sought != shown.id || (outside && now - wizard.sought_at > 0.6)) {
      actions.emplace_back(Command{cmd::Seek{.target_us = shown.at_us, .exact = true}});
      wizard.sought = shown.id;
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
  panel::note(
      i18n::tr("Sets the adaptive grade on this display in three steps: a dark scene, a dark scene with a "
               "bright glow, then a bright scene. The result is kept for this display and drives the "
               "grade."));
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
        i18n::tr("The calibration video is missing (calibration/calibration.mkv next to the player): "
                 "reinstall the player."));
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
  // La vidéo de calibration, en pause sur le premier plan ; à la fin, la vidéo d'avant à sa position.
  const bool already = choices.source_file == clip;
  wizard.on_clip = true;
  wizard.return_file = already ? std::filesystem::path() : choices.source_file;
  wizard.return_us = already ? 0 : choices.position_us;
  wizard.return_paused = choices.paused;
  const CalibrationImage first = calibration_image(CalibrationStep::Dark, 0);
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
  const CalibrationStep step = step_of(wizard);
  const auto gray_slider = [](const char* label, double& value) {
    auto shown = static_cast<float>(value);
    if (ImGui::SliderFloat(label, &shown, static_cast<float>(kMiddleGray.lo),
                           static_cast<float>(kMiddleGray.hi), "%.0f", ImGuiSliderFlags_AlwaysClamp))
      value = std::clamp(std::round(static_cast<double>(shown)), kMiddleGray.lo, kMiddleGray.hi);
  };
  g.adaptive_force = step == CalibrationStep::Dark ? 0.0 : step == CalibrationStep::Bright ? 1.0 : kForceGlow;
  if (step == CalibrationStep::Dark) {
    panel::note(
        i18n::tr("1/3 · Dark scene. Set the middle gray, the deep blacks and the colors while looking at "
                 "the example on the left, played like your episodes."));
    ImGui::SliderInt(i18n::tr("Mid gray"), &d.true_hdr_gray, limits::kTrueHdrGray.lo, limits::kTrueHdrGray.hi,
                     "%d", ImGuiSliderFlags_AlwaysClamp);
    wizard_slider(i18n::tr("Deep blacks"), g.shadows, kShadows);
    wizard_slider(i18n::tr("Colors"), g.vibrance_rtx, kVibrance);
    panel::tip(i18n::tr("When in doubt, keep the colors low (0.10 or less)."));
  } else if (step == CalibrationStep::Glow) {
    panel::note(
        i18n::tr("2/3 · Dark scene with a glow (magic, a lamp at night). The background stays that of a "
                 "dark scene; set the middle gray, the deep blacks and the colors so the glow keeps its "
                 "shape and its halo stays soft."));
    gray_slider(i18n::tr("Middle gray, glow scene"), g.adaptive_glow_gray);
    wizard_slider(i18n::tr("Deep blacks, glow scene"), g.adaptive_glow_shadows, kShadows);
    wizard_slider(i18n::tr("Colors, glow scene"), g.adaptive_glow_vibrance, kVibrance);
  } else {
    panel::note(
        i18n::tr("3/3 · Bright scene. Set the middle gray, the deep blacks and the colors cut while "
                 "looking at the example on the left, played like your episodes."));
    gray_slider(i18n::tr("Middle gray, bright scene"), g.adaptive_gray);
    wizard_slider(i18n::tr("Deep blacks, bright scene"), g.adaptive_shadows, kShadows);
    wizard_slider(i18n::tr("Colors cut, bright scene"), g.adaptive_vibrance_cut, kAdaptiveCut);
  }
  example_image(choices, wizard, scale, panel_left, actions);
  const bool last = step == CalibrationStep::Bright;
  if (step != CalibrationStep::Dark) {
    if (ImGui::Button(i18n::tr("Back"))) {
      --wizard.step;
      wizard.image = 0;
    }
    ImGui::SameLine();
  }
  if (ImGui::Button(last ? i18n::tr("Finish") : i18n::tr("Next"))) {
    if (!last) {
      ++wizard.step;
      wizard.image = 0;
    } else {
      wizard.step = 0;
      g.adaptive_force = -1.0;
      keep_calibrated_grade(d, choices.display_key, wizard.profile);
    }
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
  if (!now.live || !d.grade_values.adaptive || d.grade_values.adaptive_force >= 0.0) return;
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
