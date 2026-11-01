#include "app/play/settings_calibration.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <imgui.h>

#include "app/play/calibration_images.h"
#include "app/play/chain.h"
#include "app/play/i18n.h"
#include "app/play/panel_style.h"
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

// L'image d'exemple de l'étape : sa navigation, ce qu'il faut y voir, et sa place à gauche du panneau, en
// 16:9 dans la place libre.
void example_image(const PlaySettings& d, const SettingsChoices& choices, GradeWizard& wizard, float scale,
                   float panel_left) {
  const bool dark = wizard.step == 1;
  const CalibrationImage image = calibration_image(dark, wizard.image);
  panel::note(i18n::tr_format("Example {}/{} · {}", wizard.image + 1, kCalibrationPerStep,
                              i18n::tr_view(image.title)));
  if (ImGui::Button(i18n::tr("Previous example")))
    wizard.image = (wizard.image + kCalibrationPerStep - 1) % kCalibrationPerStep;
  ImGui::SameLine();
  if (ImGui::Button(i18n::tr("Next example"))) wizard.image = (wizard.image + 1) % kCalibrationPerStep;
  panel::tip(std::string(i18n::tr_view(image.guidance)));
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
  CalibrationShot& shot = wizard.shot;
  shot.id = image.id;
  shot.width = image.width;
  shot.height = image.height;
  shot.rgba = image.rgba;
  shot.x0 = static_cast<int>(x0);
  shot.y0 = static_cast<int>(y0);
  shot.x1 = static_cast<int>(x0 + w);
  shot.y1 = static_cast<int>(y0 + h);
  // Le grade de la route HDR adaptatif (libplacebo), scène forcée : l'image est un anime SDR.
  PlaySettings graded = d;
  graded.true_hdr = false;
  graded.hdr_native = true;
  shot.grade = grade_of(graded, false, 0, choices.auto_peak);
}

}  // namespace

void grade_calibration(PlaySettings& d, const SettingsChoices& choices, GradeWizard& wizard, float scale,
                       float panel_left) {
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
  const bool calibrated = calibrated_grade(d, choices.display_key) != nullptr;
  if (wizard.step == 0) {
    g.adaptive_force = -1.0;
    panel::note(
        i18n::tr("Sets the two ends of the adaptive grade on this display: a dark scene, then a bright "
                 "scene. The result is kept for this display and drives the grade."));
    if (calibrated) panel::note(i18n::tr("This display has a calibrated adaptive grade."));
    if (ImGui::Button(i18n::tr("Calibrate the adaptive grade"))) {
      wizard.before = d;
      d.grade = true;
      g.adaptive = true;
      wizard.step = 1;
      wizard.image = 0;
    }
    if (calibrated) {
      ImGui::SameLine();
      if (ImGui::Button(i18n::tr("Forget the grade calibration")))
        forget_calibrated_grade(d, choices.display_key);
    }
    panel::end_group();
    return;
  }
  const bool dark = wizard.step == 1;
  g.adaptive_force = dark ? 0.0 : 1.0;
  if (dark) {
    panel::note(
        i18n::tr("1/2 · Dark scene. Set the middle gray, the deep blacks and the colors while looking at "
                 "the examples on the left (or at a dark scene of your video)."));
    ImGui::SliderInt(i18n::tr("Mid gray"), &d.true_hdr_gray, limits::kTrueHdrGray.lo, limits::kTrueHdrGray.hi,
                     "%d", ImGuiSliderFlags_AlwaysClamp);
    wizard_slider(i18n::tr("Deep blacks"), g.shadows, kShadows);
    wizard_slider(i18n::tr("Colors"), g.vibrance_rtx, kVibrance);
    panel::tip(i18n::tr("When in doubt, keep the colors low (0.10 or less)."));
  } else {
    panel::note(
        i18n::tr("2/2 · Bright scene. Set the middle gray, the deep blacks and the colors cut while "
                 "looking at the examples on the left (or at a bright scene of your video)."));
    auto gray = static_cast<float>(g.adaptive_gray);
    if (ImGui::SliderFloat(i18n::tr("Middle gray, bright scene"), &gray, static_cast<float>(kMiddleGray.lo),
                           static_cast<float>(kMiddleGray.hi), "%.0f", ImGuiSliderFlags_AlwaysClamp))
      g.adaptive_gray = std::clamp(std::round(static_cast<double>(gray)), kMiddleGray.lo, kMiddleGray.hi);
    wizard_slider(i18n::tr("Deep blacks, bright scene"), g.adaptive_shadows, kShadows);
    wizard_slider(i18n::tr("Colors cut, bright scene"), g.adaptive_vibrance_cut, kAdaptiveCut);
  }
  example_image(d, choices, wizard, scale, panel_left);
  if (!dark) {
    if (ImGui::Button(i18n::tr("Back"))) {
      wizard.step = 1;
      wizard.image = 0;
    }
    ImGui::SameLine();
  }
  if (ImGui::Button(dark ? i18n::tr("Next") : i18n::tr("Finish"))) {
    if (dark) {
      wizard.step = 2;
      wizard.image = 0;
    } else {
      wizard.step = 0;
      g.adaptive_force = -1.0;
      keep_calibrated_grade(d, choices.display_key);
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

}  // namespace ajp::app::play
