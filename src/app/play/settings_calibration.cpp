#include "app/play/settings_calibration.h"

#include <algorithm>
#include <cmath>

#include <imgui.h>

#include "app/play/i18n.h"
#include "app/play/panel_style.h"
#include "render/grade_params.h"

namespace ajp::app::play {

namespace {

// Le curseur va au-delà des pics que le HDR accepte : un écran à 2 500 nits doit pouvoir le montrer.
constexpr int kSliderLo = 100;
constexpr int kSliderHi = 4000;

// La calibration fermée : ce qui est mesuré sur cet écran, le bouton qui ouvre la mire et celui qui oublie la
// mesure. Vrai : la mesure est oubliée.
bool closed_calibration(PlaySettings& d, const SettingsChoices& choices,
                        std::optional<HdrPattern>& calibration) {
  panel::note(i18n::tr(
      "Measures the real peak of the display with a test pattern, instead of the EDID value, which is "
      "often wrong."));
  const int calibrated = calibrated_peak(d, choices.display_key);
  if (calibrated > 0)
    panel::note(i18n::tr_format("Calibrated on this display: {} nits (used by Auto peak).", calibrated));
  if (ImGui::Button(i18n::tr("Calibrate the peak"))) {
    HdrPattern pattern;
    pattern.nits = d.true_hdr_peak > 0 ? d.true_hdr_peak : std::max(choices.auto_peak, kSliderLo);
    calibration = pattern;
  }
  if (calibrated <= 0) return false;
  ImGui::SameLine();
  if (!ImGui::Button(i18n::tr("Forget the calibration"))) return false;
  set_calibrated_peak(d, choices.display_key, 0);
  return true;
}

}  // namespace

bool peak_calibration(PlaySettings& d, const SettingsChoices& choices,
                      std::optional<HdrPattern>& calibration) {
  panel::section(i18n::tr("Peak calibration"));
  bool reopen = false;
  if (!choices.hdr_output) {
    calibration.reset();
    panel::note(i18n::tr("The display receives no HDR: the calibration needs an HDR output."));
  } else if (!calibration) {
    reopen = closed_calibration(d, choices, calibration);
  } else {
    HdrPattern& p = *calibration;
    panel::note(
        i18n::tr("Raise the slider until the small square disappears into its frame, then stop: "
                 "that value is the real peak of the display."));
    ImGui::SliderInt(i18n::tr("Square (nits)"), &p.nits, kSliderLo, kSliderHi, "%d",
                     ImGuiSliderFlags_AlwaysClamp);
    ImGui::SameLine();
    if (ImGui::SmallButton("-")) p.nits = std::max(p.nits - 10, kSliderLo);
    ImGui::SameLine();
    if (ImGui::SmallButton("+")) p.nits = std::min(p.nits + 10, kSliderHi);
    bool large = p.area > 0.3;
    if (ImGui::Checkbox(i18n::tr("Large window (50 %)"), &large)) p.area = large ? 0.50 : 0.10;
    panel::tip(i18n::tr("Many displays dim a large bright area (ABL): the 50 % window shows how much."));
    const int peak = std::clamp(p.nits, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi);
    if (p.nits != peak)
      panel::note(i18n::tr_format("The HDR peak goes from {} to {} nits: {} is used.",
                                  limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi, peak));
    panel::note(i18n::tr(
        "If the square never disappears, the display tone maps by itself: turn on HGIG, or turn off its "
        "dynamic tone mapping, then try again."));
    const bool use = ImGui::Button(i18n::tr_format("Use {} nits as the peak", peak).c_str());
    ImGui::SameLine();
    const bool cancel = ImGui::Button(i18n::tr("Cancel"));
    if (use && choices.display_key.empty()) {
      d.true_hdr_peak = peak;  // écran inconnu : un pic manuel, en direct
    } else if (use) {
      set_calibrated_peak(d, choices.display_key, peak);
      d.true_hdr_peak = 0;  // le pic automatique reprend la mesure de cet écran
      reopen = true;
    }
    if (use || cancel) calibration.reset();
  }
  panel::end_group();
  return reopen;
}

namespace {

bool wizard_slider(const char* label, double& value, core::Range<double> range) {
  auto shown = static_cast<float>(value);
  if (!ImGui::SliderFloat(label, &shown, static_cast<float>(range.lo), static_cast<float>(range.hi), "%.2f",
                          ImGuiSliderFlags_AlwaysClamp))
    return false;
  value = std::clamp(std::round(static_cast<double>(shown) * 100.0) / 100.0, range.lo, range.hi);
  return true;
}

}  // namespace

void grade_calibration(PlaySettings& d, const SettingsChoices& choices, GradeWizard& wizard) {
  using namespace render::grade_limits;
  render::GradeSettings& g = d.grade_values;
  panel::section(i18n::tr("Adaptive grade calibration"));
  if (!choices.hdr_output || choices.source_width <= 0) {
    wizard.step = 0;
    g.adaptive_force = -1.0;
    panel::note(i18n::tr("Open a video on an HDR display to calibrate the adaptive grade."));
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
        i18n::tr("1/2 · Dark scene. Go to a dark scene (Re:Zero at night, for example), then set the "
                 "middle gray, the deep blacks and the colors until the picture looks right."));
    ImGui::SliderInt(i18n::tr("Mid gray"), &d.true_hdr_gray, limits::kTrueHdrGray.lo, limits::kTrueHdrGray.hi,
                     "%d", ImGuiSliderFlags_AlwaysClamp);
    wizard_slider(i18n::tr("Deep blacks"), g.shadows, kShadows);
    wizard_slider(i18n::tr("Colors"), g.vibrance_rtx, kVibrance);
    panel::tip(i18n::tr("When in doubt, keep the colors low (0.10 or less)."));
  } else {
    panel::note(
        i18n::tr("2/2 · Bright scene. Go to a bright, warm scene (Slime at 18:20, for example), then "
                 "set the middle gray, the deep blacks and the colors cut until nothing turns red."));
    auto gray = static_cast<float>(g.adaptive_gray);
    if (ImGui::SliderFloat(i18n::tr("Middle gray, bright scene"), &gray, static_cast<float>(kMiddleGray.lo),
                           static_cast<float>(kMiddleGray.hi), "%.0f", ImGuiSliderFlags_AlwaysClamp))
      g.adaptive_gray = std::clamp(std::round(static_cast<double>(gray)), kMiddleGray.lo, kMiddleGray.hi);
    wizard_slider(i18n::tr("Deep blacks, bright scene"), g.adaptive_shadows, kShadows);
    wizard_slider(i18n::tr("Colors cut, bright scene"), g.adaptive_vibrance_cut, kAdaptiveCut);
  }
  if (!dark) {
    if (ImGui::Button(i18n::tr("Back"))) wizard.step = 1;
    ImGui::SameLine();
  }
  if (ImGui::Button(dark ? i18n::tr("Next") : i18n::tr("Finish"))) {
    if (dark) {
      wizard.step = 2;
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
