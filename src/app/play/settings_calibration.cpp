#include "app/play/settings_calibration.h"

#include <algorithm>

#include <imgui.h>

#include "app/play/i18n.h"
#include "app/play/panel_style.h"

namespace ajp::app::play {

namespace {

// Le curseur va au-delà des pics que le HDR accepte : un écran à 2 500 nits doit pouvoir le montrer.
constexpr int kSliderLo = 100;
constexpr int kSliderHi = 4000;

}  // namespace

void peak_calibration(PlaySettings& d, const SettingsChoices& choices,
                      std::optional<HdrPattern>& calibration) {
  panel::section(i18n::tr("Peak calibration"));
  if (!choices.hdr_output) {
    calibration.reset();
    panel::note(i18n::tr("The display receives no HDR: the calibration needs an HDR output."));
    panel::end_group();
    return;
  }
  if (!calibration) {
    panel::note(
        i18n::tr("Measures the real peak of the display with a test pattern, instead of the EDID value, "
                 "which is often wrong."));
    if (ImGui::Button(i18n::tr("Calibrate the peak"))) {
      HdrPattern pattern;
      pattern.nits = d.true_hdr_peak > 0 ? d.true_hdr_peak : std::max(choices.auto_peak, kSliderLo);
      calibration = pattern;
    }
    panel::end_group();
    return;
  }
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
  if (ImGui::Button(i18n::tr_format("Use {} nits as the peak", peak).c_str())) {
    d.true_hdr_peak = peak;
    calibration.reset();
  }
  ImGui::SameLine();
  if (ImGui::Button(i18n::tr("Cancel"))) calibration.reset();
  if (p.nits != peak)
    panel::note(i18n::tr_format("The HDR peak goes from {} to {} nits: {} is used.", limits::kTrueHdrPeak.lo,
                                limits::kTrueHdrPeak.hi, peak));
  panel::note(
      i18n::tr("If the square never disappears, the display tone maps by itself: turn on HGIG, or turn off "
               "its dynamic tone mapping, then try again."));
  panel::end_group();
}

}  // namespace ajp::app::play
