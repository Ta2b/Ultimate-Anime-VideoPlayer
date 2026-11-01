#include "app/play/settings_calibration.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <system_error>

#include <imgui.h>

#include "app/play/calibration_images.h"
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

// Le plan d'exemple de l'étape : sa navigation, ce qu'il faut y voir, la place de la vidéo à gauche du
// panneau (16:9 dans la place libre), et la vidéo de calibration tenue en pause sur ce plan.
void example_image(const SettingsChoices& choices, GradeWizard& wizard, float scale, float panel_left,
                   std::vector<Input>& actions) {
  const bool dark = wizard.step == 1;
  const CalibrationImage image = calibration_image(dark, wizard.image);
  panel::note(i18n::tr_format("Example {}/{} · {}", wizard.image + 1, kCalibrationPerStep,
                              i18n::tr_view(image.title)));
  if (ImGui::Button(i18n::tr("Previous example")))
    wizard.image = (wizard.image + kCalibrationPerStep - 1) % kCalibrationPerStep;
  ImGui::SameLine();
  if (ImGui::Button(i18n::tr("Next example"))) wizard.image = (wizard.image + 1) % kCalibrationPerStep;
  panel::tip(std::string(i18n::tr_view(image.guidance)));
  const std::filesystem::path clip = calibration_clip();
  wizard.showing = !clip.empty() && choices.source_file == clip;
  if (!wizard.showing) {
    panel::note(i18n::tr("Opening the calibration video…"));
  } else {
    const CalibrationImage shown = calibration_image(dark, wizard.image);
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
  const bool calibrated = calibrated_grade(d, choices.display_key) != nullptr;
  if (wizard.step == 0) {
    g.adaptive_force = -1.0;
    panel::note(
        i18n::tr("Sets the two ends of the adaptive grade on this display: a dark scene, then a bright "
                 "scene. The result is kept for this display and drives the grade."));
    if (calibrated) panel::note(i18n::tr("This display has a calibrated adaptive grade."));
    const std::filesystem::path clip = calibration_clip();
    std::error_code missing;
    if (clip.empty() || !std::filesystem::is_regular_file(clip, missing)) {
      panel::note(
          i18n::tr("The calibration video is missing (calibration/calibration.mkv next to the player): "
                   "reinstall the player."));
    } else if (ImGui::Button(i18n::tr("Calibrate the adaptive grade"))) {
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
      const CalibrationImage first = calibration_image(true, 0);
      wizard.sought = first.id;
      actions.emplace_back(PlayFile{.file = clip, .start_us = first.at_us, .paused = true});
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
                 "the example on the left, played like your episodes."));
    ImGui::SliderInt(i18n::tr("Mid gray"), &d.true_hdr_gray, limits::kTrueHdrGray.lo, limits::kTrueHdrGray.hi,
                     "%d", ImGuiSliderFlags_AlwaysClamp);
    wizard_slider(i18n::tr("Deep blacks"), g.shadows, kShadows);
    wizard_slider(i18n::tr("Colors"), g.vibrance_rtx, kVibrance);
    panel::tip(i18n::tr("When in doubt, keep the colors low (0.10 or less)."));
  } else {
    panel::note(
        i18n::tr("2/2 · Bright scene. Set the middle gray, the deep blacks and the colors cut while "
                 "looking at the example on the left, played like your episodes."));
    auto gray = static_cast<float>(g.adaptive_gray);
    if (ImGui::SliderFloat(i18n::tr("Middle gray, bright scene"), &gray, static_cast<float>(kMiddleGray.lo),
                           static_cast<float>(kMiddleGray.hi), "%.0f", ImGuiSliderFlags_AlwaysClamp))
      g.adaptive_gray = std::clamp(std::round(static_cast<double>(gray)), kMiddleGray.lo, kMiddleGray.hi);
    wizard_slider(i18n::tr("Deep blacks, bright scene"), g.adaptive_shadows, kShadows);
    wizard_slider(i18n::tr("Colors cut, bright scene"), g.adaptive_vibrance_cut, kAdaptiveCut);
  }
  example_image(choices, wizard, scale, panel_left, actions);
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
