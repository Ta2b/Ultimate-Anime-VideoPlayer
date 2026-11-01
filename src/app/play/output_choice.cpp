#include "app/play/output_choice.h"

#include <algorithm>

#include <imgui.h>

#include "app/play/i18n.h"
#include "app/play/panel_style.h"

namespace ajp::app::play {

OutputPick draw_output_choice(float scale) {
  const ImGuiIO& io = ImGui::GetIO();
  const float pad = 24.0F * scale;
  ImGui::SetNextWindowPos({io.DisplaySize.x * 0.5F, io.DisplaySize.y * 0.5F}, ImGuiCond_Always, {0.5F, 0.5F});
  ImGui::SetNextWindowSize({std::min(640.0F * scale, io.DisplaySize.x - 2.0F * pad), 0.0F}, ImGuiCond_Always);
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
  OutputPick picked = OutputPick::None;
  if (ImGui::Begin(i18n::tr("HDR or SDR"), nullptr, kFlags)) {
    panel::section(i18n::tr("The machine calibration is done"));
    panel::note(
        i18n::tr("This display can receive HDR. In HDR, the player then opens the adaptive grade "
                 "calibration: a few dark scenes, then a few bright ones, to tune the picture for this "
                 "display. In SDR, there is nothing more to set."));
    const float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5F;
    const float tall = ImGui::GetFrameHeight() * 1.6F;
    if (ImGui::Button(i18n::tr("HDR: calibrate the grade"), {half, tall})) picked = OutputPick::Hdr;
    ImGui::SameLine();
    if (ImGui::Button(i18n::tr("SDR"), {half, tall})) picked = OutputPick::Sdr;
    panel::note(i18n::tr("Change it later in HDR and color > HDR."));
  }
  ImGui::End();
  return picked;
}

}  // namespace ajp::app::play
