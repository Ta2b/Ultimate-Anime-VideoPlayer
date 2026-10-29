#include "app/play/home_view.h"

#include <algorithm>

#include <imgui.h>

#include "app/play/i18n.h"
#include "app/play/panel_style.h"

namespace ajp::app::play {

using panel::kAccent;
using panel::kWarn;
using panel::note;

void draw_home(HomeView& view, float scale, bool settings_open) {
  const ImGuiIO& io = ImGui::GetIO();
  const ImGuiStyle& style = ImGui::GetStyle();
  const float pad = 18.0F * scale;
  // La largeur du menu des paramètres (settings_panel.cpp) ; trop étroit à côté, on le chevauche.
  const float panel = settings_open ? std::min(760.0F * scale, io.DisplaySize.x - 2.0F * pad) + pad : 0.0F;
  float room = io.DisplaySize.x - panel;
  if (room < 420.0F * scale) room = io.DisplaySize.x;
  const float width = std::max(std::min(560.0F * scale, room - 2.0F * pad), 120.0F * scale);
  ImGui::SetNextWindowPos({room * 0.5F, io.DisplaySize.y * 0.5F}, ImGuiCond_Always, {0.5F, 0.5F});
  ImGui::SetNextWindowSize({width, 0.0F}, ImGuiCond_Always);  // hauteur : son contenu
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing;
  if (ImGui::Begin("##accueil", nullptr, kFlags)) {
    ImGui::PushFont(nullptr, style.FontSizeBase * 2.2F);
    ImGui::TextUnformatted("ajp");
    ImGui::PopFont();
    note(
        i18n::tr("Upscaling, interpolation and HDR for anime: set the startup profile and settings here "
                 "before playback."));
    ImGui::Spacing();
    const float tall = ImGui::GetFrameHeight() * 1.6F;
    ImGui::BeginDisabled(view.dialog_open);
    ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{0.05F, 0.07F, 0.12F, 1.0F});
    if (ImGui::Button(i18n::tr("Open a video"), {-1.0F, tall})) view.open_file = true;
    ImGui::PopStyleColor(2);
    ImGui::EndDisabled();
    note(view.dialog_open ? i18n::tr("Pick the video in the open window.")
                          : i18n::tr("or drop a file on this window."));
    if (!view.problem.empty()) {
      ImGui::Spacing();
      ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
      ImGui::TextWrapped("%s", view.problem.c_str());
      ImGui::PopStyleColor();
    }
    ImGui::Spacing();
    if (!settings_open && ImGui::Button(i18n::tr("Settings (p)"))) view.show_settings = true;
    if (!settings_open) ImGui::SameLine();
    note(i18n::tr("q: quit"));
  }
  ImGui::End();
}

}  // namespace ajp::app::play
