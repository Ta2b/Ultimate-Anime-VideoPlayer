#include "app/play/home_view.h"

#include <algorithm>

#include <imgui.h>

#include "app/play/i18n.h"
#include "app/play/panel_style.h"

namespace ajp::app::play {

using panel::kAccent;
using panel::kWarn;
using panel::note;

void draw_home(HomeView& view, float scale, std::optional<PanelSpan> settings) {
  const ImGuiIO& io = ImGui::GetIO();
  const ImGuiStyle& style = ImGui::GetStyle();
  const float pad = 18.0F * scale;
  // La carte grandit avec la fenêtre, texte compris : à 1280 × 720 (à l'échelle de l'interface) elle a sa
  // taille de base, au-delà elle suit le plus petit des deux rapports, jusqu'à trois fois.
  const float grow = std::clamp(
      std::min(io.DisplaySize.x / (1280.0F * scale), io.DisplaySize.y / (720.0F * scale)), 1.0F, 3.0F);
  // À côté du menu des paramètres, dans la plus grande des deux places qu'il laisse ; trop étroit, on le
  // chevauche.
  float from = 0.0F;
  float room = io.DisplaySize.x;
  if (settings) {
    const float left = settings->left;
    const float right = io.DisplaySize.x - settings->right;
    const float best = std::max(left, right);
    if (best >= 420.0F * scale) {
      from = left >= right ? 0.0F : settings->right;
      room = best;
    }
  }
  const float width = std::max(std::min(560.0F * scale * grow, room - 2.0F * pad), 120.0F * scale);
  ImGui::SetNextWindowPos({from + room * 0.5F, io.DisplaySize.y * 0.5F}, ImGuiCond_Always, {0.5F, 0.5F});
  ImGui::SetNextWindowSize({width, 0.0F}, ImGuiCond_Always);  // hauteur : son contenu
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                      {style.WindowPadding.x * grow, style.WindowPadding.y * grow});
  const bool shown = ImGui::Begin("##accueil", nullptr, kFlags);
  ImGui::PopStyleVar();
  if (shown) {
    ImGui::PushFont(nullptr, style.FontSizeBase * grow);
    ImGui::PushFont(nullptr, style.FontSizeBase * 2.2F * grow);
    ImGui::TextUnformatted("ajp");
    ImGui::PopFont();
    note(
        i18n::tr("Upscaling, interpolation and HDR for anime: adjust the settings here before "
                 "playback."));
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
      panel::wrapped(view.problem.c_str());
      ImGui::PopStyleColor();
    }
    ImGui::Spacing();
    if (!settings && ImGui::Button(i18n::tr("Settings (p)"))) view.show_settings = true;
    if (!settings) ImGui::SameLine();
    note(i18n::tr("q: quit"));
    ImGui::PopFont();
  }
  ImGui::End();
}

}  // namespace ajp::app::play
