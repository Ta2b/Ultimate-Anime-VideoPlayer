// Les couleurs, notes et titres de section communs au menu des paramètres et à ses onglets.
#pragma once

#include <string>

#include <imgui.h>

namespace ajp::app::play::panel {

inline constexpr ImVec4 kAccent{0.42F, 0.66F, 1.0F, 1.0F};
inline constexpr ImVec4 kNote{0.66F, 0.68F, 0.74F, 1.0F};
inline constexpr ImVec4 kWarn{1.0F, 0.78F, 0.42F, 1.0F};
inline constexpr ImVec4 kGood{0.45F, 0.82F, 0.52F, 1.0F};       // pastille Recommandé
inline constexpr ImVec4 kFine{0.72F, 0.58F, 1.0F, 1.0F};        // pastille Le plus fin
inline constexpr ImVec4 kExperimental{1.0F, 0.5F, 0.5F, 1.0F};  // pastille Expérimental

inline void note(const char* text) {
  ImGui::PushStyleColor(ImGuiCol_Text, kNote);
  ImGui::TextWrapped("%s", text);
  ImGui::PopStyleColor();
}
inline void note(const std::string& text) {
  note(text.c_str());
}

// Dans une page des réglages (begin_page … end_page), chaque section est un encadré titré ; ailleurs, un
// titre et un filet.
inline bool& page_open() {
  static bool open = false;
  return open;
}
inline bool& group_open() {
  static bool open = false;
  return open;
}
inline void end_group() {
  if (group_open()) ImGui::EndChild();
  group_open() = false;
}
inline void begin_page() {
  page_open() = true;
  group_open() = false;
}
inline void end_page() {
  end_group();
  page_open() = false;
}

inline void section(const char* title) {
  if (page_open()) {
    end_group();
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0F);
    ImGui::BeginChild(title, ImVec2(0, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    ImGui::PopStyleVar();
    group_open() = true;
    return;
  }
  ImGui::Spacing();
  ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
  ImGui::TextUnformatted(title);
  ImGui::PopStyleColor();
  ImGui::Separator();
}

// Une infobulle sur le dernier élément, après un court délai ; elle s'affiche aussi sur un élément grisé.
inline void tip(const char* text) {
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled | ImGuiHoveredFlags_DelayNormal)) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0F);
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}
inline void tip(const std::string& text) {
  tip(text.c_str());
}

// Une pastille : le texte sur un rectangle arrondi teinté, sur la ligne du dernier élément. Vert recommandé,
// bleu rapide, violet fin, ambre lourd, rouge expérimental.
inline void badge(const char* text, ImVec4 color) {
  ImGui::SameLine();
  ImVec2 at = ImGui::GetCursorScreenPos();
  const ImVec2 size = ImGui::CalcTextSize(text);
  // Centrée sur la hauteur d'un champ (case, liste) : la pastille suit le milieu du libellé, pas le haut de
  // la ligne.
  at.y += std::max(0.0F, (ImGui::GetFrameHeight() - (size.y + 2.0F)) * 0.5F);
  const float pad = 5.0F;
  ImDrawList* draw = ImGui::GetWindowDrawList();
  ImVec4 fill = color;
  fill.w = 0.28F;
  draw->AddRectFilled(at, ImVec2(at.x + size.x + 2 * pad, at.y + size.y + 2),
                      ImGui::ColorConvertFloat4ToU32(fill), 4.0F);
  draw->AddText(ImVec2(at.x + pad, at.y + 1), ImGui::ColorConvertFloat4ToU32(color), text);
  ImGui::Dummy(ImVec2(size.x + 2 * pad, size.y + 2));
}

}  // namespace ajp::app::play::panel
