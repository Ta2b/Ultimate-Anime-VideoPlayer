// Les couleurs, notes et titres de section communs au menu des paramètres et à ses onglets.
#pragma once

#include <string>
#include <string_view>

#include <imgui.h>

#include "app/play/french_text.h"
#include "app/play/panel_state.h"

namespace ajp::app::play::panel {

using play::french_spaces;

inline constexpr ImVec4 kAccent{0.42F, 0.66F, 1.0F, 1.0F};
inline constexpr ImVec4 kNote{0.66F, 0.68F, 0.74F, 1.0F};
inline constexpr ImVec4 kWarn{1.0F, 0.78F, 0.42F, 1.0F};
inline constexpr ImVec4 kGood{0.45F, 0.82F, 0.52F, 1.0F};       // pastille Recommandé
inline constexpr ImVec4 kFine{0.72F, 0.58F, 1.0F, 1.0F};        // pastille Le plus fin
inline constexpr ImVec4 kExperimental{1.0F, 0.5F, 0.5F, 1.0F};  // pastille Expérimental

// L'état de mise en page du menu (panel_state.h), porté par le contexte ImGui.
inline State& state() {
  return *static_cast<State*>(ImGui::GetIO().UserData);
}

inline bool& advanced() {
  return state().advanced;
}

// Un texte à la ligne, avec la typographie française (A43-07) : toute phrase qui s'écrit passe par ici, ou
// par note() et tip(), jamais par ImGui::TextWrapped directement.
inline void wrapped(const std::string& text) {
  const std::string shown = french_spaces(text);
  ImGui::TextWrapped("%s", shown.c_str());
}
inline void wrapped(const char* text) {
  wrapped(std::string(text));
}

inline void note(const char* text) {
  const std::string shown = french_spaces(text);
  ImGui::PushStyleColor(ImGuiCol_Text, kNote);
  ImGui::TextWrapped("%s", shown.c_str());
  ImGui::PopStyleColor();
}
inline void note(const std::string& text) {
  note(text.c_str());
}

// Dans une page des réglages (begin_page … end_page), chaque section est un encadré titré ; ailleurs, un
// titre et un filet.
inline bool& page_open() {
  return state().page_open;
}
inline bool& group_open() {
  return state().group_open;
}
inline void end_group() {
  if (group_open()) {
    ImGui::PopItemWidth();
    ImGui::EndChild();
  }
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
    // Un encadré est une fenêtre fille : la largeur des champs repart de son défaut. Les libellés (à droite
    // du champ) gardent 44 % de la largeur, assez pour les plus longs sans les couper.
    ImGui::PushItemWidth(-ImGui::GetContentRegionAvail().x * 0.44F);
    group_open() = true;
    return;
  }
  ImGui::Spacing();
  ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
  ImGui::TextUnformatted(title);
  ImGui::PopStyleColor();
  ImGui::Separator();
}

// Place le bouton `label` (à venir) sur la ligne du dernier élément s'il y tient, sinon à la ligne : une
// rangée de boutons ne déborde pas du cadre quand la fenêtre est étroite ou l'échelle grande.
inline void same_line_if_fits(const char* label) {
  const ImGuiStyle& style = ImGui::GetStyle();
  const float need = ImGui::CalcTextSize(label).x + 2.0F * style.FramePadding.x;
  const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
  if (ImGui::GetItemRectMax().x + style.ItemSpacing.x + need <= right) ImGui::SameLine();
}

// Une infobulle sur le dernier élément, après un court délai ; elle s'affiche aussi sur un élément grisé.
inline void tip(const char* text) {
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled | ImGuiHoveredFlags_DelayNormal)) {
    const std::string shown = french_spaces(text);
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0F);
    ImGui::TextUnformatted(shown.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}
inline void tip(const std::string& text) {
  tip(text.c_str());
}

// Une pastille : le texte sur un rectangle arrondi teinté, sur la ligne du dernier élément. Vert recommandé,
// bleu rapide, violet fin, ambre lourd, rouge expérimental.
// Les mesures suivent la taille de la police, donc l'échelle de l'interface (A24-13 : marges en pixels fixes
// avant), et une pastille qui ne tient pas au bout de la ligne passe à la suivante au lieu d'être coupée par
// le bord du panneau (« Recommenc… »).
inline void badge(const char* text, ImVec4 color) {
  const ImVec2 size = ImGui::CalcTextSize(text);
  const float unit = ImGui::GetFontSize() / 16.0F;
  const float pad = 5.0F * unit;
  const float rim = 1.0F * unit;
  const ImVec2 box{size.x + 2.0F * pad, size.y + 2.0F * rim};
  ImGui::SameLine();
  if (ImGui::GetContentRegionAvail().x < box.x) ImGui::NewLine();
  ImVec2 at = ImGui::GetCursorScreenPos();
  // Centrée sur la hauteur d'un champ (case, liste) : la pastille suit le milieu du libellé, pas le haut de
  // la ligne.
  at.y += std::max(0.0F, (ImGui::GetFrameHeight() - box.y) * 0.5F);
  ImDrawList* draw = ImGui::GetWindowDrawList();
  ImVec4 fill = color;
  fill.w = 0.28F;
  draw->AddRectFilled(at, ImVec2(at.x + box.x, at.y + box.y), ImGui::ColorConvertFloat4ToU32(fill),
                      4.0F * unit);
  draw->AddText(ImVec2(at.x + pad, at.y + rim), ImGui::ColorConvertFloat4ToU32(color), text);
  ImGui::Dummy(box);
}

}  // namespace ajp::app::play::panel
