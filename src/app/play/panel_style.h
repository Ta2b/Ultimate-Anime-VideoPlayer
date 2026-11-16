// Les couleurs, notes et titres de section communs au menu des paramètres et à ses onglets.
#pragma once

#include <algorithm>
#include <string>
#include <string_view>

#include <imgui.h>

#include "app/play/french_text.h"
#include "app/play/i18n.h"
#include "app/play/panel_state.h"
#include "app/play/panel_tip.h"

namespace ajp::app::play::panel {

using play::french_spaces;

inline constexpr ImVec4 kAccent{0.45F, 0.70F, 1.0F, 1.0F};
inline constexpr ImVec4 kNote{0.74F, 0.76F, 0.82F, 1.0F};  // assez clair pour se lire sur le fond du menu
inline constexpr ImVec4 kWarn{1.0F, 0.76F, 0.40F, 1.0F};
inline constexpr ImVec4 kTitle{0.86F, 0.90F, 1.0F, 1.0F};       // titres des encadrés
inline constexpr ImVec4 kGood{0.45F, 0.82F, 0.52F, 1.0F};       // pastille Recommandé
inline constexpr ImVec4 kFine{0.72F, 0.58F, 1.0F, 1.0F};        // pastille Le plus fin
inline constexpr ImVec4 kExperimental{1.0F, 0.5F, 0.5F, 1.0F};  // pastille Expérimental

// L'état de mise en page du menu (panel_state.h), porté par le contexte ImGui.
inline State& state() {
  return *static_cast<State*>(ImGui::GetIO().UserData);
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

// Le libellé d'un champ à gauche, en colonne comme chez Dolphin, et le champ sur le reste de la ligne : rend
// l'identifiant caché à passer au widget, `ImGui::BeginCombo(panel::left(label).c_str(), …)`. Hors d'un
// encadré (l'assistant), le libellé reste à droite du champ, comme ImGui le dessine.
inline std::string left(const char* label) {
  if (ImGui::GetIO().UserData == nullptr || !group_open()) return label;
  const std::string_view whole(label);
  const std::string shown = french_spaces(std::string(whole.substr(0, whole.find("##"))));
  const ImGuiStyle& style = ImGui::GetStyle();
  const float column = std::max(ImGui::GetContentRegionAvail().x * 0.36F,
                                ImGui::CalcTextSize(shown.c_str()).x + style.ItemSpacing.x * 2.0F);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(shown.c_str());
  ImGui::SameLine(column + style.WindowPadding.x);
  ImGui::SetNextItemWidth(-1.0F);
  return "##" + std::string(whole);
}
inline std::string left(const std::string& label) {
  return left(label.c_str());
}
// Une liste déroulante, son libellé à gauche (left).
inline bool combo(const char* label, const char* preview) {
  return ImGui::BeginCombo(left(label).c_str(), preview);
}

inline void section(const char* title) {
  if (page_open()) {
    end_group();
    ImGui::Spacing();
    // Le titre de l'encadré en gras (la police grasse de l'atlas, ui.cpp : load_fonts), comme chez Dolphin.
    ImGui::PushStyleColor(ImGuiCol_Text, kTitle);
    if (state().bold != nullptr) ImGui::PushFont(state().bold, 0.0F);
    ImGui::TextUnformatted(title);
    if (state().bold != nullptr) ImGui::PopFont();
    ImGui::PopStyleColor();
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0F);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(1.0F, 1.0F, 1.0F, 0.025F));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0F, 1.0F, 1.0F, 0.12F));
    ImGui::BeginChild(title, ImVec2(0, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    ImGui::PopStyleColor(2);
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

// Après un curseur qui s'applique en direct : tenu, il efface le panneau, pour voir l'image changer dessous.
inline void live_preview() {
  if (ImGui::IsItemActive()) state().dim = true;
}

// Le bandeau « Avancé » en bas d'un onglet : un titre et un filet, puis les encadrés des réglages rares,
// toujours visibles (plus de case « Réglages avancés »).
inline void advanced_header() {
  end_group();
  ImGui::Dummy(ImVec2(0.0F, ImGui::GetFontSize() * 0.8F));
  ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
  ImGui::TextUnformatted(i18n::tr("Advanced"));
  ImGui::PopStyleColor();
  ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(kAccent.x, kAccent.y, kAccent.z, 0.35F));
  ImGui::Separator();
  ImGui::PopStyleColor();
}

// Place le bouton `label` (à venir) sur la ligne du dernier élément s'il y tient, sinon à la ligne : une
// rangée de boutons ne déborde pas du cadre quand la fenêtre est étroite ou l'échelle grande.
inline void same_line_if_fits(const char* label) {
  const ImGuiStyle& style = ImGui::GetStyle();
  const float need = ImGui::CalcTextSize(label).x + 2.0F * style.FramePadding.x;
  const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
  if (ImGui::GetItemRectMax().x + style.ItemSpacing.x + need <= right) ImGui::SameLine();
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
