// Les couleurs, notes et titres de section communs au menu des paramètres et à ses onglets.
#pragma once

#include <algorithm>
#include <string>
#include <string_view>

#include <imgui.h>
#include <imgui_internal.h>  // GetCurrentTable : la colonne des libellés d'une grille

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
// Le cadre d'un encadré, son titre posé dans la bordure du haut, comme les QGroupBox de Dolphin : un filet
// fin aux coins arrondis, comme le reste de l'interface, interrompu par le titre en gras.
inline void frame_group(const char* title, ImVec2 min, ImVec2 max) {
  ImDrawList* draw = ImGui::GetWindowDrawList();
  const ImU32 line = ImGui::GetColorU32(ImVec4(1.0F, 1.0F, 1.0F, 0.16F));
  max.x -= 1.0F;  // le filet de droite reste dans la fenêtre fille, hors de la barre de défilement
  const float unit = ImGui::GetFontSize() / 16.0F;
  if (state().bold != nullptr) ImGui::PushFont(state().bold, 0.0F);
  const ImVec2 text = ImGui::CalcTextSize(title);
  const float start = min.x + 10.0F * unit;
  const float gap = 4.0F * unit;
  const float r = 6.0F * unit;
  // De la fin du titre, dans le sens des aiguilles d'une montre, jusqu'à son début (angles en douzièmes de
  // tour : 0 à droite, 3 en bas, 6 à gauche, 9 en haut).
  draw->PathLineTo(ImVec2(start + text.x + gap, min.y));
  draw->PathArcToFast(ImVec2(max.x - r, min.y + r), r, 9, 12);
  draw->PathArcToFast(ImVec2(max.x - r, max.y - r), r, 0, 3);
  draw->PathArcToFast(ImVec2(min.x + r, max.y - r), r, 3, 6);
  draw->PathArcToFast(ImVec2(min.x + r, min.y + r), r, 6, 9);
  draw->PathLineTo(ImVec2(start - gap, min.y));
  draw->PathStroke(line, ImDrawFlags_None, 1.0F);
  draw->AddText(ImVec2(start, min.y - text.y * 0.5F), ImGui::GetColorU32(kTitle),
                title);  // grisé avec l'encadré
  if (state().bold != nullptr) ImGui::PopFont();
}

inline void end_group() {
  if (group_open()) {
    ImGui::PopItemWidth();
    ImGui::EndChild();
    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    if (state().group_title != nullptr) frame_group(state().group_title, min, max);
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

// La largeur de la colonne des libellés d'un encadré ou d'une case de grille : le plus long libellé de
// l'image précédente, gardé dans la mémoire d'ImGui, pour que les champs d'une colonne s'alignent sans
// laisser de vide entre un libellé court et son champ (comme la QGridLayout de Dolphin).
inline float label_column(float text) {
  ImGuiStorage* storage = ImGui::GetStateStorage();
  const int column = ImGui::GetCurrentTable() != nullptr ? ImGui::TableGetColumnIndex() : -1;
  const ImGuiID used = ImGui::GetID(column) + 1U;  // la largeur de l'image précédente
  const ImGuiID widest = used + 1U;                // le maximum de cette image
  const ImGuiID frame = used + 2U;
  if (storage->GetInt(frame, -1) != ImGui::GetFrameCount()) {
    storage->SetFloat(used, storage->GetFloat(widest, text));
    storage->SetFloat(widest, 0.0F);
    storage->SetInt(frame, ImGui::GetFrameCount());
  }
  storage->SetFloat(widest, std::max(storage->GetFloat(widest), text));
  return std::max(storage->GetFloat(used), text);
}

// Le libellé d'un champ à gauche, en colonne comme chez Dolphin, et le champ juste après : rend l'identifiant
// caché à passer au widget, `ImGui::BeginCombo(panel::left(label).c_str(), …)`. Hors d'un encadré
// (l'assistant), le libellé reste à droite du champ, comme ImGui le dessine.
inline std::string left(const char* label) {
  if (ImGui::GetIO().UserData == nullptr || !group_open()) return label;
  const std::string_view whole(label);
  const std::string shown = french_spaces(std::string(whole.substr(0, whole.find("##"))));
  const ImGuiStyle& style = ImGui::GetStyle();
  const float start = ImGui::GetCursorScreenPos().x;
  const float column = label_column(ImGui::CalcTextSize(shown.c_str()).x) + style.ItemSpacing.x * 2.0F;
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(shown.c_str());
  ImGui::SameLine(0.0F, 0.0F);
  ImGui::SetCursorScreenPos(ImVec2(start + column, ImGui::GetCursorScreenPos().y));
  ImGui::SetNextItemWidth(-1.0F);  // le champ remplit le reste de sa case ou de sa ligne, comme chez Dolphin
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
    // Le titre se pose dans la bordure du haut à la fermeture (frame_group) : ici, la place de sa
    // demi-hauteur.
    ImGui::Dummy(ImVec2(0.0F, ImGui::GetFontSize() * 0.5F));
    state().group_title = title;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                        ImVec2(ImGui::GetStyle().WindowPadding.x + 2.0F, ImGui::GetFontSize() * 0.7F));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0F * ImGui::GetFontSize() / 16.0F);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(1.0F, 1.0F, 1.0F, 0.025F));
    ImGui::BeginChild(title, ImVec2(0, 0),
                      ImGuiChildFlags_AlwaysUseWindowPadding | ImGuiChildFlags_AutoResizeY);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
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

// Deux réglages côte à côte dans un encadré, comme les grilles de Dolphin : chaque cell() ouvre la case
// suivante, de gauche à droite puis à la ligne. Un libellé à gauche (left) se place dans sa case.
// Deux colonnes égales séparées d'un écart fixe ; les champs remplissent leur case, comme chez Dolphin, et
// deux réglages voisins ne sont séparés que par cet écart.
inline bool begin_grid(const char* id = "##grille") {
  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,
                      ImVec2(ImGui::GetFontSize() * 1.25F, ImGui::GetStyle().CellPadding.y));
  const bool open = ImGui::BeginTable(
      id, 2,
      ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_NoPadOuterX);
  ImGui::PopStyleVar();
  return open;
}
inline void cell() {
  ImGui::TableNextColumn();
}
inline void end_grid() {
  ImGui::EndTable();
}

// Des encadrés côte à côte, deux par ligne : next_box() avant chaque section, end_boxes() à la fin.
inline bool begin_boxes() {
  end_group();
  const float gap = ImGui::GetStyle().ItemSpacing.x;
  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(gap * 0.5F, 0.0F));
  const bool open = ImGui::BeginTable(
      "##encadres", 2,
      ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_NoPadOuterX);
  ImGui::PopStyleVar();
  return open;
}
inline void next_box() {
  end_group();
  ImGui::TableNextColumn();
}
inline void end_boxes() {
  end_group();
  ImGui::EndTable();
}

// Un curseur à la manière de Dolphin : un rail fin sur toute la largeur, une poignée, et la valeur écrite au
// bout, hors du rail. `value` est le texte de la valeur en cours (« 144 i/s ») ; sa place est réservée sur
// trois caractères au moins, pour que le rail ne bouge pas avec la valeur. `draw` dessine le curseur ImGui
// (sans texte) à la largeur posée.
template <class Draw>
bool dolphin_slider(const char* id, const std::string& value, Draw draw) {
  const ImGuiStyle& style = ImGui::GetStyle();
  const float unit = ImGui::GetFontSize() / 16.0F;
  const float text = std::max(ImGui::CalcTextSize(value.c_str()).x, ImGui::GetFontSize() * 2.5F);
  const float width =
      std::max(ImGui::GetFontSize() * 4.0F, ImGui::GetContentRegionAvail().x - text - style.ItemSpacing.x);
  const ImVec2 at = ImGui::GetCursorScreenPos();
  const float mid = at.y + ImGui::GetFrameHeight() * 0.5F;
  // Le rail, sous la poignée : tracé avant le curseur, dont le fond est transparent.
  ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(at.x, mid - 1.5F * unit),
                                            ImVec2(at.x + width, mid + 1.5F * unit),
                                            ImGui::GetColorU32(ImGuiCol_FrameBgActive), 2.0F * unit);
  ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, 10.0F * unit);
  ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 5.0F * unit);
  // Le texte du curseur est caché (la valeur s'écrit au bout), sauf pendant une saisie au clavier
  // (Ctrl + clic), qui le montre dans le champ.
  const bool typing = ImGui::GetCurrentContext()->TempInputId == ImGui::GetID(id);
  ImGui::PushStyleColor(ImGuiCol_Text, typing ? ImGui::GetStyleColorVec4(ImGuiCol_Text) : ImVec4(0, 0, 0, 0));
  ImGui::SetNextItemWidth(width);
  const bool changed = draw();
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor(4);
  // La valeur se dessine sans être un élément : le dernier élément reste le curseur, pour live_preview, tip
  // et IsItemActive.
  const ImVec2 end = ImGui::GetItemRectMax();
  const float y = at.y + (ImGui::GetFrameHeight() - ImGui::GetFontSize()) * 0.5F;
  ImGui::GetWindowDrawList()->AddText(ImVec2(end.x + style.ItemSpacing.x, y),
                                      ImGui::GetColorU32(ImGuiCol_Text), value.c_str());
  return changed;
}

// Flags : AlwaysClamp par défaut ; sans lui, Ctrl + clic peut saisir une valeur hors du rail (le délai
// audio).
inline bool slider_int(const char* id, int* v, int lo, int hi, const std::string& value,
                       ImGuiSliderFlags flags = ImGuiSliderFlags_AlwaysClamp) {
  return dolphin_slider(id, value, [&] { return ImGui::SliderInt(id, v, lo, hi, "%d", flags); });
}
inline bool slider_float(const char* id, float* v, float lo, float hi, const std::string& value) {
  return dolphin_slider(
      id, value, [&] { return ImGui::SliderFloat(id, v, lo, hi, "%.2f", ImGuiSliderFlags_AlwaysClamp); });
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
