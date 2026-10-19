// Dessin de la barre de lecture avec Dear ImGui, dans le cadre en cours (ui.cpp l'ouvre et le
// ferme). L'état vit dans Controls ; ce qui est demandé (seek, pause, volume…) revient en Input.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "app/play/controls.h"
#include "app/play/input.h"
#include "app/play/tracks.h"

namespace ajp::app::play {

struct UiLook {
  float scale = 1.0F;         // échelle de l'écran (densité de pixels × échelle du bureau)
  std::string title;          // nom du fichier, en haut
  TrackMenu tracks;           // menus des pistes audio et sous-titres
  bool keep_visible = false;  // le menu des paramètres est ouvert : la barre reste
};

// Hauteur de la barre en bas de l'écran, en pixels : les sous-titres passent au-dessus quand elle
// est visible.
float controls_height(float scale);
// Hauteur du titre en haut de l'écran, en pixels : les lignes de mesure passent dessous.
float title_height(float scale);

// Ce que la superposition montre avant la lecture, pendant la préparation des moteurs : une fine
// barre à la place de la barre de progression, un libellé court et le pourcentage ; les détails
// (étape, temps, moteur) seulement avec les statistiques.
struct PrepareView {
  std::string label;                 // « Préparation du moteur SPAN »
  std::optional<float> fraction;     // rien : durée inconnue (chargement), barre sans pourcentage
  std::vector<std::string> details;  // affichés avec les statistiques (Tab)
  bool show_details = false;
};
void draw_preparation(const PrepareView& view, const UiLook& look, std::int64_t now_us);

// Dessine la barre, le titre et le message bref s'ils sont visibles ; rend true si quelque chose
// l'est. Les tailles sont en pixels de la sortie, multipliées par look.scale.
bool draw_controls(Controls& controls, const ControlsView& view, const UiLook& look, std::int64_t now_us,
                   std::vector<Input>& actions);

}  // namespace ajp::app::play
