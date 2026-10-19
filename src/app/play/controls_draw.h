// Dessin de la barre de lecture avec Dear ImGui, dans le cadre en cours (ui.cpp l'ouvre et le
// ferme). L'état vit dans Controls ; ce qui est demandé (seek, pause, volume…) revient en Input.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "app/play/controls.h"
#include "app/play/input.h"

namespace ajp::app::play {

struct UiLook {
  float scale = 1.0F;  // échelle de l'écran (densité de pixels × échelle du bureau)
  std::string title;   // nom du fichier, en haut
};

// Hauteur de la barre en bas de l'écran, en pixels : les sous-titres passent au-dessus quand elle
// est visible.
float controls_height(float scale);
// Hauteur du titre en haut de l'écran, en pixels : les lignes de mesure passent dessous.
float title_height(float scale);

// Dessine la barre, le titre et le message bref s'ils sont visibles ; rend true si quelque chose
// l'est. Les tailles sont en pixels de la sortie, multipliées par look.scale.
bool draw_controls(Controls& controls, const ControlsView& view, const UiLook& look, std::int64_t now_us,
                   std::vector<Input>& actions);

}  // namespace ajp::app::play
