// Référence du warp arrière par flot et du mélange de deux images (spec/warp.h). Repris de
// animejanai-inference-rtx@118c272 src/kernels.cu:963-1096 (sample_plane, sample_window,
// k_warp_blend, k_flow_max, k_warp_blend_mid). Tests : tests/unit/kernels/warp_test.cpp.
#pragma once

#include <array>

#include "kernels/image.h"
#include "kernels/spec/warp.h"

namespace ajp::kernels::ref {

// Mouvement estimé par RIFE dans ses tenseurs remplis : 4 plans de déplacements (f0x, f0y,
// f1x, f1y) et un masque, tous de même taille, et la fenêtre de l'image dans ces plans.
struct Motion {
  std::array<PlaneF, 4> flow;
  PlaneF mask;
  Window area;
};

// Échantillon bilinéaire borné au plan (sample_plane), ou à une fenêtre (sample_window).
float sample(const PlaneF& p, float x, float y);
float sample(const PlaneF& p, const Window& area, float x, float y);

// Un mouvement conforme : plans de même taille, fenêtre non vide incluse dans les plans.
Result<void> check_motion(const Motion& motion);

// Sortie = a déformé × masque + b déformé × (1 − masque), au mouvement de l'instant estimé.
Result<RgbImage> warp_blend(const RgbImage& a, const RgbImage& b, const Motion& motion);

// Instant t de ]0, 1[ depuis le mouvement estimé au milieu de la paire ; t = 1/2 égale
// warp_blend. Refus : t hors de ]0, 1[.
Result<RgbImage> warp_blend_at(const RgbImage& a, const RgbImage& b, const Motion& motion, float t);

// Plus grand déplacement absolu dans la fenêtre, en pixels de la fenêtre (k_flow_max).
Result<float> flow_max(const Motion& motion);

}  // namespace ajp::kernels::ref
