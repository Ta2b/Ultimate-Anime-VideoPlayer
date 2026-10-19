// Référence du mouvement tiré du flux matériel (spec/optical_flow.h). Tests :
// tests/unit/kernels/optical_flow_test.cpp.
#pragma once

#include <cstdint>

#include "kernels/image.h"
#include "kernels/ref/warp.h"
#include "kernels/spec/optical_flow.h"

namespace ajp::kernels::ref {

// Un flot en pixels de la grille : composantes x et y, de même taille.
struct FlowField {
  PlaneF x;
  PlaneF y;
};

// Un flot S10.5 du flux matériel, composante x et composante y, converti en pixels de la
// grille (flow_to_grid). Refus : grille hors de 1, 2, 4 ; plan hors contrat.
Result<FlowField> decode_flow(const Plane<std::int16_t>& x, const Plane<std::int16_t>& y, int grid);

// Le mouvement au milieu de la paire depuis le flot avant (a → b) et arrière (b → a), de même
// taille ; la fenêtre est la grille entière. Refus : flots de tailles différentes ou hors contrat.
Result<Motion> mid_motion(const FlowField& forward, const FlowField& backward);

}  // namespace ajp::kernels::ref
