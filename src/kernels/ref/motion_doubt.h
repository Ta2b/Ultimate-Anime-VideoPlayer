// Référence du fondu local du mouvement de RIFE (spec/motion_doubt.h). Tests :
// tests/unit/kernels/motion_doubt_test.cpp.
#pragma once

#include <cstdint>
#include <span>

#include "kernels/ref/warp.h"

namespace kirari::kernels::ref {

// La grille de doute posée sur la fenêtre du mouvement : cols × rows poids, ligne par ligne.
struct DoubtGrid {
  int cols = 0;
  int rows = 0;
  std::span<const std::uint8_t> weights;
};

// Le mouvement tiré vers le fondu de masque fade selon la grille. Refus : mouvement hors contrat, grille
// vide, plus grande que kDoubtMaxCells de côté ou d'une autre taille que cols × rows, fade hors de [0, 1].
Result<Motion> motion_doubt(const Motion& motion, const DoubtGrid& grid, float fade);

}  // namespace kirari::kernels::ref
