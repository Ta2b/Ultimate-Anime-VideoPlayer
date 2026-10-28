// Référence CPU, en double, de la colle pleine résolution de RIFE fusionné (spec/rife_fused.h) : chaque
// opérateur du graphe ONNX (DepthToSpace, Resize bilinéaire en demi-pixel, flot accumulé, grille, GridSample
// aux coins alignés et bords répétés, Concat, Resize d'entrée du bloc suivant, masque sigmoïde, mélange)
// est appliqué l'un après l'autre, sans le raccourci des noyaux (qui ne calculent le flot et le warp que
// sur les deux fois deux pixels que le Resize d'entrée du bloc suivant relit). Sans arrondi fp16.
// Tests : tests/unit/kernels/rife_fused_test.cpp ; les noyaux CUDA s'y comparent
// (tests/unit/backend_tensorrt/rife_fused_test.cpp).
#pragma once

#include <vector>

#include "kernels/spec/rife_fused.h"

namespace ajp::kernels::ref {

// Ce que la colle lit : l'entrée de RIFE (11 plans h × w), l'encodage (8 plans) et les sorties Y_k des blocs
// déjà calculés (52 canaux à la moitié de la résolution d'entrée du bloc), dans l'ordre des blocs.
struct RifeFusedState {
  RifeFusedPlan plan;
  int height = 0;
  int width = 0;
  std::vector<double> input;
  std::vector<double> encode;
  std::vector<std::vector<double>> y;
};

// L'entrée X_k du bloc k : les y[0..k-1] doivent être là (k = 0 : aucun).
std::vector<double> rife_fused_block_input(const RifeFusedState& state, int k);

// La fin du réseau, une fois y[0..blocks-1] connus.
struct RifeFusedResult {
  std::vector<double> image;  // 3 plans h × w : mélange des deux images warpées
  std::vector<double> flow;   // 4 plans : flot final
  std::vector<double> mask;   // 1 plan : masque après sigmoïde
};
RifeFusedResult rife_fused_finish(const RifeFusedState& state);

}  // namespace ajp::kernels::ref
