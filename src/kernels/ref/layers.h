// Référence de la composition par calques (spec/layers.h). Tests : tests/unit/kernels/layers_test.cpp.
#pragma once

#include "kernels/image.h"
#include "kernels/spec/layers.h"

namespace kirari::kernels::ref {

// La composition sur place dans out, la source 0 dans a, la source 1 dans b (lue si params.count vaut 2).
// Refus : images de tailles différentes, count hors de [1, 2], grille vide ou plus grande que la limite,
// garde vide (guard_high ≤ guard_low).
Result<void> compose_layers(RgbImage& out, const RgbImage& a, const RgbImage& b, const LayerParams& params);

// Le masque d'une grille au point (x, y) de l'image qu'elle couvre (étape 1 de spec/layers.h).
float layer_mask(const LayerGrid& grid, float x, float y);

}  // namespace kirari::kernels::ref
