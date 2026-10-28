// Référence du rééchantillonnage séparable : tables de poids et passes (spec/resample.h).
// Repris de animejanai-inference-rtx@118c272 src/resample.h:53-84 (compute) et
// src/kernels.cu:415-445 (k_rs_h, k_rs_v). Tests : tests/unit/kernels/resample_test.cpp.
#pragma once

#include <span>
#include <vector>

#include "kernels/image.h"
#include "kernels/spec/resample.h"

namespace ajp::kernels::ref {

struct Weights {
  int taps = 0;
  int src = 0;
  int dst = 0;
  std::vector<int> start;  // [dst] : premier indice source, avant le miroir
  std::vector<float> w;    // [dst × taps] : poids normalisés

  std::span<const float> row(int i) const;
};

// Table de dst sorties sur src entrées. Refus : dimensions hors de [1, kMaxDimension],
// décalage non fini ou hors de [−1, 1], Area qui agrandirait ou décalerait.
Result<Weights> compute_weights(int src, int dst, double shift, Filter filter);

// Une passe : horizontale (largeur src → dst) ou verticale (hauteur src → dst).
Result<PlaneF> resample_rows(const PlaneF& in, const Weights& wx);
Result<PlaneF> resample_columns(const PlaneF& in, const Weights& wy);

// Rééchantillonnage d'une image de la chaîne, sans décalage : passe horizontale puis
// verticale. Dimensions source et cible paires.
Result<PlaneF> resize(const PlaneF& in, int width, int height, Filter filter);
Result<RgbImage> resize(const RgbImage& in, int width, int height, Filter filter);

}  // namespace ajp::kernels::ref
