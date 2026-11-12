// Référence des décisions temporelles : score de coupe de scène et pire bloc des dessins tenus
// (spec/detect.h). Repris de animejanai-inference-rtx@118c272 src/kernels.cu:698-743
// (k_scd_diff) et 923-959 (k_block_diff). Tests : tests/unit/kernels/detect_test.cpp.
#pragma once

#include <cstdint>
#include <vector>

#include "kernels/image.h"
#include "kernels/spec/detect.h"

namespace ajp::kernels::ref {

// Σ |a − b| × scene_norm(format) sur deux plans de luma de même taille. Chaque terme se calcule
// en float comme sur GPU ; la somme est en double, sans dépendre de l'ordre.
Result<double> scene_diff_sum(const PlaneU16& a, const PlaneU16& b, Format format);

// Score de RIFE : la somme sur la luma source (width × height de la géométrie), divisée par
// l'aire remplie pw × ph. À comparer avec is_scene_cut.
Result<double> scene_score(const PlaneU16& a, const PlaneU16& b, Format format, const RifeGeometry& geometry);

// Pire écart moyen d'un bloc 16 × 16, en 1/256 de niveau 8 bits. À comparer avec is_held.
Result<std::uint32_t> block_diff_max(const PlaneU16& a, const PlaneU16& b, Format format);

// Ce que mesurent rgb_pair_diff.comp et pair_diff (spec/detect.h, « la mesure que le lecteur fait
// réellement ») sur deux images RGB ingérées : score de coupe, pire bloc, somme de chaque bloc 16 × 16.
struct PairMeasure {
  double scene_score = 0.0;
  std::uint32_t worst_block = 0;
  double total = 0.0;
  std::vector<double> block_sums;  // un par bloc, rangée après rangée
};

Result<PairMeasure> pair_scores(const RgbImage& a, const RgbImage& b, float levels, const RifeGeometry& geometry,
                                float kr = coefficients(Matrix::Bt709).kr,
                                float kb = coefficients(Matrix::Bt709).kb);

}  // namespace ajp::kernels::ref
