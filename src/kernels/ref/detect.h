// Référence des décisions temporelles : score de coupe de scène et pire bloc des dessins tenus
// (spec/detect.h). Repris de animejanai-inference-rtx@118c272 src/kernels.cu:698-743
// (k_scd_diff) et 923-959 (k_block_diff). Tests : tests/unit/kernels/detect_test.cpp.
#pragma once

#include <cstdint>

#include "kernels/image.h"
#include "kernels/spec/detect.h"

namespace ajp::kernels::ref {

// Σ |a − b| × scene_norm(format) sur deux plans de luma de même taille. Chaque terme se calcule
// en float comme sur GPU ; la somme est en double, sans dépendre de l'ordre.
Result<double> scene_diff_sum(const PlaneU16& a, const PlaneU16& b, Format format);

// Score de RIFE : la somme sur la luma source (width × height de la géométrie), divisée par
// l'aire remplie pw × ph. À comparer avec is_scene_cut.
Result<double> scene_score(const PlaneU16& a, const PlaneU16& b, Format format, const RifeGeometry& geometry);

// Pire écart moyen d'un bloc 32 × 32, en 1/256 de niveau 8 bits. À comparer avec is_held.
Result<std::uint32_t> block_diff_max(const PlaneU16& a, const PlaneU16& b, Format format);

}  // namespace ajp::kernels::ref
