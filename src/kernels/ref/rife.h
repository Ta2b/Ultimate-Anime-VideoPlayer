// Référence du contrat d'entrée de RIFE : remplissage mod-64 centré, conversion propre à RIFE,
// tenseur d'entrée à 11 plans et recadrage inverse (spec/rife.h). Repris de
// animejanai-inference-rtx@118c272 src/aji_trt.cpp:1294-1385 et 3700-3860.
// Tests : tests/unit/kernels/rife_test.cpp.
#pragma once

#include <array>
#include <vector>

#include "kernels/image.h"
#include "kernels/spec/rife.h"

namespace ajp::kernels::ref {

// L'image au centre de pw × ph, bords noir studio (rife_pad_luma, rife_pad_chroma).
Result<YuvImage> rife_pad(const YuvImage& frame);

// La fenêtre (left, top, width, height) d'une sortie remplie.
Result<RgbImage> rife_crop(const RgbImage& padded, const RifeGeometry& geometry);
Result<YuvImage> rife_crop(const YuvImage& padded, const RifeGeometry& geometry);

// Entrée : remplissage, puis conversion BT.709, chroma bilinéaire, siting gauche. pw × ph.
Result<RgbImage> rife_prepare(const YuvImage& frame, Range range);

// Sortie : RGB rempli → YUV rempli du format (même conversion), puis recadrage.
Result<YuvImage> rife_finish(const RgbImage& padded, Format format, Range range,
                             const RifeGeometry& geometry);

// Les quatre plans constants 7 à 10 : grille X, grille Y, 2 / (pw − 1), 2 / (ph − 1).
// Refus : dimensions qui ne sont pas des multiples de 64.
Result<std::array<PlaneF, 4>> rife_constant_planes(int padded_width, int padded_height);

// Les 11 plans de l'entrée, dans l'ordre de spec/rife.h ; a et b déjà préparés (pw × ph).
// Refus : tailles différentes ou non multiples de 64, t hors de [0, 1].
Result<std::vector<PlaneF>> rife_input(const RgbImage& a, const RgbImage& b, float t);

}  // namespace ajp::kernels::ref
