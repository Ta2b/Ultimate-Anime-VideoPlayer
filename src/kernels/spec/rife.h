// Contrat d'entrée de RIFE : remplissage mod-64 centré, valeurs du bord, couches constantes,
// et recadrage inverse de la sortie.
//
// Repris de animejanai-inference-rtx@118c272 :
//   src/aji_trt.cpp:1294-1385 (setup_rife : taille remplie, bords centrés pairs, 11 couches),
//   src/aji_trt.cpp:3700-3830 (bords noir studio, BT.709, chroma bilinéaire, siting gauche),
//   src/kernels.cu:629-644 (k_rife_consts : grille et multiplicateurs).
// Tests : tests/unit/kernels/rife_test.cpp.
#pragma once

#include <cstdint>

#include "kernels/spec/resample.h"

namespace ajp::kernels {

// RIFE travaille sur des dimensions multiples de 64.
inline constexpr int kRifeAlign = 64;

// La fenêtre de l'image dans les tenseurs remplis.
struct RifeGeometry {
  int width = 0;  // image source
  int height = 0;
  int padded_width = 0;  // arrondi au multiple de 64 supérieur
  int padded_height = 0;
  int left = 0;  // bord gauche : moitié du remplissage, arrondie au pair inférieur
  int top = 0;
};

// pw = ceil(w / 64) × 64 ; left = ((pw − w) / 2) & ~1 : centré comme AddBorders, mais pair
// pour garder la chroma 4:2:0 alignée. Le reste du remplissage va à droite et en bas.
constexpr RifeGeometry rife_geometry(int width, int height) {
  const int pw = (width + kRifeAlign - 1) / kRifeAlign * kRifeAlign;
  const int ph = (height + kRifeAlign - 1) / kRifeAlign * kRifeAlign;
  return {.width = width,
          .height = height,
          .padded_width = pw,
          .padded_height = ph,
          .left = ((pw - width) / 2) & ~1,
          .top = ((ph - height) / 2) & ~1};
}

// Les bords sont noir studio en YUV, quelle que soit la plage : Y = 16 et Cb = Cr = 128
// niveaux 8 bits (std.AddBorders). En plage pleine, ce noir vaut donc 16/255 après la matrice.
constexpr std::uint16_t rife_pad_luma(Format f) {
  return static_cast<std::uint16_t>(16U * level_unit(f));
}

constexpr std::uint16_t rife_pad_chroma(Format f) {
  return static_cast<std::uint16_t>(128U * level_unit(f));
}

// Conversion propre à RIFE, celle de rife_cuda.py : matrice BT.709 quelle que soit la source,
// chroma 4:2:0 bilinéaire en siting gauche, à l'entrée comme à la sortie.
inline constexpr Matrix kRifeMatrix = Matrix::Bt709;
inline constexpr Filter kRifeChromaFilter = Filter::Bilinear;
inline constexpr Siting kRifeSiting = Siting::Left;

// Entrée v1 de RIFE (vsmlrt get_rife_input) : 11 plans pw × ph, en fp16.
//   0-2 : image a en RGB, 3-5 : image b, 6 : instant t,
//   7 : grille X = 2x / (pw − 1) − 1, 8 : grille Y = 2y / (ph − 1) − 1,
//   9 : 2 / (pw − 1), 10 : 2 / (ph − 1).
inline constexpr int kRifeInputChannels = 11;
// Entrée v2 (dossier rife_v2 de vs-mlrt) : les plans 0 à 6 seuls, la grille est calculée dans le
// graphe.
inline constexpr int kRifeV2InputChannels = 7;
constexpr bool is_rife_input_planes(std::int64_t planes) {
  return planes == kRifeInputChannels || planes == kRifeV2InputChannels;
}
inline constexpr int kRifeChannelA = 0;
inline constexpr int kRifeChannelB = 3;
inline constexpr int kRifeChannelTime = 6;
inline constexpr int kRifeChannelMeshX = 7;
inline constexpr int kRifeChannelMeshY = 8;
inline constexpr int kRifeChannelStepX = 9;
inline constexpr int kRifeChannelStepY = 10;

constexpr float rife_mesh(int i, int n) {
  return 2.0F * static_cast<float>(i) / static_cast<float>(n - 1) - 1.0F;
}

constexpr float rife_mesh_step(int n) {
  return 2.0F / static_cast<float>(n - 1);
}

// Sortie : RGB rempli → YUV rempli (même conversion qu'à l'entrée), puis recadrage de la
// fenêtre (left, top, width, height).

}  // namespace ajp::kernels
