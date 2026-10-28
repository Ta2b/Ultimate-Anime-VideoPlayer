// DistilDRBA « fusionné » : le réseau (v2 lite, trois blocs aux échelles 16, 8 et 4) coupé en quatre petits
// moteurs TensorRT (l'encodeur, en lot de trois images, et les blocs 0 à 2, des convolutions à basse
// résolution) reliés par des noyaux CUDA maison pour tout ce qui n'est pas convolution (réductions d'images,
// choix du côté de la paire, Resize, DepthToSpace, flot, grille, GridSample, Concat, mélange à masque). Ce
// fichier donne la forme du découpage et la disposition des tenseurs intermédiaires ; les noyaux la lisent.
//
// Échelles (reduce) : s0, s1, s2 = 4 t, 2 t, t pour t = 4 (graphe normal : 16, 8, 4) ou t = 8 (flot à
// l'échelle 0,5, 4K : 32, 16, 8). L'encodeur travaille sur les images réduites de t, comme le bloc 2.
//   - X_k : entrée du bloc k, à la résolution 1/s_k de l'image remplie ;
//   - Y_k : sortie du bloc k, 52 canaux à la moitié de cette résolution (avant DepthToSpace) ; Z_k (13 canaux
//   :
//     flot 4, masque 1, attributs 8) en est le DepthToSpace.
// Le flot, le masque et les attributs du bloc k sont doublés pour entrer dans le bloc k + 1 ; ceux du bloc 2
// sont agrandis ×t jusqu'à la pleine résolution.
//
// Disposition des canaux de X_1 et X_2 (109, celle du graphe ONNX) :
//   0-8 images 0, 1, 2 réduites, 9-56 encodages f0, f1, f2 réduits (16 canaux chacun), 57-59 image 1 warpée,
//   60-62 image du côté choisi warpée, 63-78 f1 warpé, 79-94 f du côté choisi warpé, 95 temps, 96 masque,
//   97-104 attributs, 105-108 flot.
// X_0 (58 canaux) : 0-8 images réduites, 9-56 encodages réduits, 57 temps.
//
// L'entrée de l'encodeur (lot 3 × 3 plans, les images réduites de t) et sa sortie (3 × 16 plans) sont
// adressées DANS X_2, aux plans 0-8 et 9-56 : ces 57 canaux n'ont pas à être copiés.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ajp::kernels {

inline constexpr int kDrbaFusedBlocks = 3;
inline constexpr int kDrbaFusedInputPlanes = 7;  // pair01 et pair12 (kernels/spec/distil_drba.h)
inline constexpr int kDrbaFusedFirstChannels = 58;
inline constexpr int kDrbaFusedNextChannels = 109;
inline constexpr int kDrbaFusedOutChannels = 52;  // canaux de Y_k, avant DepthToSpace
inline constexpr int kDrbaFusedEncodeChannels = 16;
inline constexpr int kDrbaFusedImagePlanes = 9;    // trois images RVB réduites
inline constexpr int kDrbaFusedEncodePlanes = 48;  // trois encodages de 16 canaux
inline constexpr int kDrbaFusedTimeChannel = 95;   // X_1 et X_2

// Le découpage : t, le plus petit des trois facteurs de réduction (4 ou 8).
struct DrbaFusedPlan {
  int tail = 4;
  friend bool operator==(const DrbaFusedPlan&, const DrbaFusedPlan&) = default;
};

constexpr bool drba_fused_plan_valid(const DrbaFusedPlan& plan) {
  return plan.tail == 4 || plan.tail == 8;
}

// Réduction d'entrée du bloc k : 4 t, 2 t, t.
constexpr int drba_fused_reduce(const DrbaFusedPlan& plan, int k) {
  return plan.tail << (2 - k);
}

// Une taille remplie que le découpage prend : chaque Y_k est entier, donc multiple de 2 s_0 (32 ou 64).
constexpr bool drba_fused_size_valid(const DrbaFusedPlan& plan, int height, int width) {
  if (!drba_fused_plan_valid(plan) || height <= 0 || width <= 0) return false;
  const int d = 2 * drba_fused_reduce(plan, 0);
  return height % d == 0 && width % d == 0;
}

// Forme d'un tenseur C × H × W (le lot est de 1).
struct DrbaFusedDims {
  int channels = 0;
  int height = 0;
  int width = 0;
  constexpr std::int64_t elements() const { return static_cast<std::int64_t>(channels) * height * width; }
};

constexpr DrbaFusedDims drba_fused_x(const DrbaFusedPlan& plan, int k, int height, int width) {
  const int d = drba_fused_reduce(plan, k);
  return {k == 0 ? kDrbaFusedFirstChannels : kDrbaFusedNextChannels, height / d, width / d};
}

constexpr DrbaFusedDims drba_fused_y(const DrbaFusedPlan& plan, int k, int height, int width) {
  const int d = 2 * drba_fused_reduce(plan, k);
  return {kDrbaFusedOutChannels, height / d, width / d};
}

// Les tampons intermédiaires d'une exécution, rangés les uns après les autres dans une zone (en octets, FP16)
// : X_k puis Y_k de chaque bloc. Chaque début est un multiple de 256 octets.
struct DrbaFusedLayout {
  std::array<std::size_t, kDrbaFusedBlocks> x{};
  std::array<std::size_t, kDrbaFusedBlocks> y{};
  std::size_t bytes = 0;
};

constexpr std::size_t drba_fused_align(std::size_t bytes) {
  return (bytes + 255U) / 256U * 256U;
}

constexpr DrbaFusedLayout drba_fused_layout(const DrbaFusedPlan& plan, int height, int width) {
  DrbaFusedLayout layout;
  std::size_t at = 0;
  const auto take = [&at](std::int64_t elements) {
    const std::size_t begin = at;
    at += drba_fused_align(static_cast<std::size_t>(elements) * 2U);
    return begin;
  };
  for (int k = 0; k < kDrbaFusedBlocks; ++k) {
    const auto i = static_cast<std::size_t>(k);
    layout.x[i] = take(drba_fused_x(plan, k, height, width).elements());
    layout.y[i] = take(drba_fused_y(plan, k, height, width).elements());
  }
  layout.bytes = at;
  return layout;
}

}  // namespace ajp::kernels
