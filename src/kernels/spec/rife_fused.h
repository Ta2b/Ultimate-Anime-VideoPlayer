// RIFE « fusionné » : le réseau coupé en six petits moteurs TensorRT (l'encodeur et les blocs 0 à N-1,
// des convolutions à basse résolution) reliés par des noyaux CUDA maison pour tout ce qui travaille en
// pleine résolution (DepthToSpace, Resize, flot, grille, GridSample, Concat, mélange à masque). Ce fichier
// donne la forme du découpage et la disposition des tenseurs intermédiaires ; les noyaux et leur référence
// CPU (kernels/ref/rife_fused.h) la lisent.
//
// Le bloc k du réseau (RIFE 4.22 lite : 4 blocs, 4.25 : 5) reçoit l'entrée X_k à la résolution 1/d_k de
// l'image remplie (Resize d'entrée d'échelle 1/d_k) et sort Y_k, 52 canaux à la moitié de cette résolution
// (avant le DepthToSpace) ; la sortie du DepthToSpace, Z_k (13 canaux : flot 4, masque 1, attributs 8),
// retrouve la pleine résolution par un Resize de facteur d_k (aucun pour le dernier bloc, d = 1).
// Réduction d'entrée et agrandissement de sortie d'un bloc sont donc le même d_k (8, 8, 4, 2, 1 pour 4.25 ;
// 8, 4, 2, 1 pour 4.22 lite).
//
// Disposition des 28 canaux de X_k, k ≥ 1 (celle du graphe ONNX) :
//   0-2 image A warpée, 3-5 image B warpée, 6-9 encodage de A warpé, 10-13 encodage de B warpé, 14 instant,
//   15 masque, 16-23 attributs (ces trois-là venus du bloc k-1), 24-27 flot accumulé × (1/d_k).
// X_0 a 15 canaux : images A et B, encodages de A et de B, instant, sans warp.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ajp::kernels {

inline constexpr int kFusedMaxBlocks = 5;
inline constexpr int kFusedInputPlanes = 11;    // l'entrée de RIFE v1 (kernels/spec/rife.h)
inline constexpr int kFusedEncodePlanes = 8;    // encodage de A (4 plans) puis de B (4 plans)
inline constexpr int kFusedFirstChannels = 15;  // canaux de X_0
inline constexpr int kFusedNextChannels = 28;   // canaux de X_k, k ≥ 1
inline constexpr int kFusedOutChannels = 52;    // canaux de Y_k, avant DepthToSpace
inline constexpr int kFusedFlowChannels = 4;
inline constexpr int kFusedAttributeChannels = 9;  // masque puis 8 attributs

// Le découpage : nombre de blocs et réduction d'entrée de chacun.
struct RifeFusedPlan {
  int blocks = 0;
  std::array<int, kFusedMaxBlocks> reduce{};
  friend bool operator==(const RifeFusedPlan&, const RifeFusedPlan&) = default;
};

// Un découpage que les noyaux savent faire : 2 à 5 blocs, réduction 8, 4, 2 ou 1 (puissances de deux), la
// première au moins 2 (X_0 est une moyenne 2 × 2), la dernière 1 (Z en pleine résolution).
constexpr bool rife_fused_plan_valid(const RifeFusedPlan& plan) {
  if (plan.blocks < 2 || plan.blocks > kFusedMaxBlocks) return false;
  for (int k = 0; k < plan.blocks; ++k) {
    const int d = plan.reduce[static_cast<std::size_t>(k)];
    if (d != 1 && d != 2 && d != 4 && d != 8) return false;
  }
  return plan.reduce[0] >= 2 && plan.reduce[static_cast<std::size_t>(plan.blocks - 1)] == 1;
}

// Une taille remplie que le découpage prend : chaque Y_k est entier (hauteur et largeur multiples de 2 d_k),
// et la plus petite entrée a au moins deux lignes et deux colonnes.
constexpr bool rife_fused_size_valid(const RifeFusedPlan& plan, int height, int width) {
  if (!rife_fused_plan_valid(plan) || height <= 0 || width <= 0) return false;
  for (int k = 0; k < plan.blocks; ++k) {
    const int d = 2 * plan.reduce[static_cast<std::size_t>(k)];
    if (height % d != 0 || width % d != 0) return false;
  }
  return true;
}

constexpr int rife_fused_block_channels(int k) {
  return k == 0 ? kFusedFirstChannels : kFusedNextChannels;
}

// Forme d'un tenseur C × H × W (le lot est de 1).
struct FusedDims {
  int channels = 0;
  int height = 0;
  int width = 0;
  constexpr std::int64_t elements() const { return static_cast<std::int64_t>(channels) * height * width; }
};

constexpr FusedDims rife_fused_x(const RifeFusedPlan& plan, int k, int height, int width) {
  const int d = plan.reduce[static_cast<std::size_t>(k)];
  return {rife_fused_block_channels(k), height / d, width / d};
}

constexpr FusedDims rife_fused_y(const RifeFusedPlan& plan, int k, int height, int width) {
  const int d = 2 * plan.reduce[static_cast<std::size_t>(k)];
  return {kFusedOutChannels, height / d, width / d};
}

// Les tampons intermédiaires d'une exécution, rangés les uns après les autres dans une zone (en octets, FP16)
// : l'encodage de la paire, puis X_k et Y_k de chaque bloc. Chaque début est un multiple de 256 octets.
struct RifeFusedLayout {
  std::size_t encode = 0;
  std::array<std::size_t, kFusedMaxBlocks> x{};
  std::array<std::size_t, kFusedMaxBlocks> y{};
  std::size_t bytes = 0;
};

constexpr std::size_t fused_align(std::size_t bytes) {
  return (bytes + 255U) / 256U * 256U;
}

constexpr RifeFusedLayout rife_fused_layout(const RifeFusedPlan& plan, int height, int width) {
  RifeFusedLayout layout;
  std::size_t at = 0;
  const auto take = [&at](std::int64_t elements) {
    const std::size_t begin = at;
    at += fused_align(static_cast<std::size_t>(elements) * 2U);
    return begin;
  };
  layout.encode = take(std::int64_t{kFusedEncodePlanes} * height * width);
  for (int k = 0; k < plan.blocks; ++k) {
    const auto i = static_cast<std::size_t>(k);
    layout.x[i] = take(rife_fused_x(plan, k, height, width).elements());
    layout.y[i] = take(rife_fused_y(plan, k, height, width).elements());
  }
  layout.bytes = at;
  return layout;
}

}  // namespace ajp::kernels
