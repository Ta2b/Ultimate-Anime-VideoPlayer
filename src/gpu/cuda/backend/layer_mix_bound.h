#pragma once

namespace kirari::gpu::cuda::layer {

// Sources FP16 finies dans [-2, 2], masque du warp dans [0, 1].
inline constexpr unsigned kSourceBoundHalf = 0x4000;
inline constexpr unsigned kUnitHalf = 0x3c00;

// here est alors un FP16 dans [-2, 2] après le warp. P = sum·inv est le produit réel, target = RN(P)
// dans [0,1]. Pour mk = 1, l'ancienne formule calcule RN(here + RN(P - here)), avec ou sans contraction
// de P dans la soustraction. L'écart en float s'explique par l'arrondi du produit, de la soustraction
// (|P - here| < 4) et de l'addition (|résultat| < 2). Même avec l'arrondi du centre de l'intervalle,
// la somme des bornes est ≤ 2^-22 ; 2^-21 laisse de la marge. On refuse un intervalle qui touche zéro.
inline constexpr float kMixUncertainty = 0x1p-21F;

}  // namespace kirari::gpu::cuda::layer
