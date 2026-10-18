// Contrat des décisions temporelles : coupe de scène avant RIFE, et dessins tenus.
//
// Repris de animejanai-inference-rtx@118c272 :
//   src/kernels.cu:698-743 (k_scd_diff : métrique de misc.SCDetect),
//   src/aji_trt.cpp:3729-3829 (somme sur la luma non remplie, divisée par l'aire remplie),
//   src/kernels.cu:923-959 (k_block_diff : pire bloc 32 × 32),
//   src/aji_trt.cpp:3530-3531 (comparaison entière au seuil).
// Le seuil de dessin tenu vient du plan : 0,70 niveau 8 bits ; 0,40 (défaut du fork) tombe dans
// le bruit de compression. Tests : tests/unit/kernels/detect_test.cpp.
#pragma once

#include <cstdint>

#include "kernels/spec/rife.h"

namespace ajp::kernels {

// Coupe de scène. Pour chaque pixel de luma de la source (non remplie) :
//   d = |a − b| × (1 / crête), a et b en float, crête = max_code du format (255 en 8 bits,
//   65472 en P010, 65535 en 16 bits, 1023 en 10 bits poids faible) ;
//   score = Σ d / (pw × ph) : l'aire remplie de RIFE, que la mesure à 720p a fixée ;
//   coupe si score > 0,150, strictement.
inline constexpr double kSceneCutThreshold = 0.150;

constexpr float scene_norm(Format f) {
  return 1.0F / static_cast<float>(max_code(f));
}

constexpr bool is_scene_cut(double score) {
  return score > kSceneCutThreshold;
}

// Dessins tenus. Blocs 32 × 32 de la luma depuis (0, 0), tronqués aux bords :
//   s = Σ |a − b| en entiers, n = pixels du bloc ;
//   écart = tronque(s × (256 / unité) / n), en 1/256 de niveau 8 bits (unité = level_unit) ;
//   pire = max des écarts ; tenu si pire × 100 ≤ seuil × 256, seuil en centièmes de niveau.
inline constexpr int kHeldBlock = 32;
inline constexpr std::uint32_t kHeldThreshold = 70;   // 0,70 niveau 8 bits
inline constexpr std::uint32_t kHeldNoiseFloor = 40;  // 0,40 : dans le bruit, jamais par défaut

constexpr std::uint32_t held_scale(Format f) {
  return 256U / level_unit(f);
}

constexpr bool is_held(std::uint32_t worst_block, std::uint32_t threshold = kHeldThreshold) {
  return std::uint64_t{worst_block} * 100U <= std::uint64_t{threshold} * 256U;
}

}  // namespace ajp::kernels
