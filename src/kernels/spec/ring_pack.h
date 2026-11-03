// Empaquetage d'un pixel de l'anneau (étape 5.5b) : RGBA16 UNORM (color.h, unorm16) ou RGB10A2 UNORM,
// r | g << 10 | b << 20 | 3 << 30 (VK_FORMAT_A2B10G10R10_UNORM_PACK32 : rouge dans les bits de poids faible).
// Le code 10 bits suit la règle de unorm16 : clamp(v, 0, 1) × 1023 + 0,5, tronqué ; NaN donne 0.
// Tramage TPDF en option : un bruit triangulaire dans ]−1, 1[ code, fixe par pixel (même hachage d'une image
// à l'autre, un aplat immobile ne scintille pas), ajouté avant la troncature.
// Les noyaux CUDA (pixel_kernels.cuh, store_ring) en sont la copie, aux mêmes arrondis.
#pragma once

#include <cstdint>

#include "kernels/spec/color.h"

namespace ajp::kernels {

inline constexpr float kUnorm10Max = 1023.0F;
inline constexpr std::uint32_t kOpaqueAlpha2 = 3U;

constexpr std::uint32_t unorm10(float v) {
  return static_cast<std::uint32_t>(clamp_unit(v) * kUnorm10Max + 0.5F);
}

// Hachage entier (PCG) du pixel, celui du débandage (deband_kernels.cuh) avec d'autres sels.
constexpr std::uint32_t ring_hash(std::uint32_t x, std::uint32_t y, std::uint32_t salt) {
  std::uint32_t v = x * 0x9E3779B1U ^ y * 0x85EBCA77U ^ salt * 0xC2B2AE3DU;
  v = v * 747796405U + 2891336453U;
  v = ((v >> ((v >> 28U) + 4U)) ^ v) * 277803737U;
  return (v >> 22U) ^ v;
}

constexpr float ring_uniform(std::uint32_t x, std::uint32_t y, std::uint32_t salt) {
  return static_cast<float>(ring_hash(x, y, salt) >> 8U) * (1.0F / 16777216.0F);  // [0, 1)
}

// Bruit TPDF du pixel (x, y), en codes : somme de deux uniformes moins 1, dans ]−1, 1[.
constexpr float ring_tpdf(std::uint32_t x, std::uint32_t y) {
  return ring_uniform(x, y, 5U) + ring_uniform(x, y, 6U) - 1.0F;
}

// Code 10 bits tramé : clamp(v, 0, 1) × 1023 + 0,5 + bruit, ramené dans [0, 1023], tronqué.
constexpr std::uint32_t unorm10_dithered(float v, float noise) {
  const float biased = clamp_unit(v) * kUnorm10Max + 0.5F + noise;
  if (!(biased > 0.0F)) return 0U;
  return biased >= kUnorm10Max ? 1023U : static_cast<std::uint32_t>(biased);
}

constexpr std::uint32_t pack_rgb10a2(std::uint32_t r, std::uint32_t g, std::uint32_t b) {
  return r | (g << 10U) | (b << 20U) | (kOpaqueAlpha2 << 30U);
}

}  // namespace ajp::kernels
