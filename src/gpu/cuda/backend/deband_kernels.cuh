// Le tirage du débandage (deband_kernels.cu), partagé avec l'ingestion qui débande à même taille
// (slice_kernels.cu, étape 5.4) : mêmes points en croix, même règle, donc même résultat.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "gpu/cuda/backend/deband_kernels.h"

namespace ajp::gpu::cuda {

// Bruit uniforme dans [0, 1) tiré d'un hachage entier (PCG) du pixel : fixe d'une image à l'autre, pour
// qu'un aplat immobile ne scintille pas.
__device__ __forceinline__ float deband_uniform(std::uint32_t x, std::uint32_t y, std::uint32_t salt) {
  std::uint32_t v = x * 0x9E3779B1U ^ y * 0x85EBCA77U ^ salt * 0xC2B2AE3DU;
  v = v * 747796405U + 2891336453U;
  v = ((v >> ((v >> 28U) + 4U)) ^ v) * 277803737U;
  v = (v >> 22U) ^ v;
  return static_cast<float>(v >> 8U) * (1.0F / 16777216.0F);
}

// Le décalage (dx, dy) du pixel (x, y) : les quatre points sont (±dx, ±dy) et (∓dy, ±dx).
__device__ __forceinline__ int2 deband_offset(int x, int y) {
  const auto ux = static_cast<std::uint32_t>(x);
  const auto uy = static_cast<std::uint32_t>(y);
  const float distance = kDebandRadius * deband_uniform(ux, uy, 1U);
  float s = 0.0F;
  float c = 1.0F;
  sincosf(6.2831853F * deband_uniform(ux, uy, 2U), &s, &c);
  return make_int2(__float2int_rn(distance * c), __float2int_rn(distance * s));
}

// La règle : la moyenne des quatre points, sauf si elle s'écarte de la valeur de plus du seuil.
__device__ __forceinline__ float deband_rule(float value, float a, float b, float d, float e) {
  const float average = 0.25F * (a + b + d + e);
  return fabsf(average - value) > kDebandThreshold ? value : average;
}

}  // namespace ajp::gpu::cuda
