// Règles communes des noyaux pixel du backend TensorRT (étude 3) : blocs de 256 fils (16 × 16, ou
// 32 × 8 pour la mesure), au moins quatre blocs résidents par SM, soit 64 registres au plus par fil ;
// trois pour les noyaux lourds (warp, Area : 85 registres, la moitié des fils d'un SM d'Ada). ptxas
// refuse tout débordement de registres (--warn-on-spills en erreur, CMakeLists.txt).
// Réductions d'un bloc par mélange de registres, sans arbre en mémoire partagée.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "gpu/cuda/backend/ring_surface.h"

namespace ajp::gpu::cuda {

inline constexpr int kPixelThreads = 256;
inline constexpr int kPixelBlocksPerSm = 4;
inline constexpr int kHeavyBlocksPerSm = 3;
inline constexpr int kWarpSize = 32;

// Combine value sur tout le bloc (op associative, identity neutre) ; le résultat vaut dans le fil 0.
// Chaque fil du bloc doit l'appeler, y compris hors de l'image (avec identity).
template <typename Op>
__device__ float block_reduce(float value, float identity, Op op) {
  __shared__ float warps[kPixelThreads / kWarpSize];
  const int t = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
    value = op(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
  if (t % kWarpSize == 0) warps[t / kWarpSize] = value;
  __syncthreads();
  const int count = static_cast<int>(blockDim.x * blockDim.y) / kWarpSize;
  value = t < count ? warps[t] : identity;
  if (t < kWarpSize) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
      value = op(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
  }
  return value;
}

// kernels/spec/color.h unorm16 : clamp, × 65535, + 0,5, troncature.
__device__ inline unsigned short unorm16(float v) {
  v = v != v ? 0.0F : fminf(fmaxf(v, 0.0F), 1.0F);  // NaN donne 0
  return static_cast<unsigned short>(v * 65535.0F + 0.5F);
}

// kernels/spec/ring_pack.h, aux mêmes arrondis (sans contraction en FMA) : unorm10, ring_hash, ring_tpdf,
// unorm10_dithered.
__device__ inline float clamp_unit(float v) {
  return v != v ? 0.0F : fminf(fmaxf(v, 0.0F), 1.0F);  // NaN donne 0
}
__device__ inline std::uint32_t unorm10(float v) {
  return static_cast<std::uint32_t>(__fadd_rn(__fmul_rn(clamp_unit(v), 1023.0F), 0.5F));
}
__device__ inline std::uint32_t ring_hash(std::uint32_t x, std::uint32_t y, std::uint32_t salt) {
  std::uint32_t v = x * 0x9E3779B1U ^ y * 0x85EBCA77U ^ salt * 0xC2B2AE3DU;
  v = v * 747796405U + 2891336453U;
  v = ((v >> ((v >> 28U) + 4U)) ^ v) * 277803737U;
  return (v >> 22U) ^ v;
}
__device__ inline float ring_uniform(std::uint32_t x, std::uint32_t y, std::uint32_t salt) {
  return __fmul_rn(static_cast<float>(ring_hash(x, y, salt) >> 8U), 1.0F / 16777216.0F);
}
__device__ inline std::uint32_t unorm10_dithered(float v, int x, int y) {
  const auto ux = static_cast<std::uint32_t>(x);
  const auto uy = static_cast<std::uint32_t>(y);
  const float noise = __fadd_rn(__fadd_rn(ring_uniform(ux, uy, 5U), ring_uniform(ux, uy, 6U)), -1.0F);
  const float biased = __fadd_rn(__fadd_rn(__fmul_rn(clamp_unit(v), 1023.0F), 0.5F), noise);
  if (!(biased > 0.0F)) return 0U;
  return biased >= 1023.0F ? 1023U : static_cast<std::uint32_t>(biased);
}

// Un pixel RGB dans un emplacement de l'anneau, alpha opaque (étape 5.5b, seule fonction d'écriture de
// l'anneau). RGBA16 : unorm16 de chaque canal ; RGB10A2 : r | g << 10 | b << 20 | 3 << 30
// (kernels/spec/ring_pack.h), tramé si surface.dither.
__device__ inline void store_ring(const RingSurface& surface, int x, int y, float fr, float fg, float fb) {
  if (surface.packing == RingPacking::Rgb10a2) {
    const std::uint32_t pr = surface.dither ? unorm10_dithered(fr, x, y) : unorm10(fr);
    const std::uint32_t pg = surface.dither ? unorm10_dithered(fg, x, y) : unorm10(fg);
    const std::uint32_t pb = surface.dither ? unorm10_dithered(fb, x, y) : unorm10(fb);
    const std::uint32_t pixel = pr | (pg << 10U) | (pb << 20U) | (3U << 30U);
    surf2Dwrite(pixel, surface.object, x * static_cast<int>(sizeof(std::uint32_t)), y);
    return;
  }
  const ushort4 pixel = make_ushort4(unorm16(fr), unorm16(fg), unorm16(fb), 65535);
  surf2Dwrite(pixel, surface.object, x * static_cast<int>(sizeof(ushort4)), y);
}

// Le même depuis le FP16 d'un tenseur : RGBA16 au bit près de ce que write_rgba16 écrivait (ref::to_rgba16).
__device__ inline void store_ring(const RingSurface& surface, int x, int y, __half r, __half g, __half b) {
  store_ring(surface, x, y, __half2float(r), __half2float(g), __half2float(b));
}

struct Sum {
  __device__ float operator()(float a, float b) const { return a + b; }
};
struct Max {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

}  // namespace ajp::gpu::cuda
