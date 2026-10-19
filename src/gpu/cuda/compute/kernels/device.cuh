// Outils des noyaux CUDA : lectures et écritures typées, miroir des bords, quantification.
// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:87-97 (mirr, quant) et
// src/kernels.cu:963-989 (sample_plane, sample_window), réécrits sur kernels/spec.h.
//
// Les .cu se compilent sans contraction FMA (--fmad=false) : chaque a × b + c s'arrondit en
// deux temps, dans l'ordre écrit de la spécification, comme la référence CPU. Les sorties
// sont donc exactes là où la référence l'est.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/compute/kernels/views.h"

namespace ajp::gpu::cuda::dev {

inline constexpr int kBlockX = 32;
inline constexpr int kBlockY = 8;

inline dim3 block2d() {
  return {kBlockX, kBlockY, 1};
}

inline dim3 grid2d(int width, int height, int depth = 1) {
  return {static_cast<unsigned>((width + kBlockX - 1) / kBlockX),
          static_cast<unsigned>((height + kBlockY - 1) / kBlockY), static_cast<unsigned>(depth)};
}

// Miroir sans répétition du bord, puis clamp (mirror_index de spec/resample.h).
__device__ __forceinline__ int mirror(int i, int n) {
  i = i < 0 ? -i - 1 : i;
  i = i >= n ? 2 * n - 1 - i : i;
  return min(max(i, 0), n - 1);
}

__device__ __forceinline__ std::size_t at(int x, int y, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

template <class T>
__device__ __forceinline__ const T* row_of(Plane p, int y) {
  return reinterpret_cast<const T*>(p.ptr + static_cast<std::size_t>(y) * p.pitch);
}

template <class T>
__device__ __forceinline__ T* mutable_row_of(Plane p, int y) {
  return reinterpret_cast<T*>(p.ptr + static_cast<std::size_t>(y) * p.pitch);
}

// Éléments de tenseur : float tel quel, fp16 arrondi au plus proche, égalités au pair.
__device__ __forceinline__ float to_float(float v) {
  return v;
}
__device__ __forceinline__ float to_float(__half v) {
  return __half2float(v);
}

template <class T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half_rn(v);
}

// quantize() de spec/color.h : x = raw / qdiv ; qmax au-delà, arrondi pair dans ]0, qmax[,
// 0 sinon (NaN compris) ; puis × qdiv.
__device__ __forceinline__ unsigned quantize(float raw, float qdiv, float qmax) {
  const float x = raw / qdiv;
  float code = 0.0F;
  if (x >= qmax) {
    code = qmax;
  } else if (x > 0.0F) {
    code = rintf(x);
  }
  return static_cast<unsigned>(code * qdiv);
}

// unorm16() de spec/color.h : tronque(clamp(v, 0, 1) × 65535 + 0,5), NaN donne 0.
__device__ __forceinline__ unsigned short unorm16(float v) {
  const float clamped = !(v > 0.0F) ? 0.0F : (v < 1.0F ? v : 1.0F);
  return static_cast<unsigned short>(clamped * 65535.0F + 0.5F);
}

// Échantillon bilinéaire d'un plan width × height du tenseur, ou d'une fenêtre d'un plan de
// pas stride (spec/warp.h) : coordonnées bornées, NaN ramené à 0 par fmaxf.
template <class T>
__device__ __forceinline__ float bilinear(const T* p, int stride, int x0w, int y0w, int width, int height,
                                          float x, float y) {
  x = fminf(fmaxf(x, 0.0F), static_cast<float>(width - 1));
  y = fminf(fmaxf(y, 0.0F), static_cast<float>(height - 1));
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = min(x0 + 1, width - 1);
  const int y1 = min(y0 + 1, height - 1);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const T* r0 = p + at(x0w, y0w + y0, stride);
  const T* r1 = p + at(x0w, y0w + y1, stride);
  const float top = to_float(r0[x0]) * (1.0F - fx) + to_float(r0[x1]) * fx;
  const float bottom = to_float(r1[x0]) * (1.0F - fx) + to_float(r1[x1]) * fx;
  return top * (1.0F - fy) + bottom * fy;
}

}  // namespace ajp::gpu::cuda::dev
