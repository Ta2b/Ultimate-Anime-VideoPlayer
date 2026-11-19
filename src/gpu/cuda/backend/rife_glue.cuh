// La colle de RIFE fusionné au point (rife_fused_kernels.h) : les 28 canaux de X_k en un point pleine
// résolution, depuis les Y des blocs précédents relevés en mémoire partagée (warp_lib.cuh), l'entrée et les
// deux encodages. Partagée par la colle (fused_glue, rife_fused_kernels.cu) et par la première convolution
// du dernier bloc du RIFE natif, qui la calcule à la volée sans écrire X (rife_native_glue.cuh) : les mêmes
// fonctions, donc les mêmes arrondis.
#pragma once

#include <cstddef>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/rife_fused_kernels.h"
#include "gpu/cuda/backend/warp_lib.cuh"

namespace kirari::gpu::cuda::glue {

// Canaux relevés par pixel de Y : 16 pour un flot seul (les 4 canaux de flot de Z à ses quatre parités), 56
// pour le dernier bloc (flot, masque, attributs : 52 canaux, complétés au pas HWC8).
constexpr int kFlowStride = 16;
constexpr int kLastStride = 56;

// Z (DepthToSpace CRD de Y, facteur 2) lu dans un Y relevé : Z[c][zy][zx] = Y[4c + 2 (zy & 1) + (zx & 1)].
__device__ __forceinline__ float z_at(const warp::Staged& s, int c, int zy, int zx) {
  return __half2float(s.at(zy >> 1, zx >> 1)[4 * c + 2 * (zy & 1) + (zx & 1)]);
}

// Resize bilinéaire demi-pixel du canal c de Z au point pleine résolution, arrondi FP16 comme la sortie du
// Resize du graphe.
__device__ __forceinline__ float z_enlarge(const warp::Staged& s, int c, const warp::Axis& ax,
                                           const warp::Axis& ay) {
  return warp::rounded(warp::lerp2(z_at(s, c, ay.i0, ax.i0), z_at(s, c, ay.i0, ax.i1),
                                   z_at(s, c, ay.i1, ax.i0), z_at(s, c, ay.i1, ax.i1), ax, ay));
}

// Relève dans at la région du Y de f que lisent les points pleine résolution px × py : les 16 canaux de flot,
// ou les 52 du dernier bloc (last). Tous les fils du bloc participent ; l'appelant synchronise.
__device__ __forceinline__ warp::Staged stage_source(const FusedFlowSource& f, bool last, warp::Span px,
                                                     warp::Span py, __half* at) {
  const float inv = 1.0F / static_cast<float>(f.scale);
  const warp::Span zr = warp::coarse_span(py.lo, py.hi, inv, 2 * f.height);
  const warp::Span zc = warp::coarse_span(px.lo, px.hi, inv, 2 * f.width);
  const warp::Span rows{zr.lo >> 1, zr.hi >> 1};
  const warp::Span cols{zc.lo >> 1, zc.hi >> 1};
  const int stride = last ? kLastStride : kFlowStride;
  warp::stage_region(at, stride, static_cast<const __half*>(f.y), f.height, f.width, f.pitch, 0,
                     last ? 52 : 16, rows, cols);
  return {at, rows.lo, cols.lo, cols.hi - cols.lo + 1, stride};
}

// Le flot accumulé au point (px, py) (Add : sommes FP16 de chaque bloc, multiplié par l'échelle de son
// Resize) ; ax, ay : les axes du dernier bloc, que lisent le masque et les attributs.
template <int K>
__device__ __forceinline__ void flow_at(const FusedGlueParams& p, const warp::Staged (&src)[K], int px,
                                        int py, float (&fl)[4], warp::Axis& ax, warp::Axis& ay) {
#pragma unroll
  for (int j = 0; j < K; ++j) {
    const FusedFlowSource& f = p.source[j];
    const float inv = 1.0F / static_cast<float>(f.scale);
    ax = warp::coarse_axis(px, inv, 2 * f.width);
    ay = warp::coarse_axis(py, inv, 2 * f.height);
    const float scale = static_cast<float>(f.scale);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float r = z_enlarge(src[j], k, ax, ay) * scale;
      fl[k] = j == 0 ? r : warp::rounded(fl[k] + r);
    }
  }
}

// Canaux 0 à 13 : le warp des deux images (3 canaux chacune) et de leurs encodages (4 canaux chacun).
__device__ __forceinline__ void warped(const FusedGlueParams& p, int px, int py, const float (&fl)[4],
                                       float (&v)[28]) {
  const std::size_t plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  const auto* input = static_cast<const __half*>(p.input);
  const warp::Bounds whole{0, 0, p.width, p.height};
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const warp::Tap tap = warp::tap_at(px, py, fl[2 * n], fl[2 * n + 1], whole);
    const auto* encode = static_cast<const __half*>(n == 0 ? p.encode : p.encode_b);
#pragma unroll
    for (int ch = 0; ch < 7; ++ch) {
      const __half* from = ch < 3 ? input + static_cast<std::size_t>(3 * n + ch) * plane
                                  : encode + static_cast<std::size_t>(ch - 3) * plane;
      v[ch < 3 ? 3 * n + ch : 6 + 4 * n + ch - 3] = warp::rounded(warp::bilinear(from, tap, p.width));
    }
  }
}

// L'instant, à l'élément 0 du plan 6 de l'entrée (étape 5.2) : canal 14.
__device__ __forceinline__ float instant(const FusedGlueParams& p) {
  const std::size_t plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  return __half2float(static_cast<const __half*>(p.input)[6 * plane]);
}

// Les 28 canaux de X au point pleine résolution (px, py) d'une entrée réduite d'un facteur d : warps,
// instant, masque et attributs du dernier bloc (canaux 15 à 23), flot accumulé × (1/d).
template <int K>
__device__ __forceinline__ void values(const FusedGlueParams& p, const warp::Staged (&src)[K], int px, int py,
                                       int d, float (&v)[28]) {
  float fl[4];
  warp::Axis ax{};
  warp::Axis ay{};
  flow_at<K>(p, src, px, py, fl, ax, ay);
  warped(p, px, py, fl, v);
  v[14] = instant(p);
#pragma unroll
  for (int k = 0; k < 9; ++k) v[15 + k] = z_enlarge(src[K - 1], 4 + k, ax, ay);
  const float flow_scale = 1.0F / static_cast<float>(d);
#pragma unroll
  for (int k = 0; k < 4; ++k) v[24 + k] = warp::rounded(fl[k] * flow_scale);
}

}  // namespace kirari::gpu::cuda::glue
