// Lecture et mélange des calques communs au noyau séparé et au banc du warp fusionné.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/layer_kernels.h"
#include "gpu/cuda/backend/warp_lib.cuh"

namespace kirari::gpu::cuda::layer {

constexpr float kLumaR = 0.2126F;
constexpr float kLumaG = 0.7152F;
constexpr float kLumaB = 0.0722F;

__device__ __forceinline__ float grid_bit(const LayerGridArgs& g, int col, int row) {
  const int i = row * g.cols + col;
  return ((g.bits[i >> 5] >> (i & 31)) & 1U) != 0U ? 1.0F : 0.0F;
}

// Étape 1 de la spec : bits de la grille en bilinéaire, puis 2 g − 1 borné.
__device__ __forceinline__ float grid_mask(const LayerGridArgs& g, float x, float y) {
  const float u = fminf(fmaxf((x + 0.5F) / g.tile_w - 0.5F, 0.0F), static_cast<float>(g.cols - 1));
  const float v = fminf(fmaxf((y + 0.5F) / g.tile_h - 0.5F, 0.0F), static_cast<float>(g.rows - 1));
  const int c0 = static_cast<int>(u);
  const int r0 = static_cast<int>(v);
  const int c1 = min(c0 + 1, g.cols - 1);
  const int r1 = min(r0 + 1, g.rows - 1);
  const float fu = u - static_cast<float>(c0);
  const float fv = v - static_cast<float>(r0);
  const float top = grid_bit(g, c0, r0) * (1.0F - fu) + grid_bit(g, c1, r0) * fu;
  const float bottom = grid_bit(g, c0, r1) * (1.0F - fu) + grid_bit(g, c1, r1) * fu;
  return fminf(fmaxf(2.0F * (top * (1.0F - fv) + bottom * fv) - 1.0F, 0.0F), 1.0F);
}

// Après lecture des sources, la garde des paires est indépendante du pixel de RIFE.
struct Pixel {
  float sum[3] = {};
  float first[3] = {};
  float d = 0.0F;
  float total = 0.0F;
  float m = 0.0F;
  float source_mask = 1.0F;
};

template <bool PairOnly = false>
__device__ __forceinline__ Pixel sample(const LayerArgs& p, int x, int y) {
  const auto fx = static_cast<float>(x);
  const auto fy = static_cast<float>(y);
  const float m = grid_mask(p.grid, fx, fy);
  if (m <= 0.0F) return {};
  const std::int64_t plane = static_cast<std::int64_t>(p.width) * p.height;
  const warp::Bounds bounds{0, 0, p.width, p.height};
  const float wmax = static_cast<float>(p.width - 1);
  const float hmax = static_cast<float>(p.height - 1);
  float value[2][3] = {};
  bool inside[2] = {false, false};
  float sum[3] = {0.0F, 0.0F, 0.0F};
  float total = 0.0F;
  float source_mask = 1.0F;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (!PairOnly && i >= p.count) break;
    const float* s = p.shift[i];
    const float dx = s[0] + s[1] * fx + s[2] * fy;
    const float dy = s[3] + s[4] * fx + s[5] * fy;
    const float qx = fx + dx;
    const float qy = fy + dy;
    if (!(qx >= 0.0F && qx <= wmax && qy >= 0.0F && qy <= hmax)) continue;
    inside[i] = true;
    float w = p.weight[i];
    if (!PairOnly && i == 0 && p.source_grid.cols > 0) {
      source_mask = grid_mask(p.source_grid, qx, qy);
      w *= source_mask;
    }
    const auto* src = static_cast<const __half*>(i == 0 ? p.a : p.b);
    const warp::Tap tap = warp::tap_at(x, y, dx, dy, bounds);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const __half* channel = src + c * plane;
      value[i][c] = p.cubic != 0 ? warp::catmull_rom(channel, tap, p.width, bounds)
                                 : warp::bilinear(channel, tap, p.width);
    }
#pragma unroll
    for (int c = 0; c < 3; ++c) sum[c] += w * value[i][c];
    total += w;
  }
  if (total <= 0.0F) return {};
  float d = 0.0F;
  if ((PairOnly || p.count == 2) && inside[0] && inside[1]) {
    d = fabsf(kLumaR * (value[0][0] - value[1][0]) + kLumaG * (value[0][1] - value[1][1]) +
              kLumaB * (value[0][2] - value[1][2]));
  }
  return {{sum[0], sum[1], sum[2]}, {value[0][0], value[0][1], value[0][2]}, d, total, m, source_mask};
}

__device__ __forceinline__ float mix_weight(const LayerArgs& p, const Pixel& v, float d, float a) {
  const float k = 1.0F - fminf(fmaxf((d - p.guard_low) / (p.guard_high - p.guard_low), 0.0F), 1.0F);
  return v.m * k * a;
}

__device__ __forceinline__ __half finish(float here, float mk, float sum, float inv) {
  // Le SASS de compose_layers sur sm_89 contracte les deux opérations. Le calcul de target dans le
  // raccourci pourrait sinon faire réutiliser le produit arrondi sur le chemin complet.
  const float v = __fmaf_rn(mk, __fmaf_rn(sum, inv, -here), here);
  return __float2half_rn(fminf(fmaxf(v, 0.0F), 1.0F));
}

}  // namespace kirari::gpu::cuda::layer
