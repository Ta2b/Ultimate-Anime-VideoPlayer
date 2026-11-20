// La composition par calques (layer_kernels.h) : un fil par pixel de sortie. Les fils dont le masque des
// tuiles est nul sortent avant toute lecture des sources : sur un plan de dialogue, seul le décor coûte.
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/layer_kernels.h"
#include "gpu/cuda/backend/warp_lib.cuh"

namespace kirari::gpu::cuda {
namespace {

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

__global__ void __launch_bounds__(256) compose_layers(LayerArgs p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const auto fx = static_cast<float>(x);
  const auto fy = static_cast<float>(y);
  const float m = grid_mask(p.grid, fx, fy);
  if (m <= 0.0F) return;
  const std::int64_t plane = static_cast<std::int64_t>(p.width) * p.height;
  const std::int64_t at = static_cast<std::int64_t>(y) * p.width + x;
  const warp::Bounds bounds{0, 0, p.width, p.height};
  const float wmax = static_cast<float>(p.width - 1);
  const float hmax = static_cast<float>(p.height - 1);
  float value[2][3] = {};
  bool inside[2] = {false, false};
  float sum[3] = {0.0F, 0.0F, 0.0F};
  float total = 0.0F;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (i >= p.count) break;
    const float* s = p.shift[i];
    const float dx = s[0] + s[1] * fx + s[2] * fy;
    const float dy = s[3] + s[4] * fx + s[5] * fy;
    const float qx = fx + dx;
    const float qy = fy + dy;
    if (!(qx >= 0.0F && qx <= wmax && qy >= 0.0F && qy <= hmax)) continue;
    inside[i] = true;
    float w = p.weight[i];
    if (i == 0 && p.source_grid.cols > 0) w *= grid_mask(p.source_grid, qx, qy);
    const auto* src = static_cast<const __half*>(i == 0 ? p.a : p.b);
    const warp::Tap tap = warp::tap_at(x, y, dx, dy, bounds);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const __half* channel = src + c * plane;
      const float v = p.cubic != 0 ? warp::catmull_rom(channel, tap, p.width, bounds)
                                   : warp::bilinear(channel, tap, p.width);
      value[i][c] = v;
      sum[c] += w * v;
    }
    total += w;
  }
  if (total <= 0.0F) return;
  auto* out = static_cast<__half*>(p.out);
  float here[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) here[c] = __half2float(out[c * plane + at]);
  float d = 0.0F;
  if (p.count == 2 && inside[0] && inside[1]) {
    d = fabsf(kLumaR * (value[0][0] - value[1][0]) + kLumaG * (value[0][1] - value[1][1]) +
              kLumaB * (value[0][2] - value[1][2]));
  } else if (p.count == 1) {
    // La plaque : l'écart le plus grand des trois canaux (spec layers.h, étape 4).
#pragma unroll
    for (int c = 0; c < 3; ++c) d = fmaxf(d, fabsf(value[0][c] - here[c]));
  }
  const float k = 1.0F - fminf(fmaxf((d - p.guard_low) / (p.guard_high - p.guard_low), 0.0F), 1.0F);
  const float mk = m * k;
  if (mk <= 0.0F) return;
  const float inv = 1.0F / total;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const float v = here[c] + mk * (sum[c] * inv - here[c]);
    out[c * plane + at] = __float2half_rn(fminf(fmaxf(v, 0.0F), 1.0F));
  }
}

LayerGridArgs grid_args(const kernels::LayerGrid& g) {
  LayerGridArgs args;
  args.cols = g.cols;
  args.rows = g.rows;
  args.tile_w = g.tile_w;
  args.tile_h = g.tile_h;
  for (int i = 0; i < kernels::kLayerGridWords; ++i) args.bits[i] = g.bits[static_cast<std::size_t>(i)];
  return args;
}

}  // namespace

LayerArgs layer_args(const kernels::LayerParams& params) {
  LayerArgs args;
  args.count = params.count;
  for (int i = 0; i < 2; ++i) {
    const auto& source = params.sources[static_cast<std::size_t>(i)];
    for (int k = 0; k < 6; ++k) args.shift[i][k] = source.shift[static_cast<std::size_t>(k)];
    args.weight[i] = source.weight;
  }
  args.guard_low = params.guard_low;
  args.guard_high = params.guard_high;
  args.cubic = params.cubic ? 1 : 0;
  args.grid = grid_args(params.grid);
  if (params.source_grid.cols > 0) args.source_grid = grid_args(params.source_grid);
  return args;
}

cudaError_t launch_compose_layers(const LayerArgs& args, cudaStream_t stream) {
  if (args.out == nullptr || args.a == nullptr || args.width < 1 || args.height < 1 || args.count < 1 ||
      args.count > 2 || (args.count == 2 && args.b == nullptr) || args.grid.cols < 1 || args.grid.rows < 1 ||
      !(args.guard_high > args.guard_low))
    return cudaErrorInvalidValue;
  const dim3 block(32, 8);
  const dim3 grid((static_cast<unsigned>(args.width) + 31U) / 32U,
                  (static_cast<unsigned>(args.height) + 7U) / 8U);
  compose_layers<<<grid, block, 0, stream>>>(args);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
