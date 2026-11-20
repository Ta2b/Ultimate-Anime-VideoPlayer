// La composition par calques (layer_kernels.h) : un fil par pixel de sortie. Les fils dont le masque des
// tuiles est nul sortent avant toute lecture des sources : sur un plan de dialogue, seul le décor coûte.
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/layer_kernels.h"
#include "gpu/cuda/backend/layer_sample.cuh"

namespace kirari::gpu::cuda {
namespace {

__global__ void __launch_bounds__(256) compose_layers(LayerArgs p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const layer::Pixel v = layer::sample(p, x, y);
  if (v.m <= 0.0F) return;
  const std::int64_t plane = static_cast<std::int64_t>(p.width) * p.height;
  const std::int64_t at = static_cast<std::int64_t>(y) * p.width + x;
  auto* out = static_cast<__half*>(p.out);
  float here[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) here[c] = __half2float(out[c * plane + at]);
  float d = v.d;
  if (p.count == 1) {
#pragma unroll
    for (int c = 0; c < 3; ++c) d = fmaxf(d, fabsf(v.first[c] - here[c]));
  }
  const float mk = layer::mix_weight(p, v, d, p.count == 1 ? v.source_mask : 1.0F);
  if (mk <= 0.0F) return;
  const float inv = 1.0F / v.total;
#pragma unroll
  for (int c = 0; c < 3; ++c) out[c * plane + at] = layer::finish(here[c], mk, v.sum[c], inv);
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
