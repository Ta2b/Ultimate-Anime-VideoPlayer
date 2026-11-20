#include <cstdint>
#include <cuda_fp16.h>
#include <limits>

#include "gpu/cuda/backend/gmflow_scale_kernels.h"
#include "gpu/cuda/backend/tfix_fused.cuh"
#include "prepare/gmflow_attention.h"

namespace kirari::gpu::cuda {
namespace {

constexpr int kChannels = prepare::kGmflowModel;
constexpr int kAxisChannels = kChannels / 2;
constexpr int kPack = 8;  // un uint4
constexpr int kLanes = kChannels / kPack;
constexpr int kThreads = 128;
constexpr int kPixels = kThreads / kLanes;

struct Grid {
  int height, width, wh, ww;
  int pixels() const { return height * width; }
};

Grid grid_of(int height, int width, int splits) {
  const auto pixels = static_cast<std::int64_t>(height) * width;
  if (height < 1 || width < 1 || splits < 1 || height % splits != 0 || width % splits != 0 ||
      pixels > std::numeric_limits<int>::max() - kPixels)
    return {};
  return {height, width, height / splits, width / splits};
}

bool aligned(const void* p, std::size_t bytes) {
  return p != nullptr && reinterpret_cast<std::uintptr_t>(p) % bytes == 0;
}

__global__ void position_kernel(Grid g, int count, __half* table) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  const int c = i % kAxisChannels;
  const int axis = i / kAxisChannels;
  const int j = axis < g.wh ? axis : axis - g.wh;
  const int n = axis < g.wh ? g.wh : g.ww;
  constexpr float kTurn = 6.283185307179586F;
  const float frequency = powf(10000.F, 2.F * static_cast<float>(c / 2) / kAxisChannels);
  const float angle = ((static_cast<float>(j) + 1.F) / (static_cast<float>(n) + 1e-6F)) * kTurn / frequency;
  table[i] = __float2half_rn(c % 2 == 0 ? sinf(angle) : cosf(angle));
}

__device__ __forceinline__ float2 expanded_flow(const float2* flow, Grid g, int x, int y) {
  const int w = g.width / 2, h = g.height / 2;
  const float sx = static_cast<float>(w - 1) / static_cast<float>(g.width - 1);
  const float sy = static_cast<float>(h - 1) / static_cast<float>(g.height - 1);
  const float px = sx * static_cast<float>(x), py = sy * static_cast<float>(y);
  const int x0 = static_cast<int>(px), y0 = static_cast<int>(py);
  const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
  const warp::Axis ax{x0, x1, px - static_cast<float>(x0)}, ay{y0, y1, py - static_cast<float>(y0)};
  const float2 a = flow[y0 * w + x0], b = flow[y0 * w + x1], c = flow[y1 * w + x0], d = flow[y1 * w + x1];
  return make_float2(2.F * warp::lerp2(a.x, b.x, c.x, d.x, ax, ay),
                     2.F * warp::lerp2(a.y, b.y, c.y, d.y, ax, ay));
}

__device__ __forceinline__ uint4 warp_pack(const __half* image, int c, const tfix::ZeroTap& tap) {
  float2 sum[kPack / 2]{};
#pragma unroll
  for (int corner = 0; corner < 4; ++corner) {
    if (tap.weight[corner] == 0.F) continue;
    const auto at = static_cast<std::int64_t>(tap.index[corner]) * kChannels + c;
    const auto raw = *reinterpret_cast<const uint4*>(image + at);
    const auto* halfs = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int i = 0; i < kPack / 2; ++i) {
      const float2 pair = __half22float2(halfs[i]);
      sum[i].x += tap.weight[corner] * pair.x;
      sum[i].y += tap.weight[corner] * pair.y;
    }
  }
  uint4 result;
  auto* halfs = reinterpret_cast<__half2*>(&result);
#pragma unroll
  for (int i = 0; i < kPack / 2; ++i) halfs[i] = __floats2half2_rn(sum[i].x, sum[i].y);
  return result;
}

template <bool Quarter, bool Decimate = false>
__global__ void __launch_bounds__(kThreads)
    input_kernel(Grid g, const __half* features, const __half* positions, const float2* coarse,
                 float2* quarter, __half* out, bool swapped) {
  static_assert(!Quarter || !Decimate);
  const int c = static_cast<int>(threadIdx.x) % kLanes * kPack;
  const int pixel = static_cast<int>(blockIdx.x) * kPixels + static_cast<int>(threadIdx.x) / kLanes;
  const int plane = g.height * g.width;
  const int x = pixel % g.width, y = pixel / g.width;
  const int batch = static_cast<int>(blockIdx.y);
  const auto image_stride = static_cast<std::int64_t>(plane) * kChannels;
  const int side = batch < 2 ? batch : 3 - batch;
  const int source = Quarter || Decimate ? side ^ static_cast<int>(swapped) : side;
  const __half* image = features + source * image_stride * (Decimate ? 4 : 1);
  uint4 value;
  if constexpr (Quarter) {
    if (batch >= 2) {
      float2 f{};
      if (c == 0 && pixel < plane) {
        const int coarse_plane = plane / 4;
        f = expanded_flow(coarse + (batch - 2) * static_cast<std::int64_t>(coarse_plane), g, x, y);
        quarter[(batch - 2) * static_cast<std::int64_t>(plane) + pixel] = f;
      }
      // Une moitié de warp partage le flot d'un pixel. Même la fin de grille participe aux échanges.
      f.x = __shfl_sync(0xffffffffU, f.x, 0, kLanes);
      f.y = __shfl_sync(0xffffffffU, f.y, 0, kLanes);
      if (pixel >= plane) return;
      const auto tap =
          tfix::zero_tap(static_cast<float>(x) + f.x, static_cast<float>(y) + f.y, g.height, g.width);
      value = warp_pack(image, c, tap);
    }
  }
  if (!Quarter || batch < 2) {
    if (pixel >= plane) return;
    const auto at = Decimate ? 4 * static_cast<std::int64_t>(y) * g.width + 2 * static_cast<std::int64_t>(x)
                             : static_cast<std::int64_t>(pixel);
    value = *reinterpret_cast<const uint4*>(image + at * kChannels + c);
  }
  const int axis = c < kAxisChannels ? y % g.wh : g.wh + x % g.ww;
  const auto pos_index = static_cast<std::int64_t>(axis) * kAxisChannels + c % kAxisChannels;
  const auto pos = *reinterpret_cast<const uint4*>(positions + pos_index);
  const auto* add = reinterpret_cast<const __half2*>(&pos);
  auto* dst = reinterpret_cast<__half2*>(&value);
#pragma unroll
  for (int i = 0; i < kPack / 2; ++i) dst[i] = __hadd2(dst[i], add[i]);
  const auto at = batch * image_stride + static_cast<std::int64_t>(pixel) * kChannels + c;
  *reinterpret_cast<uint4*>(out + at) = value;
}

}  // namespace

std::size_t gmflow_position_bytes(int height, int width, int splits) {
  const Grid g = grid_of(height, width, splits);
  const auto axes = static_cast<std::size_t>(g.wh) + static_cast<std::size_t>(g.ww);
  return axes <= std::numeric_limits<int>::max() / kAxisChannels - kThreads
             ? axes * kAxisChannels * sizeof(__half)
             : 0;
}

cudaError_t launch_gmflow_position(cudaStream_t stream, int height, int width, int splits, void* table) {
  const auto bytes = gmflow_position_bytes(height, width, splits);
  if (bytes == 0 || !aligned(table, 16)) return cudaErrorInvalidValue;
  const auto count = static_cast<int>(bytes / sizeof(__half));
  position_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      grid_of(height, width, splits), count, static_cast<__half*>(table));
  return cudaGetLastError();
}

cudaError_t launch_gmflow_eighth_input(cudaStream_t stream, const void* features, const void* positions,
                                       int height, int width, int splits, void* tokens, bool decimate,
                                       bool swapped) {
  const Grid g = grid_of(height, width, splits);
  if (g.pixels() == 0 || !aligned(features, 16) || !aligned(positions, 16) || !aligned(tokens, 16) ||
      (decimate && features == tokens))
    return cudaErrorInvalidValue;
  const dim3 grid(static_cast<unsigned>((g.pixels() + kPixels - 1) / kPixels), 2U);
  const auto* input = static_cast<const __half*>(features);
  const auto* pos = static_cast<const __half*>(positions);
  auto* result = static_cast<__half*>(tokens);
  if (decimate)
    input_kernel<false, true>
        <<<grid, kThreads, 0, stream>>>(g, input, pos, nullptr, nullptr, result, swapped);
  else
    input_kernel<false><<<grid, kThreads, 0, stream>>>(g, input, pos, nullptr, nullptr, result, false);
  return cudaGetLastError();
}

cudaError_t launch_gmflow_quarter_input(cudaStream_t stream, const void* features, const void* coarse_flow,
                                        const void* positions, int height, int width, int splits,
                                        void* tokens, void* quarter_flow, bool swapped) {
  const Grid g = grid_of(height, width, splits);
  if (g.pixels() == 0 || height % 2 != 0 || width % 2 != 0 || !aligned(features, 16) ||
      !aligned(positions, 16) || !aligned(tokens, 16) || !aligned(coarse_flow, alignof(float2)) ||
      !aligned(quarter_flow, alignof(float2)) || features == tokens || coarse_flow == quarter_flow)
    return cudaErrorInvalidValue;
  input_kernel<true>
      <<<dim3(static_cast<unsigned>((g.pixels() + kPixels - 1) / kPixels), 4U), kThreads, 0, stream>>>(
          g, static_cast<const __half*>(features), static_cast<const __half*>(positions),
          static_cast<const float2*>(coarse_flow), static_cast<float2*>(quarter_flow),
          static_cast<__half*>(tokens), swapped);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
