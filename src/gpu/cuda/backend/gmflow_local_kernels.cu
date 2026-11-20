#include <cstdint>
#include <cuda_fp16.h>
#include <limits>

#include "gpu/cuda/backend/gmflow_local_kernels.h"
#include "prepare/gmflow_attention.h"

namespace kirari::gpu::cuda {
namespace {

constexpr int kD = prepare::kGmflowModel;
constexpr int kWarp = 32;
constexpr int kThreads = 128;
constexpr int kPixels = kThreads / kWarp;
constexpr int kLaneChannels = kD / kWarp;
constexpr float kScaleLog2 = 1.4426950408889634F / 11.313708498984761F;
static_assert(kLaneChannels == 4);

struct Feature {
  float2 a, b;
};

__device__ __forceinline__ Feature read(const __half* channels, int lane) {
  const uint2 pack = *reinterpret_cast<const uint2*>(channels + lane * kLaneChannels);
  const auto* pairs = reinterpret_cast<const __half2*>(&pack);
  return {__half22float2(pairs[0]), __half22float2(pairs[1])};
}

__device__ __forceinline__ float dot(Feature a, Feature b) {
  float score = a.a.x * b.a.x + a.a.y * b.a.y + a.b.x * b.b.x + a.b.y * b.b.y;
#pragma unroll
  for (int s = kWarp / 2; s > 0; s /= 2) score += __shfl_down_sync(0xffffffffU, score, s);
  return score;
}

struct Params {
  const __half* query;
  const __half* keys;
  const float2* flow;
  float2* out;
  int height, width;
};

template <bool kPropagation>
__global__ void __launch_bounds__(kThreads) local_kernel(Params p) {
  constexpr int kRadius = kPropagation ? 1 : 4;
  const int lane = static_cast<int>(threadIdx.x) % kWarp;
  const int point = static_cast<int>(blockIdx.x) * kPixels + static_cast<int>(threadIdx.x) / kWarp;
  const int plane = p.width * p.height;
  if (point >= plane) return;  // Le warp entier a le même point.
  const int dir = static_cast<int>(blockIdx.y);
  const int x = point % p.width, y = point / p.width;
  const auto base = static_cast<std::int64_t>(dir) * plane;
  const __half* keys = p.keys + base * kD;
  const float2* flow = p.flow + base;
  const Feature q = read(p.query + (base + point) * kD, lane);
  float best = -INFINITY, denominator = 0.F, sx = 0.F, sy = 0.F;
#pragma unroll 1
  for (int dy = -kRadius; dy <= kRadius; ++dy) {
    const int ny = y + dy;
#pragma unroll 1
    for (int dx = -kRadius; dx <= kRadius; ++dx) {
      const int nx = x + dx;
      const bool valid = ny >= 0 && ny < p.height && nx >= 0 && nx < p.width;
      if (!kPropagation && !valid) continue;
      const int neighbor = valid ? ny * p.width + nx : 0;
      float score = 0.F;
      if (valid) score = dot(q, read(keys + static_cast<std::int64_t>(neighbor) * kD, lane)) * kScaleLog2;
      if (lane == 0) {
        float2 value;
        if constexpr (kPropagation)
          value = valid ? flow[neighbor] : make_float2(0.F, 0.F);
        else
          value = make_float2(static_cast<float>(dx), static_cast<float>(dy));
        const float next = fmaxf(best, score);
        const float rescale = exp2f(best - next);
        const float weight = exp2f(score - next);
        best = next;
        denominator = denominator * rescale + weight;
        sx = sx * rescale + weight * value.x;
        sy = sy * rescale + weight * value.y;
      }
    }
  }
  if (lane == 0) {
    float2 result = make_float2(sx / denominator, sy / denominator);
    if constexpr (!kPropagation) {
      const float2 initial = flow[point];
      result.x += initial.x;
      result.y += initial.y;
    }
    p.out[base + point] = result;
  }
}

// Rejeu des frontières du matcher local ONNX : chaque produit FP16 précède la
// réduction FP32, puis le softmax est stocké en FP16. Le produit suivant est en
// coordonnées absolues, donc un poids arrondi ne se simplifie plus en déplacement.
constexpr int kMatchWidth = 9;
constexpr int kMatchTaps = kMatchWidth * kMatchWidth;
// sqrt(128) est d’abord arrondie en Half, puis le noyau divise en FP32.
constexpr float kScoreScale = 1.F / 11.3125F;

__device__ __forceinline__ float round_to_half(float x) {
  return __half2float(__float2half_rn(x));
}

__device__ __forceinline__ float product_half_dot(uint2 q, uint2 k) {
  const auto* qh = reinterpret_cast<const __half2*>(&q);
  const auto* kh = reinterpret_cast<const __half2*>(&k);
  const auto a = __half22float2(__hmul2(qh[0], kh[0]));
  const auto b = __half22float2(__hmul2(qh[1], kh[1]));
  float sum = a.x + a.y + b.x + b.y;
#pragma unroll
  for (int s = kWarp / 2; s > 0; s /= 2) sum += __shfl_down_sync(0xffffffffU, sum, s);
  return sum;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int s = kWarp / 2; s > 0; s /= 2) value += __shfl_down_sync(0xffffffffU, value, s);
  return value;
}

__global__ void __launch_bounds__(kThreads) local_reference_match(Params p) {
  __shared__ float logits[kPixels][kMatchTaps];
  const int lane = static_cast<int>(threadIdx.x) % kWarp;
  const int warp = static_cast<int>(threadIdx.x) / kWarp;
  const int point = static_cast<int>(blockIdx.x) * kPixels + warp;
  const int plane = p.width * p.height;
  if (point >= plane) return;
  const int x = point % p.width, y = point / p.width;
  const auto base = static_cast<std::int64_t>(blockIdx.y) * plane;
  const auto* keys = reinterpret_cast<const uint2*>(p.keys + base * kD);
  const auto q = reinterpret_cast<const uint2*>(p.query)[(base + point) * kWarp + lane];
  float peak = -INFINITY;
#pragma unroll 1
  for (int tap = 0; tap < kMatchTaps; ++tap) {
    const int nx = x + tap % kMatchWidth - kMatchWidth / 2;
    const int ny = y + tap / kMatchWidth - kMatchWidth / 2;
    const bool valid = nx >= 0 && nx < p.width && ny >= 0 && ny < p.height;
    float value = -INFINITY;
    if (valid) {
      const auto k = keys[(static_cast<std::int64_t>(ny) * p.width + nx) * kWarp + lane];
      value = round_to_half(__fmul_rn(round_to_half(product_half_dot(q, k)), kScoreScale));
    }
    if (lane == 0) {
      logits[warp][tap] = value;
      peak = fmaxf(peak, value);
    }
  }
  __syncwarp();
  peak = __shfl_sync(0xffffffffU, peak, 0);
  float ex[(kMatchTaps + kWarp - 1) / kWarp];
  float denominator = 0.F;
#pragma unroll
  for (int i = 0; i < (kMatchTaps + kWarp - 1) / kWarp; ++i) {
    const int tap = lane + i * kWarp;
    const float v = tap < kMatchTaps ? round_to_half(expf(round_to_half(logits[warp][tap] - peak))) : 0.F;
    ex[i] = v;
    denominator += v;
  }
  denominator = __shfl_sync(0xffffffffU, warp_sum(denominator), 0);
  const float inverse = __fdiv_rn(1.F, round_to_half(denominator));
  float sx = 0.F, sy = 0.F;
#pragma unroll
  for (int i = 0; i < (kMatchTaps + kWarp - 1) / kWarp; ++i) {
    const int tap = lane + i * kWarp;
    const float weight = round_to_half(__fmul_rn(ex[i], inverse));
    const int nx = x + tap % kMatchWidth - kMatchWidth / 2;
    const int ny = y + tap / kMatchWidth - kMatchWidth / 2;
    sx += weight * static_cast<float>(nx);
    sy += weight * static_cast<float>(ny);
  }
  sx = warp_sum(sx);
  sy = warp_sum(sy);
  if (lane == 0) {
    const auto initial = p.flow[base + point];
    p.out[base + point] =
        make_float2(round_to_half(round_to_half(initial.x) + round_to_half(sx - static_cast<float>(x))),
                    round_to_half(round_to_half(initial.y) + round_to_half(sy - static_cast<float>(y))));
  }
}

bool aligned(const void* p, std::size_t bytes) {
  return p != nullptr && reinterpret_cast<std::uintptr_t>(p) % bytes == 0;
}

template <bool kPropagation>
cudaError_t launch(cudaStream_t stream, const void* query, const void* keys, const void* flow, int height,
                   int width, void* out, bool reference_match = false) {
  const auto area = static_cast<std::int64_t>(height) * width;
  if (height < 1 || width < 1 || area > std::numeric_limits<int>::max() - kPixels || !aligned(query, 16) ||
      !aligned(keys, 16) || !aligned(flow, alignof(float2)) || !aligned(out, alignof(float2)) ||
      query == out || keys == out || (kPropagation && flow == out))
    return cudaErrorInvalidValue;
  const auto* q = static_cast<const __half*>(query);
  const auto* k = kPropagation ? static_cast<const __half*>(keys) : q + 2 * area * kD;
  const Params p{q, k, static_cast<const float2*>(flow), static_cast<float2*>(out), height, width};
  const auto blocks = static_cast<unsigned>((area + kPixels - 1) / kPixels);
  const dim3 grid(blocks, 2);
  if constexpr (kPropagation) {
    local_kernel<true><<<grid, kThreads, 0, stream>>>(p);
  } else if (reference_match) {
    local_reference_match<<<grid, kThreads, 0, stream>>>(p);
  } else {
    local_kernel<false><<<grid, kThreads, 0, stream>>>(p);
  }
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_gmflow_local_match(cudaStream_t stream, const void* tokens, const void* initial,
                                      int height, int width, void* out, bool reference_match) {
  return launch<false>(stream, tokens, tokens, initial, height, width, out, reference_match);
}

cudaError_t launch_gmflow_local_propagate(cudaStream_t stream, const void* query, const void* keys,
                                          const void* flow, int height, int width, void* out) {
  return launch<true>(stream, query, keys, flow, height, width, out);
}

}  // namespace kirari::gpu::cuda
