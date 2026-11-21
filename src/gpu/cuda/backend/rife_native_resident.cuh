// Le résiduel C32 est le pixel central de l'entrée déjà chargée pour chaque MMA.
#pragma once

#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::rife_native::resident {
using Stripe = Geometry<3, 1, 1>;
inline constexpr int kChannels = 32, kPitch = kChannels + 8;
inline constexpr int kInputValues = Stripe::kIR * Stripe::kIC * kPitch;

__device__ __forceinline__ __half* at_center(__half* tile, int y, int x, int channel) {
  return tile + ((y + 1) * Stripe::kIC + x + 1) * kPitch + channel;
}

__device__ __forceinline__ void stage_resident(const Frag<Acc::kF16> (&acc)[2][kChannels / 8], __half* tile,
                                               const Params& p, int warp, int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int n = 0; n < kChannels / 8; ++n) {
    const int ch = n * 8 + 2 * t;
    const float2 bias = __ldg(reinterpret_cast<const float2*>(p.bias + ch));
    const float2 slope = __ldg(reinterpret_cast<const float2*>(p.slope + ch));
#pragma unroll
    for (int fragment = 0; fragment < 2; ++fragment) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int x = 16 * fragment + g + 8 * h;
        auto* skip = reinterpret_cast<__half2*>(at_center(tile, warp, x, ch));
        const float2 sum = acc[fragment][n].pair(h);
        const __half2 biased = to_half2(sum.x + bias.x, sum.y + bias.y);
        const float2 active = activate_pair(add_residual_pair(biased, *skip), slope);
        *skip = to_half2(active.x, active.y);
      }
    }
  }
}

__global__ void __launch_bounds__(kThreads, 2) resident_residual_kernel(Params p) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto* tile = reinterpret_cast<__half*>(storage);
  const int top = static_cast<int>(blockIdx.y) * Stripe::kTH;
  const int left = static_cast<int>(blockIdx.x) * kTW;
  const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;
  Frag<Acc::kF16> acc[2][kChannels / 8];
#pragma unroll
  for (auto& fragment : acc)
#pragma unroll
    for (auto& channels : fragment) channels.clear();

  load_nhwc<kChannels, 1, Stripe::kIR, Stripe::kIC>(tile, p, 0, top - 1, left - 1);
  cp_async_wait_all();
  __syncthreads();
  accumulate<kChannels, kChannels, 3, 1, 1, Acc::kF16>(acc, tile, p.w, 0, kChannels / 16, 0, 9, warp, lane);
  __syncthreads();
  stage_resident(acc, tile, p, warp, lane);
  __syncthreads();

  constexpr int kVectors = kChannels / 8;
  for (int i = tid; i < Stripe::kTH * kTW * kVectors; i += kThreads) {
    const int pixel = i / kVectors, ch = (i % kVectors) * 8;
    const int y = pixel / kTW, x = pixel % kTW;
    if (top + y >= p.oh || left + x >= p.ow) continue;
    auto* out = p.out + (static_cast<long>(top + y) * p.ow + left + x) * p.out_ld + ch;
    *reinterpret_cast<uint4*>(out) = *reinterpret_cast<const uint4*>(at_center(tile, y, x, ch));
  }
}

inline void launch(dim3 grid, int smem, cudaStream_t stream, const Params& p) {
  resident_residual_kernel<<<grid, kThreads, static_cast<std::size_t>(smem), stream>>>(p);
}
}  // namespace kirari::gpu::cuda::rife_native::resident
