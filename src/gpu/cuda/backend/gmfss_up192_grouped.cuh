#pragma once

#include <algorithm>
#include <cstddef>

#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::gmfss_up192_grouped {
namespace rn = rife_native;

inline constexpr int kInput = 192, kOutput = 128, kCT = 64, kGroups = kOutput / kCT;
inline constexpr int kSpatialWarps = rn::kWarps / kGroups, kPitch = kCT + 8, kWidth = rn::kTW;
inline constexpr int kKc = 64, kRows = 2, kHeight = kRows * kSpatialWarps;
inline constexpr int kSmem =
    2 * std::max((kHeight + 1) * (kWidth + 1) * (kKc + 8), kHeight* kWidth* kPitch* kGroups);
inline constexpr auto kAcc = rn::Acc::kF16;
static_assert(kGroups * kSpatialWarps == rn::kWarps && kSmem <= 48 * 1024);

__device__ __forceinline__ void store(const __half* shared, const rn::Params& p, int y0, int x0, int phase) {
  constexpr int pixels = kHeight * kWidth;
  constexpr int group_words = pixels * kPitch, chunks = kCT / 8, output_chunks = kOutput / 8;
  for (int i = static_cast<int>(threadIdx.x); i < pixels * output_chunks; i += rn::kThreads) {
    const int pixel = (i / chunks) % pixels;
    const int group = i / (pixels * chunks);
    const int c = (i % chunks) * 8;
    const int y = y0 + pixel / kWidth, x = x0 + pixel % kWidth;
    if (y >= p.oh || x >= p.ow) continue;
    const long out = (2L * y + (phase >> 1)) * (2L * p.ow) + 2L * x + (phase & 1);
    const auto v = *reinterpret_cast<const uint4*>(shared + group * group_words + pixel * kPitch + c);
    *reinterpret_cast<uint4*>(p.out + out * p.out_ld + group * kCT + c) = v;
  }
}

// Deux groupes CT64 d'une phase partagent la lecture et le PReLU de la même tuile d'entrée.
__global__ void __launch_bounds__(rn::kThreads, 2) conv(rn::Params p) {
  constexpr int halo_h = kHeight + 1, halo_w = kWidth + 1;
  constexpr int words = kHeight * kWidth * kPitch;
  extern __shared__ __align__(16) unsigned char memory[];
  auto* shared = reinterpret_cast<__half*>(memory);
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int stripe = warp % kSpatialWarps, group = warp / kSpatialWarps;
  const int y = static_cast<int>(blockIdx.y) * kHeight, x = static_cast<int>(blockIdx.x) * kWidth;
  const int phase = static_cast<int>(blockIdx.z), tile = phase * kGroups + group;
  const int kcs = p.cin / 16;
  const auto* weights = p.w + static_cast<long>(tile) * 4 * kcs * (kCT / 16) * 32;
  rn::Frag<kAcc> sums[kRows * 2][kCT / 8];
#pragma unroll
  for (int f = 0; f < kRows * 2; ++f)
#pragma unroll
    for (int n = 0; n < kCT / 8; ++n) sums[f][n].clear();

#pragma unroll 1
  for (int q = 0; q < p.cin / kKc; ++q) {
    if (q != 0) __syncthreads();
    rn::load_nhwc<kKc, 1, halo_h, halo_w>(shared, p, q * kKc, y + (phase >> 1) - 1, x + (phase & 1) - 1);
    rn::cp_async_wait_all();
    __syncthreads();
    if (p.in_slope != 1.F) {
      rn::prelu_tile<kKc, halo_h * halo_w>(shared, p.in_slope);
      __syncthreads();
    }
    rn::accumulate<kKc, kCT, 2, 1, kRows, kAcc, false>(sums, shared, weights, q, kcs, 0, 4, stripe, lane);
  }
  __syncthreads();
  rn::stage<kCT, kRows, kAcc>(sums, shared + group * words, p.bias + tile * kCT, stripe, lane);
  __syncthreads();
  store(shared, p, y, x, phase);
}

inline void launch(dim3 grid, int bytes, cudaStream_t stream, const rn::Params& p) {
  conv<<<grid, rn::kThreads, static_cast<std::size_t>(bytes), stream>>>(p);
}
}  // namespace kirari::gpu::cuda::gmfss_up192_grouped
