#pragma once

#include "gpu/cuda/backend/gmfss_shuffle_reuse.cuh"
#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::gmfss_grid_rgb {
namespace rn = rife_native;

__device__ __forceinline__ __half clamp_half(__half x) {
  return __float2half(fminf(fmaxf(__half2float(x), 0.F), 1.F));
}

template <int Kc, int Rows, rn::Acc A>
__device__ __forceinline__ void ring_sums(rn::Frag<A> (&acc)[Rows * 2][2], __half* input, rn::Params p,
                                          int y0, int x0, int warp, int lane) {
  using G = rn::Geometry<3, 1, Rows>;
  constexpr int chunks = Kc / 8, ld = Kc + 8;
  constexpr int mask = gmfss_shuffle_reuse::kRingRows - 1;
#pragma unroll
  for (int f = 0; f < Rows * 2; ++f)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) acc[f][nt].clear();
#pragma unroll 1
  for (int q = 0; q < 64 / Kc; ++q) {
    if (q > 0) __syncthreads();
    const auto base = rn::smem_u32(input);
    for (int i = static_cast<int>(threadIdx.x); i < G::kIR * G::kIC * chunks; i += rn::kThreads) {
      const int pixel = i / chunks, c = (i % chunks) * 8;
      const int y = y0 - 1 + pixel / G::kIC, x = x0 - 1 + pixel % G::kIC;
      const bool ok = y >= 0 && y < p.ih && x >= 0 && x < p.iw;
      const long at = static_cast<long>(y & mask) * p.iw + x;
      const __half* ptr = ok ? p.in + at * p.in_ld + q * Kc + c : p.in;
      rn::cp_async16(base + static_cast<std::uint32_t>((pixel * ld + c) * 2), ptr, ok);
    }
    rn::cp_async_wait_all();
    __syncthreads();
    rn::accumulate<Kc, 16, 3, 1, Rows, A>(acc, input, p.w, q, 4, 0, 9, warp, lane);
  }
}

// Conv_last 64→3. Le paquet CT=16 reste commun, mais le second fragment ne
// contribue pas à l'image. La somme et l'arrondi précédant la saturation
// suivent les mêmes instructions que pour conv_kernel puis unpack_image.
template <int Kc, int Rows, rn::Acc A, bool Ring>
__device__ __forceinline__ void rgb(rn::Params p, int first_y, int last_y, __half* input) {
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int y0 = first_y + static_cast<int>(blockIdx.y) * rn::kWarps * Rows;
  const int x0 = static_cast<int>(blockIdx.x) * rn::kTW;
  rn::Frag<A> acc[Rows * 2][2];
  if constexpr (Ring)
    ring_sums<Kc, Rows, A>(acc, input, p, y0, x0, warp, lane);
  else
    rn::conv_sums<Kc, 16, 3, 1, Rows, rn::In::kNhwc, A, false>(acc, input, p, y0, x0, 0, 0, warp, lane);

  const int channel = 2 * (lane % 4);
  if (channel >= 3) return;
  const float b0 = __ldg(p.bias + channel);
  const float b1 = channel == 0 ? __ldg(p.bias + 1) : 0.F;
  const long plane = static_cast<long>(p.oh) * p.ow;
#pragma unroll
  for (int f = 0; f < Rows * 2; ++f) {
    const int y = y0 + warp * Rows + f / 2;
    const int x = x0 + 16 * (f % 2) + lane / 4;
    if (y >= last_y) continue;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int px = x + 8 * h;
      if (px >= p.ow) continue;
      const float2 v = acc[f][0].pair(h);
      const auto rounded = rn::to_half2(v.x + b0, v.y + b1);
      const long i = static_cast<long>(y) * p.ow + px;
      p.out[channel * plane + i] = clamp_half(__low2half(rounded));
      if (channel == 0) p.out[plane + i] = clamp_half(__high2half(rounded));
    }
  }
}

template <int Kc, int Rows, rn::Acc A>
__global__ void __launch_bounds__(rn::kThreads, 2) conv_rgb(rn::Params p) {
  extern __shared__ __align__(16) unsigned char shared[];
  rgb<Kc, Rows, A, false>(p, 0, p.oh, reinterpret_cast<__half*>(shared));
}

template <int Kc>
__global__ void __launch_bounds__(rn::kThreads, 2) conv_rgb_ring(rn::Params p, int first_y, int last_y) {
  extern __shared__ __align__(16) unsigned char shared[];
  rgb<Kc, 2, rn::Acc::kF32, true>(p, first_y, last_y, reinterpret_cast<__half*>(shared));
}
}  // namespace kirari::gpu::cuda::gmfss_grid_rgb
