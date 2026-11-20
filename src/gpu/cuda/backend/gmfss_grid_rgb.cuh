#pragma once

#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::gmfss_grid_rgb {
namespace rn = rife_native;

__device__ __forceinline__ __half clamp_half(__half x) {
  return __float2half(fminf(fmaxf(__half2float(x), 0.F), 1.F));
}

// Conv_last 64→3. Le paquet CT=16 reste commun, mais le second fragment ne
// contribue pas à l'image. La somme et l'arrondi précédant la saturation
// suivent les mêmes instructions que pour conv_kernel puis unpack_image.
template <int Kc, int Rows, rn::Acc A>
__global__ void __launch_bounds__(rn::kThreads, 2) conv_rgb(rn::Params p) {
  extern __shared__ __align__(16) unsigned char shared[];
  auto* input = reinterpret_cast<__half*>(shared);
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int y0 = static_cast<int>(blockIdx.y) * rn::kWarps * Rows, x0 = static_cast<int>(blockIdx.x) * rn::kTW;
  rn::Frag<A> acc[Rows * 2][2];
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
    if (y >= p.oh) continue;
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
}  // namespace kirari::gpu::cuda::gmfss_grid_rgb
