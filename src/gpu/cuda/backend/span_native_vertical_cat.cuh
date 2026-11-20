#pragma once

#include <type_traits>

#include "gpu/cuda/backend/span_native_conv.cuh"

namespace kirari::gpu::cuda::span_native::vertical {

// La porte reste en mémoire partagée. Les trois autres tranches sont déjà en NHWC.
// Les lectures ordinaires permettent à cat.in[0] et cat.out de désigner le même tampon.

template <int C>
__device__ __forceinline__ void gather_cat(std::uint32_t (&a)[4], const __half* in, const Params& cat, int x,
                                           int y, int ch) {
  const __half* row = in + static_cast<long>(y) * cat.fw * C + ch;
  if (x < cat.fw) {
    a[0] = *reinterpret_cast<const unsigned*>(row + x * C);
    a[2] = *reinterpret_cast<const unsigned*>(row + x * C + 8);
  }
  if (x + 8 < cat.fw) {
    a[1] = *reinterpret_cast<const unsigned*>(row + (x + 8) * C);
    a[3] = *reinterpret_cast<const unsigned*>(row + (x + 8) * C + 8);
  }
}

template <int C>
__device__ __forceinline__ void fused_cat(__half* s, const Params& cat, int x0, int y0, int warp, int lane) {
  constexpr int kKC = C / 16, kPitch = C + 8;
  const int g = lane >> 2, t = lane & 3, y = y0 + warp;
  constexpr int kWidth = 32;
  const auto saddr = smem_u32(s);
  const auto base = saddr + static_cast<unsigned>(((warp * kWidth + (lane & 15)) * kPitch + (lane >> 4) * 8) *
                                                  sizeof(__half));
  Frag<Acc::kF16> acc[2][C / 8];
#pragma unroll
  for (auto& row : acc)
#pragma unroll
    for (auto& frag : row) frag.clear();
  const auto multiply = [&](auto source_index) {
    constexpr int src = decltype(source_index)::value;
#pragma unroll
    for (int kc = 0; kc < kKC; ++kc) {
      std::uint32_t a[2][4]{};
#pragma unroll
      for (int f = 0; f < 2; ++f) {
        if constexpr (src == 1) {
          ldmatrix4(a[f], base + static_cast<unsigned>((f * 16 * kPitch + kc * 16) * sizeof(__half)));
        } else if (y >= cat.y0 && y < row_end(cat)) {
          gather_cat<C>(a[f], cat.in[src], cat, x0 + f * 16 + g, y, kc * 16 + t * 2);
        }
      }
#pragma unroll
      for (int np = 0; np < kKC; ++np) {
        const uint4 w = __ldg(cat.w + ((((src * kKC + kc) * kKC) + np) * 32 + lane));
#pragma unroll
        for (int f = 0; f < 2; ++f) {
          acc[f][2 * np].mma(a[f], w.x, w.y);
          acc[f][2 * np + 1].mma(a[f], w.z, w.w);
        }
      }
    }
  };
  multiply(std::integral_constant<int, 0>{});
  multiply(std::integral_constant<int, 1>{});
  multiply(std::integral_constant<int, 2>{});
  multiply(std::integral_constant<int, 3>{});
  __syncwarp();
#pragma unroll
  for (int n = 0; n < C / 8; ++n) {
    const int ch = n * 8 + t * 2;
    const auto bias = __ldg(reinterpret_cast<const float2*>(cat.bias + ch));
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        bool hit = false;
        const auto v = sums(acc[f][n], h, hit);
        const int x = f * 16 + g + h * 8;
        *reinterpret_cast<__half2*>(s + (warp * kWidth + x) * kPitch + ch) =
            to_half2(v.x + bias.x, v.y + bias.y);
      }
  }
}

}  // namespace kirari::gpu::cuda::span_native::vertical
