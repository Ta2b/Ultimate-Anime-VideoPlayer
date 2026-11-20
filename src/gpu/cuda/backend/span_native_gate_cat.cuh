#pragma once

#include "gpu/cuda/backend/span_native_conv.cuh"

namespace kirari::gpu::cuda::span_native {

// Première source de conv_cat. Les quatre mots du fragment A sont chargés selon le même ordre
// m16k16 que gate les assemble en registres. La mémoire partagée garde la sortie Half de la porte.
template <int C>
__device__ __forceinline__ void cat_first_source(Frag<Acc::kF16> (&acc)[kMF][C / 8], const Params& cat,
                                                 int ty0, int tx0, int warp, int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int kc = 0; kc < C / 16; ++kc) {
    uint4 weights[C / 16];
#pragma unroll
    for (int np = 0; np < C / 16; ++np) weights[np] = __ldg(cat.w + (kc * (C / 16) + np) * 32 + lane);
#pragma unroll
    for (int f = 0; f < kMF; ++f) {
      const int y = ty0 + warp * kRows + (f >> 1);
      const int x = tx0 + 16 * (f & 1) + g;
      const int ch = kc * 16 + 2 * t;
      std::uint32_t a[4] = {};
      // S0 est aussi la sortie finale de cat. Les lectures de la tuile précèdent les écritures et
      // restent sur les charges globales ordinaires, puisque le tampon est modifié par le noyau.
      if (y < cat.fh && x < cat.fw) {
        const __half* p = cat.in[0] + (static_cast<long>(y) * cat.fw + x) * C + ch;
        a[0] = *reinterpret_cast<const std::uint32_t*>(p);
        a[2] = *reinterpret_cast<const std::uint32_t*>(p + 8);
      }
      if (y < cat.fh && x + 8 < cat.fw) {
        const __half* p = cat.in[0] + (static_cast<long>(y) * cat.fw + x + 8) * C + ch;
        a[1] = *reinterpret_cast<const std::uint32_t*>(p);
        a[3] = *reinterpret_cast<const std::uint32_t*>(p + 8);
      }
#pragma unroll
      for (int np = 0; np < C / 16; ++np) {
        acc[f][2 * np].mma(a, weights[np].x, weights[np].y);
        acc[f][2 * np + 1].mma(a, weights[np].z, weights[np].w);
      }
    }
  }
}

// Dernière 3×3 et porte, puis la 1×1 conv_cat. La sortie de la porte est sa deuxième source et ne
// quitte pas la mémoire partagée. Le reste du calcul est dans l'ordre de conv_kernel.
template <int C, bool Shared = false>
__global__ void __launch_bounds__(kThreads) gate_cat_kernel(Params last, Params cat) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* s = reinterpret_cast<__half*>(smem_raw);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int y = cat.y0 + static_cast<int>(blockIdx.y) * kTH, x = static_cast<int>(blockIdx.x) * kTW;
  {
    Frag<Acc::kF16> acc[kMF][C / 8];
#pragma unroll
    for (auto& row : acc)
#pragma unroll
      for (auto& fragment : row) fragment.clear();
    load_nhwc<C, 1, Shared>(s, last.in[0], C, last.fh, last.fw, y, x);
    cp_async_wait_all();
    __syncthreads();
    if constexpr (Shared) {
      constexpr int kTileBytes = (kTH + 2) * (kTW + 2) * Tile<C, true>::kPitch * 16;
      accumulate_shared<C, C, 9, Acc::kF16>(acc, s, last.w, smem_u32(s) + kTileBytes, 0, warp, lane);
    } else {
      accumulate<C, C, 9, Acc::kF16>(acc, s, last.w, 0, warp, lane);
    }
    __syncthreads();
    bool hit = false;
    gate_tile<C, Acc::kF16>(acc, s, last, y, x, warp, lane, hit);
  }
  __syncthreads();

  Frag<Acc::kF16> acc[kMF][C / 8];
#pragma unroll
  for (auto& row : acc)
#pragma unroll
    for (auto& fragment : row) fragment.clear();
  cat_first_source<C>(acc, cat, y, x, warp, lane);
  accumulate<C, C, 1, Acc::kF16>(acc, s, cat.w, C / 16, warp, lane);
  for (int i = 2; i < 4; ++i) {
    __syncthreads();
    const __half* from = i == 2 ? cat.in[2] : cat.in[3];
    load_nhwc<C, 0>(s, from, C, cat.fh, cat.fw, y, x);
    cp_async_wait_all();
    __syncthreads();
    accumulate<C, C, 1, Acc::kF16>(acc, s, cat.w, i * (C / 16), warp, lane);
  }
  __syncthreads();
  finish<C, Mode::kPlain, false, Acc::kF16>(acc, s, cat, y, x, warp, lane);
}

}  // namespace kirari::gpu::cuda::span_native
