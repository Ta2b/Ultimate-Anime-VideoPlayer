#pragma once

#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::gmfss_shuffle_reuse {
namespace rn = rife_native;

template <int Rows>
constexpr int smem_bytes() {
  using G = rn::Geometry<3, 1, Rows>;
  constexpr int in = G::kIR * G::kIC * (64 + 8) * sizeof(__half);
  constexpr int out = G::kTH * rn::kTW * (64 + 8) * sizeof(__half);
  return in + out;
}

// PixelShuffle(2) : le halo et le PReLU éventuel servent à plusieurs phases.
// Les poids, biais, ordre des sommes et stores sont ceux des phases indépendantes.
template <int Rows, int Phases, rn::Acc A>
__global__ void __launch_bounds__(rn::kThreads, 2) conv_shuffle(rn::Params p) {
  static_assert(Phases == 2 || Phases == 4);
  using G = rn::Geometry<3, 1, Rows>;
  constexpr int kKc = 64, kCt = 64, kSteps = kKc / 16;
  extern __shared__ __align__(16) unsigned char shared[];
  auto* input = reinterpret_cast<__half*>(shared);
  auto* output = input + G::kIR * G::kIC * (kKc + 8);
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int y0 = static_cast<int>(blockIdx.y) * G::kTH, x0 = static_cast<int>(blockIdx.x) * rn::kTW;
  const int first = static_cast<int>(blockIdx.z) * Phases;
  rn::load_nhwc<kKc, 1, G::kIR, G::kIC>(input, p, 0, y0 - 1, x0 - 1);
  rn::cp_async_wait_all();
  __syncthreads();
  if (p.in_slope != 1.F) {
    rn::prelu_tile<kKc, G::kIR * G::kIC>(input, p.in_slope);
    __syncthreads();
  }
#pragma unroll 1
  for (int phase = first; phase < first + Phases; ++phase) {
    rn::Frag<A> acc[Rows * 2][kCt / 8];
#pragma unroll
    for (int f = 0; f < Rows * 2; ++f)
#pragma unroll
      for (int nt = 0; nt < kCt / 8; ++nt) acc[f][nt].clear();
    const uint4* weights = p.w + phase * 9 * kSteps * (kCt / 16) * 32;
    rn::accumulate<kKc, kCt, 3, 1, Rows, A, false>(acc, input, weights, 0, kSteps, 0, 9, warp, lane);
    rn::stage<kCt, Rows, A>(acc, output, p.bias + phase * kCt, warp, lane);
    __syncthreads();
    rn::store<kCt, Rows, rn::Out::kPhase>(output, p, y0, x0, phase, 0);
    if (phase + 1 < first + Phases) __syncthreads();  // réemploi du tampon de sortie
  }
}
}  // namespace kirari::gpu::cuda::gmfss_shuffle_reuse
