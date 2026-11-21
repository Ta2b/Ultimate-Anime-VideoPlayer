// La dernière couche du dernier bloc du RIFE natif, fin comprise (rife_native_plan.h, pick_finish). Cette
// couche, la transposée 4 × 4 de pas 2, écrivait Y (52 canaux, 56 rangés, à la moitié de la pleine
// résolution : 58 Mo en 1920 × 1088), dont la fin (fused_finish) ne relisait que les 20 premiers : le flot
// du bloc et son masque, en quatre parités. Ici la couche ne calcule que sa première tuile de 32 canaux et
// fait la fin dans son épilogue, depuis la mémoire partagée où stage l'a rangée : les sommes de conv_kernel
// (même tuile, même ordre), le même arrondi FP16 qu'à l'écriture de Y, puis les fonctions de la fin
// (rife_finish.cuh). La sortie est donc celle de la couche suivie de fused_finish, au bit près ; Y n'est pas
// écrit.
#include <cstddef>

#include "gpu/cuda/backend/rife_finish.cuh"
#include "gpu/cuda/backend/rife_native_plan.h"

namespace kirari::gpu::cuda::rife_native {
namespace {

constexpr int kCT = 32;  // pas des poids pour la première tuile de Y
constexpr int kUsedChannels = 20;
constexpr int kHeadFragments = (kUsedChannels + 7) / 8;
constexpr int kHeadPitch = 8 * kHeadFragments;

// La fin sur la tuile de sortie (o : la mémoire partagée au pas Pitch). Le pixel (y, x) de la
// grille, à la phase (py, px), est le pixel (2y + py, 2x + px) de Y ; son canal 4c + 2a + b est le canal c de
// Z au point pleine résolution (2(2y + py) + a, 2(2x + px) + b). Un fil par point ; deux fils voisins font
// deux points voisins d'une ligne.
template <int Rows, int Pitch = kCT + 8>
__device__ __forceinline__ void finish_tile(const __half* o, const Params& p, const FusedFinishParams& f,
                                            int ty0, int tx0, int phase) {
  constexpr int kTH = kWarps * Rows;
  const int py = phase >> 1, px = phase & 1;
  const std::size_t plane = static_cast<std::size_t>(f.height) * static_cast<std::size_t>(f.width);
  for (int i = static_cast<int>(threadIdx.x); i < kTH * kTW * 4; i += kThreads) {
    const int b = i & 1, col = (i >> 1) % kTW, rest = (i >> 1) / kTW;
    const int a = rest & 1, row = rest >> 1;
    const int gy = ty0 + row, gx = tx0 + col;
    if (gy >= p.oh || gx >= p.ow) continue;
    const int x = 2 * (2 * gx + px) + b, y = 2 * (2 * gy + py) + a;
    const __half* v = o + (row * kTW + col) * Pitch + 2 * a + b;
    float z[5];
#pragma unroll
    for (int c = 0; c < 5; ++c) z[c] = __half2float(v[4 * c]);
    const std::size_t pixel =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(f.width) + static_cast<std::size_t>(x);
    float acc[4];
    finish::accumulated(f, pixel, plane, acc);
    finish::at(f, x, y, pixel, plane, acc, z);
  }
}

// La couche : la phase blockIdx.z, sa première tuile de kCT canaux (poids rangés pour kCT, p.tiles tuiles
// par phase), K entier.
template <int Kc, int Rows, Acc A>
__global__ void __launch_bounds__(kThreads, 2) finish_conv_kernel(Params p, FusedFinishParams f) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* s = reinterpret_cast<__half*>(smem_raw);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int ty0 = static_cast<int>(blockIdx.y) * kWarps * Rows, tx0 = static_cast<int>(blockIdx.x) * kTW;
  const int phase = static_cast<int>(blockIdx.z);
  Frag<A> acc[Rows * 2][kCT / 8];
  conv_sums<Kc, kCT, 2, 1, Rows, In::kNhwc, A, false>(acc, s, p, ty0, tx0, phase * p.tiles, 0, warp, lane);
  __syncthreads();  // la tuile d'entrée n'est plus lue : la mémoire partagée reçoit la sortie
  stage<kCT, Rows, A>(acc, s, p.bias + static_cast<long>(phase) * p.tiles * kCT, warp, lane);
  __syncthreads();
  finish_tile<Rows>(s, p, f, ty0, tx0, phase);
}

template <int Kc, int Rows, Acc A>
void finish_one(dim3 grid, int smem, cudaStream_t stream, const Params& p, const FusedFinishParams& f) {
  finish_conv_kernel<Kc, Rows, A><<<grid, kThreads, static_cast<std::size_t>(smem), stream>>>(p, f);
}

template <int Rows, Acc A>
FinishLaunch by_slice(int kc) {
  if (kc == 16) return finish_one<16, Rows, A>;
  if (kc == 32) return finish_one<32, Rows, A>;
  if (kc == 64) return finish_one<64, Rows, A>;
  return nullptr;
}

// Seulement 24 canaux, dont les 20 lus par le warp final.
__device__ __forceinline__ void stage_head(const Frag<Acc::kF16> (&acc)[2][kHeadFragments], __half* output,
                                           const float* bias, int warp, int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int f = 0; f < 2; ++f) {
    const int q = warp * kTW + 16 * f + g;
#pragma unroll
    for (int n = 0; n < kHeadFragments; ++n) {
      const int ch = n * 8 + 2 * t;
      const float2 b = __ldg(reinterpret_cast<const float2*>(bias + ch));
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float2 v = acc[f][n].pair(h);
        *reinterpret_cast<__half2*>(output + (q + 8 * h) * kHeadPitch + ch) = to_half2(v.x + b.x, v.y + b.y);
      }
    }
  }
}

// RIFE Lite en 1080p : les quatre phases partagent l’entrée, gardée jusqu’au dernier warp.
__global__ void __launch_bounds__(kThreads, 2) finish_four(Params p, FusedFinishParams f) {
  constexpr int kK = 32, kLd = kK + 8;
  constexpr int kRowsIn = kWarps + 2;
  constexpr int kColsIn = kTW + 2;
  extern __shared__ __align__(16) unsigned char raw[];
  auto* input = reinterpret_cast<__half*>(raw);
  auto* output = input + kRowsIn * kColsIn * kLd;
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int y = static_cast<int>(blockIdx.y) * kWarps;
  const int x = static_cast<int>(blockIdx.x) * kTW;
  load_nhwc<kK, 1, kRowsIn, kColsIn>(input, p, 0, y - 1, x - 1);
  cp_async_wait_all();
  __syncthreads();
  const auto base = smem_u32(input) + static_cast<std::uint32_t>(((lane & 15) * kLd + (lane >> 4) * 8) * 2);
#pragma unroll 1
  for (int phase = 0; phase < 4; ++phase) {
    const int z = phase * p.tiles;
    const auto* weights = p.w + static_cast<long>(z) * 4 * 2 * 2 * 32 + lane;
    Frag<Acc::kF16> acc[2][kHeadFragments];
#pragma unroll
    for (int frag = 0; frag < 2; ++frag)
#pragma unroll
      for (int n = 0; n < kHeadFragments; ++n) acc[frag][n].clear();
#pragma unroll 1
    for (int tap = 0; tap < 4; ++tap) {
      const int row = warp + (phase >> 1) + (tap >> 1);
      const int col = (phase & 1) + (tap & 1);
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const auto* at = weights + (tap * 2 + sub) * 64;
        const uint4 lo = __ldg(at);
        const uint2 hi = __ldg(reinterpret_cast<const uint2*>(at + 32));
#pragma unroll
        for (int frag = 0; frag < 2; ++frag) {
          std::uint32_t a[4];
          ldmatrix4(
              a, base + static_cast<std::uint32_t>(((row * kColsIn + col + 16 * frag) * kLd + sub * 16) * 2));
          acc[frag][0].mma(a, lo.x, lo.y);
          acc[frag][1].mma(a, lo.z, lo.w);
          acc[frag][2].mma(a, hi.x, hi.y);
        }
      }
    }
    if (phase != 0) __syncthreads();  // la fin précédente a fini de lire output
    stage_head(acc, output, p.bias + static_cast<long>(z) * kCT, warp, lane);
    __syncthreads();
    finish_tile<1, kHeadPitch>(output, p, f, y, x, phase);
  }
}
void finish_four_phases(dim3 grid, int, cudaStream_t stream, const Params& p, const FusedFinishParams& f) {
  constexpr int kHeight = kWarps + 2;
  constexpr int kBytes = (kHeight * (kTW + 2) * (kCT + 8) + kWarps * kTW * kHeadPitch) * 2;
  grid.z = 1;
  finish_four<<<grid, kThreads, kBytes, stream>>>(p, f);
}

}  // namespace

FinishLaunch pick_finish(RifeNativeSums sums, int kc, int rows, bool shared_phases) {
  if (shared_phases && sums == RifeNativeSums::kF16 && kc == 32 && rows == 1) return finish_four_phases;
  switch (sums) {
    case RifeNativeSums::kF16:
      if (rows == 1) return by_slice<1, Acc::kF16>(kc);
      return rows == 2 ? by_slice<2, Acc::kF16>(kc) : nullptr;
    case RifeNativeSums::kF32:
      return rows == 2 ? by_slice<2, Acc::kF32>(kc) : nullptr;
    case RifeNativeSums::kMixed:
      return nullptr;  // jamais une tête
  }
  return nullptr;
}

}  // namespace kirari::gpu::cuda::rife_native
