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

constexpr int kCT = 32;  // une tuile : les canaux 0 à 19 de Y y sont, 4c + parité pour c < 5

// La fin sur la tuile de sortie (o : la mémoire partagée de stage, pas kCT + 8). Le pixel (y, x) de la
// grille, à la phase (py, px), est le pixel (2y + py, 2x + px) de Y ; son canal 4c + 2a + b est le canal c de
// Z au point pleine résolution (2(2y + py) + a, 2(2x + px) + b). Un fil par point ; deux fils voisins font
// deux points voisins d'une ligne.
template <int Rows>
__device__ __forceinline__ void finish_tile(const __half* o, const Params& p, const FusedFinishParams& f,
                                            int ty0, int tx0, int phase) {
  constexpr int kTH = kWarps * Rows, kLdo = kCT + 8;
  const int py = phase >> 1, px = phase & 1;
  const std::size_t plane = static_cast<std::size_t>(f.height) * static_cast<std::size_t>(f.width);
  for (int i = static_cast<int>(threadIdx.x); i < kTH * kTW * 4; i += kThreads) {
    const int b = i & 1, col = (i >> 1) % kTW, rest = (i >> 1) / kTW;
    const int a = rest & 1, row = rest >> 1;
    const int gy = ty0 + row, gx = tx0 + col;
    if (gy >= p.oh || gx >= p.ow) continue;
    const int x = 2 * (2 * gx + px) + b, y = 2 * (2 * gy + py) + a;
    const __half* v = o + (row * kTW + col) * kLdo + 2 * a + b;
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

}  // namespace

FinishLaunch pick_finish(RifeNativeSums sums, int kc, int rows) {
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
