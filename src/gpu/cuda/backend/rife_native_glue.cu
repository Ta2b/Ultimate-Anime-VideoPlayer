// La première convolution du dernier bloc du RIFE natif, entrée calculée à la volée (rife_native_plan.h,
// pick_glue). En pleine résolution, X du dernier bloc pèse 64 octets par pixel (134 Mo en 1920 × 1088) : la
// colle l'écrivait, la convolution le relisait. Ici chaque bloc de fils relève les Y des blocs précédents
// (rife_glue.cuh, comme la colle), calcule les canaux de X de sa tuile d'entrée en mémoire partagée par les
// fonctions de la colle (mêmes arrondis), puis somme comme conv_kernel<16, CT, 3, 2, 1> : même tuile, même
// ordre des sommes. X tient en deux tranches de 16 canaux, calculées l'une après l'autre (la seconde refait
// le flot, peu coûteux, plutôt que de doubler la tuile) : 28 Ko de tuile et moins de 15 Ko de relèvement,
// deux blocs par multiprocesseur. Seul le flot accumulé (canaux 24 à 27) part en mémoire, dans X, pour la fin
// (fused_finish) : chaque pixel par la tuile qui le possède.
#include <cstddef>
#include <cstdint>
#include <span>

#include "gpu/cuda/backend/rife_glue.cuh"
#include "gpu/cuda/backend/rife_native_plan.h"

namespace kirari::gpu::cuda::rife_native {
namespace {

using G = Geometry<3, 2, 1>;  // pas 2, une ligne par warp : 65 × 9 pixels d'entrée pour 32 × 4 sorties
constexpr int kLds = 16 + 8;  // une tranche de 16 canaux

__device__ __forceinline__ std::uint32_t half_bits(float lo, float hi) {
  const __half2 pair = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const std::uint32_t*>(&pair);
}

// La tranche Half de X (0 : canaux 0 à 15, 1 : canaux 16 à 31) sur la tuile d'entrée d'origine (y0, x0), au
// pas kLds, colonnes paires puis impaires (smem_col) ; zéros hors de l'image. Tranche 0 : warps, instant,
// masque ; tranche 1 : attributs, flot × 1/d (d = 1), zéros de rembourrage. La tranche 1 écrit aussi le flot
// des pixels propres de la tuile dans g.x (pas 32, canaux 24 à 27) : tous sauf ceux de la première ligne et
// de la première colonne, propres aux tuiles voisines.
template <int K, int Half>
__device__ __forceinline__ void glue_half(__half* s, const FusedGlueParams& g, const warp::Staged (&src)[K],
                                          int y0, int x0) {
  for (int q = static_cast<int>(threadIdx.x); q < G::kIR * G::kIC; q += kThreads) {
    const int r = q / G::kIC, col = q - r * G::kIC;
    const int y = y0 + r, x = x0 + col;
    float v[16] = {};
    if (y >= 0 && y < g.height && x >= 0 && x < g.width) {
      float fl[4];
      warp::Axis ax{};
      warp::Axis ay{};
      glue::flow_at<K>(g, src, x, y, fl, ax, ay);
      if constexpr (Half == 0) {
        float w[28];
        glue::warped(g, x, y, fl, w);
#pragma unroll
        for (int c = 0; c < 14; ++c) v[c] = w[c];
        v[14] = glue::instant(g);
        v[15] = glue::z_enlarge(src[K - 1], 4, ax, ay);
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = glue::z_enlarge(src[K - 1], 5 + k, ax, ay);
#pragma unroll
        for (int k = 0; k < 4; ++k) v[8 + k] = warp::rounded(fl[k]);
        if (r > 0 && col > 0) {
          const std::size_t pixel =
              static_cast<std::size_t>(y) * static_cast<std::size_t>(g.width) + static_cast<std::size_t>(x);
          *reinterpret_cast<uint2*>(static_cast<__half*>(g.x) + pixel * 32 + 24) =
              make_uint2(half_bits(v[8], v[9]), half_bits(v[10], v[11]));
        }
      }
    }
    auto* to = reinterpret_cast<uint4*>(s + (r * G::kIC + smem_col<2, G::kIC>(col)) * kLds);
    to[0] = make_uint4(half_bits(v[0], v[1]), half_bits(v[2], v[3]), half_bits(v[4], v[5]),
                       half_bits(v[6], v[7]));
    to[1] = make_uint4(half_bits(v[8], v[9]), half_bits(v[10], v[11]), half_bits(v[12], v[13]),
                       half_bits(v[14], v[15]));
  }
}

// La couche : 32 canaux d'entrée (28 de X et 4 nuls) vers CT sorties, en une tuile (blockIdx.z = 0).
template <int K, int CT, Acc A>
__global__ void __launch_bounds__(kThreads, 2) glue_conv_kernel(Params p, FusedGlueParams g) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* s = reinterpret_cast<__half*>(smem_raw);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int ty0 = static_cast<int>(blockIdx.y) * G::kTH, tx0 = static_cast<int>(blockIdx.x) * kTW;
  const int y0 = 2 * ty0 - 1, x0 = 2 * tx0 - 1;

  // Le relèvement, après la tuile : la région de chaque Y que lisent les pixels de la tuile dans l'image.
  warp::Staged src[K];  // NOLINT(modernize-avoid-c-arrays)
  __half* relay = s + G::kIR * G::kIC * kLds;
  const warp::Span px{max(x0, 0), min(x0 + G::kIC - 1, g.width - 1)};
  const warp::Span py{max(y0, 0), min(y0 + G::kIR - 1, g.height - 1)};
#pragma unroll
  for (int j = 0; j < K; ++j) {
    src[j] = glue::stage_source(g.source[j], j == K - 1, px, py, relay);
    relay += (j == K - 1 ? glue::kLastStride : glue::kFlowStride) * glue_capacity(g.source[j].scale);
  }
  __syncthreads();

  Frag<A> acc[2][CT / 8];
#pragma unroll
  for (int f = 0; f < 2; ++f)
#pragma unroll
    for (int nt = 0; nt < CT / 8; ++nt) acc[f][nt].clear();
  // Les deux tranches, comme les deux tours de conv_kernel<16, …> : prises 0 à 8 de chacune.
  glue_half<K, 0>(s, g, src, y0, x0);
  __syncthreads();
  accumulate<16, CT, 3, 2, 1, A>(acc, s, p.w, 0, 2, 0, 9, warp, lane);
  __syncthreads();
  glue_half<K, 1>(s, g, src, y0, x0);
  __syncthreads();
  accumulate<16, CT, 3, 2, 1, A>(acc, s, p.w, 1, 2, 0, 9, warp, lane);
  __syncthreads();  // la tuile d'entrée n'est plus lue : la mémoire partagée reçoit la sortie
  stage<CT, 1, A>(acc, s, p.bias, warp, lane);
  __syncthreads();
  store<CT, 1, Out::kNhwc>(s, p, ty0, tx0, 0, 0);
}

template <int K, int CT, Acc A>
void glue_one(dim3 grid, int smem, cudaStream_t stream, const Params& p, const FusedGlueParams& g) {
  glue_conv_kernel<K, CT, A><<<grid, kThreads, static_cast<std::size_t>(smem), stream>>>(p, g);
}

template <int K, Acc A>
GlueLaunch by_tile(int ct) {
  if (ct == 16) return glue_one<K, 16, A>;
  if (ct == 32) return glue_one<K, 32, A>;
  return nullptr;
}

template <Acc A>
GlueLaunch by_flows(int flows, int ct) {
  if (flows == 3) return by_tile<3, A>(ct);
  if (flows == 4) return by_tile<4, A>(ct);
  return nullptr;
}

}  // namespace

GlueLaunch pick_glue(RifeNativeSums sums, int flows, int ct) {
  switch (sums) {
    case RifeNativeSums::kF16:
      return by_flows<Acc::kF16>(flows, ct);
    case RifeNativeSums::kMixed:
      return by_flows<Acc::kMixed>(flows, ct);
    case RifeNativeSums::kF32:
      return by_flows<Acc::kF32>(flows, ct);
  }
  return nullptr;
}

int glue_smem(std::span<const int> scales) {
  int halves = G::kIR * G::kIC * kLds;
  for (std::size_t j = 0; j < scales.size(); ++j)
    halves += (j + 1 == scales.size() ? glue::kLastStride : glue::kFlowStride) * glue_capacity(scales[j]);
  return halves * 2;
}

}  // namespace kirari::gpu::cuda::rife_native
