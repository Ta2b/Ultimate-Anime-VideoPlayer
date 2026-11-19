// La première convolution d'un bloc du RIFE natif, entrée calculée à la volée (rife_native_plan.h,
// pick_glue) : le dernier bloc en pleine résolution (D = 1), ou l'avant-dernier à moitié (D = 2, chaque point
// de X la moyenne de quatre points pleine résolution, comme fused_glue). En pleine résolution, X du dernier
// bloc pèse 64 octets par pixel (134 Mo en 1920 × 1088), 33 Mo à moitié : la colle l'écrivait, la
// convolution le relisait. Ici chaque bloc de fils relève les Y des blocs précédents
// (rife_glue.cuh, comme la colle), calcule les canaux de X de sa tuile d'entrée en mémoire partagée par les
// fonctions de la colle (mêmes arrondis), puis somme comme conv_kernel<16, CT, 3, 2, 1> : même tuile, même
// ordre des sommes. X tient en deux tranches de 16 canaux, calculées l'une après l'autre (la seconde refait
// le flot, peu coûteux, plutôt que de doubler la tuile) : 28 Ko de tuile et au plus 15 Ko de relèvement (des
// échelles d'au moins 2, que glue_layer exige), deux blocs par multiprocesseur ; un seul sur Turing (64 Ko),
// sauf pour le relèvement de 4.25 lite. En pleine résolution, seul le flot accumulé (canaux 24 à 27) part en
// mémoire, dans X, pour la fin (fused_finish) : chaque pixel par la tuile qui le possède. À moitié, rien : la
// colle du bloc suivant refait son flot depuis les Y.
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

// Les 16 canaux de la tranche Half de X en un point pleine résolution (x, y) : tranche 0, warps, instant,
// masque ; tranche 1, attributs, flot × 1/d, zéros de rembourrage. fl : le flot accumulé au point.
template <int K, int Half>
__device__ __forceinline__ void glue_point(const FusedGlueParams& g, const warp::Staged (&src)[K], int x,
                                           int y, float flow_scale, float (&v)[16], float (&fl)[4]) {
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
    for (int k = 0; k < 4; ++k) v[8 + k] = warp::rounded(fl[k] * flow_scale);
#pragma unroll
    for (int k = 12; k < 16; ++k) v[k] = 0.F;
  }
}

// La tranche Half de X sur la tuile d'entrée d'origine (y0, x0), au pas kLds, colonnes paires puis impaires
// (smem_col) ; zéros hors de X. D = 1 : le point pleine résolution lui-même ; la tranche 1 écrit aussi le
// flot des pixels propres de la tuile dans g.x (pas 32, canaux 24 à 27) : tous sauf ceux de la première ligne
// et de la première colonne, propres aux tuiles voisines. D = 2 : la moyenne des points (2x + tx, 2y + ty),
// sommés dans l'ordre de warp::sum_quad (fused_glue) : ((00 + 10) + (01 + 11)) / 4.
template <int K, int D, int Half>
__device__ __forceinline__ void glue_half(__half* s, const FusedGlueParams& g, const warp::Staged (&src)[K],
                                          int y0, int x0) {
  static_assert(D == 1 || D == 2, "pleine résolution ou moitié");
  const int xh = D == 1 ? g.height : g.out_height, xw = D == 1 ? g.width : g.out_width;
  for (int q = static_cast<int>(threadIdx.x); q < G::kIR * G::kIC; q += kThreads) {
    const int r = q / G::kIC, col = q - r * G::kIC;
    const int y = y0 + r, x = x0 + col;
    float v[16] = {};
    if (y >= 0 && y < xh && x >= 0 && x < xw) {
      float fl[4];
      if constexpr (D == 1) {
        glue_point<K, Half>(g, src, x, y, 1.F, v, fl);
        if (Half == 1 && r > 0 && col > 0) {
          const std::size_t pixel =
              static_cast<std::size_t>(y) * static_cast<std::size_t>(g.width) + static_cast<std::size_t>(x);
          *reinterpret_cast<uint2*>(static_cast<__half*>(g.x) + pixel * 32 + 24) =
              make_uint2(half_bits(v[8], v[9]), half_bits(v[10], v[11]));
        }
      } else {
        float top[16];
        float a[16];
        glue_point<K, Half>(g, src, 2 * x, 2 * y, 0.5F, top, fl);
        glue_point<K, Half>(g, src, 2 * x + 1, 2 * y, 0.5F, a, fl);
#pragma unroll
        for (int c = 0; c < 16; ++c) top[c] += a[c];
        glue_point<K, Half>(g, src, 2 * x, 2 * y + 1, 0.5F, v, fl);
        glue_point<K, Half>(g, src, 2 * x + 1, 2 * y + 1, 0.5F, a, fl);
#pragma unroll
        for (int c = 0; c < 16; ++c) v[c] = (top[c] + (v[c] + a[c])) * 0.25F;
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
template <int K, int D, int CT, Acc A>
__global__ void __launch_bounds__(kThreads, 2) glue_conv_kernel(Params p, FusedGlueParams g) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* s = reinterpret_cast<__half*>(smem_raw);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int ty0 = static_cast<int>(blockIdx.y) * G::kTH, tx0 = static_cast<int>(blockIdx.x) * kTW;
  const int y0 = 2 * ty0 - 1, x0 = 2 * tx0 - 1;

  // Le relèvement, après la tuile : la région de chaque Y que lisent les pixels de la tuile dans l'image.
  warp::Staged src[K];  // NOLINT(modernize-avoid-c-arrays)
  __half* relay = s + G::kIR * G::kIC * kLds;
  const int xh = D == 1 ? g.height : g.out_height, xw = D == 1 ? g.width : g.out_width;
  const warp::Span px{D * max(x0, 0), D * min(x0 + G::kIC - 1, xw - 1) + D - 1};
  const warp::Span py{D * max(y0, 0), D * min(y0 + G::kIR - 1, xh - 1) + D - 1};
#pragma unroll
  for (int j = 0; j < K; ++j) {
    src[j] = glue::stage_source(g.source[j], j == K - 1, px, py, relay);
    relay += (j == K - 1 ? glue::kLastStride : glue::kFlowStride) * glue_capacity(g.source[j].scale, D);
  }
  __syncthreads();

  Frag<A> acc[2][CT / 8];
#pragma unroll
  for (int f = 0; f < 2; ++f)
#pragma unroll
    for (int nt = 0; nt < CT / 8; ++nt) acc[f][nt].clear();
  // Les deux tranches, comme les deux tours de conv_kernel<16, …> : prises 0 à 8 de chacune.
  glue_half<K, D, 0>(s, g, src, y0, x0);
  __syncthreads();
  accumulate<16, CT, 3, 2, 1, A>(acc, s, p.w, 0, 2, 0, 9, warp, lane);
  __syncthreads();
  glue_half<K, D, 1>(s, g, src, y0, x0);
  __syncthreads();
  accumulate<16, CT, 3, 2, 1, A>(acc, s, p.w, 1, 2, 0, 9, warp, lane);
  __syncthreads();  // la tuile d'entrée n'est plus lue : la mémoire partagée reçoit la sortie
  stage<CT, 1, A>(acc, s, p.bias, warp, lane);
  __syncthreads();
  store<CT, 1, Out::kNhwc>(s, p, ty0, tx0, 0, 0);
}

template <int K, int D, int CT, Acc A>
void glue_one(dim3 grid, int smem, cudaStream_t stream, const Params& p, const FusedGlueParams& g) {
  glue_conv_kernel<K, D, CT, A><<<grid, kThreads, static_cast<std::size_t>(smem), stream>>>(p, g);
}

template <int K, int D, Acc A>
GlueLaunch by_tile(int ct) {
  if (ct == 16) return glue_one<K, D, 16, A>;
  if (ct == 32) return glue_one<K, D, 32, A>;
  return nullptr;
}

// Les découpages de RIFE : 4.25 (16, 8, 4, 2, 1) et 4.22 lite (8, 4, 2, 1). Le dernier bloc après trois ou
// quatre flots, l'avant-dernier à moitié après deux ou trois.
template <Acc A>
GlueLaunch by_flows(int flows, int reduce, int ct) {
  if (reduce == 1 && flows == 3) return by_tile<3, 1, A>(ct);
  if (reduce == 1 && flows == 4) return by_tile<4, 1, A>(ct);
  if (reduce == 2 && flows == 2) return by_tile<2, 2, A>(ct);
  if (reduce == 2 && flows == 3) return by_tile<3, 2, A>(ct);
  return nullptr;
}

}  // namespace

GlueLaunch pick_glue(RifeNativeSums sums, int flows, int reduce, int ct) {
  switch (sums) {
    case RifeNativeSums::kF16:
      return by_flows<Acc::kF16>(flows, reduce, ct);
    case RifeNativeSums::kMixed:
      return by_flows<Acc::kMixed>(flows, reduce, ct);
    case RifeNativeSums::kF32:
      return by_flows<Acc::kF32>(flows, reduce, ct);
  }
  return nullptr;
}

int glue_smem(std::span<const int> scales, int reduce) {
  int halves = G::kIR * G::kIC * kLds;
  for (std::size_t j = 0; j < scales.size(); ++j)
    halves +=
        (j + 1 == scales.size() ? glue::kLastStride : glue::kFlowStride) * glue_capacity(scales[j], reduce);
  return halves * 2;
}

}  // namespace kirari::gpu::cuda::rife_native
