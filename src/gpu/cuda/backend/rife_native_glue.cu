// La première convolution d'un bloc du RIFE natif, entrée calculée à la volée (rife_native_plan.h,
// pick_glue) : le dernier bloc en pleine résolution (D = 1), ou l'avant-dernier à moitié (D = 2, chaque point
// de X la moyenne de quatre points pleine résolution, comme fused_glue). En pleine résolution, X du dernier
// bloc pèse 64 octets par pixel (134 Mo en 1920 × 1088), 33 Mo à moitié : la colle l'écrivait, la
// convolution le relisait. Ici chaque bloc de fils relève les Y des blocs précédents
// (rife_glue.cuh, comme la colle), calcule les canaux de X de sa tuile d'entrée en mémoire partagée par les
// fonctions de la colle (mêmes arrondis), puis somme comme conv_kernel<16, CT, 3, 2, 1> : même tuile, même
// ordre des sommes. X tient en deux tranches de 16 canaux, calculées l'une après l'autre. Le rembourrage de
// chaque pixel garde le flot de la première tranche pour la seconde ; à moitié, il garde la moyenne arrondie.
// Il faut 28 Ko de tuile et au plus 15 Ko de relèvement (des
// échelles d'au moins 2, que glue_layer exige), deux blocs par multiprocesseur ; un seul sur Turing (64 Ko),
// sauf pour le relèvement de 4.25 lite. En pleine résolution, seul le flot accumulé (canaux 24 à 27) part en
// mémoire, groupé par pixel, pour la fin (fused_finish) : chaque pixel par la tuile qui le possède. À moitié,
// rien : la colle du bloc suivant refait son flot depuis les Y.
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
template <int K, int Half, bool Reuse, bool Packed, bool Fixed = false>
__device__ __forceinline__ void glue_point(const FusedGlueParams& g, const warp::Staged (&src)[K], int x,
                                           int y, float flow_scale, float (&v)[16], float (&fl)[4],
                                           const __half* cached = nullptr) {
  warp::Axis ax{};
  warp::Axis ay{};
  if constexpr (Half == 1 && Reuse) {
    const auto& f = g.source[K - 1];
    const float inv = Fixed ? 0.5F : 1.F / static_cast<float>(f.scale);
    ax = warp::coarse_axis(x, inv, 2 * f.width);
    ay = warp::coarse_axis(y, inv, 2 * f.height);
#pragma unroll
    for (int k = 0; k < 4; ++k) fl[k] = cached != nullptr ? __half2float(cached[k]) : 0.F;
  } else {
    glue::flow_at<K>(g, src, x, y, fl, ax, ay);
  }
  if constexpr (Half == 0) {
    float w[28];
    glue::warp_images<Packed>(g, x, y, fl, w);
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

__device__ __forceinline__ __half* flow_slot(const FusedGlueParams& params, std::size_t pixel) {
  const int first = params.x_pitch == kRifeCompactFlowPitch ? 0 : 24;
  return static_cast<__half*>(params.x) + pixel * params.x_pitch + first;
}

// À moitié, chaque pixel garde les mêmes quatre prises pleine résolution et le même ordre de somme.
template <int K, int Half, bool Reuse, bool Packed>
__device__ __forceinline__ void glue_half_resolution(__half* slot, const FusedGlueParams& g,
                                                     const warp::Staged (&src)[K], int x, int y,
                                                     float (&v)[16]) {
  float fl[4];
  float top[16];
  float a[16];
  float top_flow[4], bottom_flow[4];
  glue_point<K, Half, Reuse, Packed>(g, src, 2 * x, 2 * y, 0.5F, top, fl);
  if constexpr (Reuse && Half == 0) {
#pragma unroll
    for (int k = 0; k < 4; ++k) top_flow[k] = warp::rounded(fl[k] * 0.5F);
  }
  glue_point<K, Half, Reuse, Packed>(g, src, 2 * x + 1, 2 * y, 0.5F, a, fl);
  if constexpr (Reuse && Half == 0) {
#pragma unroll
    for (int k = 0; k < 4; ++k) top_flow[k] += warp::rounded(fl[k] * 0.5F);
  }
#pragma unroll
  for (int c = 0; c < 16; ++c) top[c] += a[c];
  glue_point<K, Half, Reuse, Packed>(g, src, 2 * x, 2 * y + 1, 0.5F, v, fl);
  if constexpr (Reuse && Half == 0) {
#pragma unroll
    for (int k = 0; k < 4; ++k) bottom_flow[k] = warp::rounded(fl[k] * 0.5F);
  }
  glue_point<K, Half, Reuse, Packed>(g, src, 2 * x + 1, 2 * y + 1, 0.5F, a, fl);
#pragma unroll
  for (int c = 0; c < 16; ++c) v[c] = (top[c] + (v[c] + a[c])) * 0.25F;
  if constexpr (Reuse && Half == 0) {
    // Comme Half=1 sans cache : FP16 par point, somme ((00 + 01) + (10 + 11)), puis stockage FP16.
#pragma unroll
    for (int k = 0; k < 4; ++k)
      fl[k] = (top_flow[k] + (bottom_flow[k] + warp::rounded(fl[k] * 0.5F))) * 0.25F;
    *reinterpret_cast<uint2*>(slot + 16) = make_uint2(half_bits(fl[0], fl[1]), half_bits(fl[2], fl[3]));
  } else if constexpr (Reuse) {
#pragma unroll
    for (int k = 0; k < 4; ++k) v[8 + k] = __half2float(slot[16 + k]);
  }
}

// La tranche Half de X sur la tuile d'entrée d'origine (y0, x0), au pas kLds, colonnes paires puis impaires
// (smem_col) ; zéros hors de X. D = 1 : le point pleine résolution lui-même ; la tranche 1 écrit aussi le
// flot des pixels propres de la tuile dans g.x : tous sauf ceux de la première ligne
// et de la première colonne, propres aux tuiles voisines. D = 2 : la moyenne des points (2x + tx, 2y + ty),
// sommés dans l'ordre de warp::sum_quad (fused_glue) : ((00 + 10) + (01 + 11)) / 4.
template <int K, int D, int Half, bool Reuse, bool Packed, class Geo = G, bool Fixed = false>
__device__ __forceinline__ void glue_half(__half* s, const FusedGlueParams& g, const warp::Staged (&src)[K],
                                          int y0, int x0) {
  static_assert(D == 1 || D == 2, "pleine résolution ou moitié");
  static_assert(!Reuse || K > 1);
  const int xh = D == 1 ? g.height : g.out_height, xw = D == 1 ? g.width : g.out_width;
  for (int q = static_cast<int>(threadIdx.x); q < Geo::kIR * Geo::kIC; q += kThreads) {
    const int r = q / Geo::kIC, col = q - r * Geo::kIC;
    const int y = y0 + r, x = x0 + col;
    __half* const slot = s + (r * Geo::kIC + smem_col<2, Geo::kIC>(col)) * kLds;
    float v[16] = {};
    if (y >= 0 && y < xh && x >= 0 && x < xw) {
      if constexpr (D == 1) {
        float fl[4];
        glue_point<K, Half, Reuse, Packed, Fixed>(g, src, x, y, 1.F, v, fl, slot + 16);
        if constexpr (Reuse && Half == 0) {
          // flow_at arrondit son dernier Add en FP16. Le padding est hors de la tranche lue par accumulate.
          *reinterpret_cast<uint2*>(slot + 16) = make_uint2(half_bits(fl[0], fl[1]), half_bits(fl[2], fl[3]));
        }
        if (Half == 1 && r > 0 && col > 0) {
          const std::size_t pixel =
              static_cast<std::size_t>(y) * static_cast<std::size_t>(g.width) + static_cast<std::size_t>(x);
          *reinterpret_cast<uint2*>(flow_slot(g, pixel)) =
              make_uint2(half_bits(v[8], v[9]), half_bits(v[10], v[11]));
        }
      } else {
        glue_half_resolution<K, Half, Reuse, Packed>(slot, g, src, x, y, v);
      }
    }
    auto* to = reinterpret_cast<uint4*>(slot);
    to[0] = make_uint4(half_bits(v[0], v[1]), half_bits(v[2], v[3]), half_bits(v[4], v[5]),
                       half_bits(v[6], v[7]));
    to[1] = make_uint4(half_bits(v[8], v[9]), half_bits(v[10], v[11]), half_bits(v[12], v[13]),
                       half_bits(v[14], v[15]));
  }
}

// À l'échelle 4, 8, etc. le changement de texel grossier tombe entre deux paires de
// points 2x et 2x+1. On charge les quatre mêmes texels une seule fois.
struct QuadAxes {
  warp::Axis left, right, above, below;
};

template <bool Fixed>
__device__ __forceinline__ QuadAxes d2_axes(const FusedFlowSource& source, int x, int y, int scale) {
  const float inv = 1.F / static_cast<float>(scale);
  if constexpr (!Fixed) {
    return {.left = warp::coarse_axis(2 * x, inv, 2 * source.width),
            .right = warp::coarse_axis(2 * x + 1, inv, 2 * source.width),
            .above = warp::coarse_axis(2 * y, inv, 2 * source.height),
            .below = warp::coarse_axis(2 * y + 1, inv, 2 * source.height)};
  } else {
    // Pour ces échelles dyadiques, le changement de texel tombe entre deux couples (2x, 2x + 1).
    const auto pair = [&](int at, int n, warp::Axis& a, warp::Axis& b) {
      const float raw = (static_cast<float>(2 * at) + 0.5F) * inv - 0.5F;
      const float limit = static_cast<float>(n - 1);
      const float lo = fminf(fmaxf(raw, 0.F), limit), hi = fminf(fmaxf(raw + inv, 0.F), limit);
      const int i0 = static_cast<int>(lo), i1 = min(i0 + 1, n - 1);
      a = {i0, i1, lo - static_cast<float>(i0)};
      b = {i0, i1, hi - static_cast<float>(i0)};
    };
    QuadAxes axes{};
    pair(x, 2 * source.width, axes.left, axes.right);
    pair(y, 2 * source.height, axes.above, axes.below);
    return axes;
  }
}

[[maybe_unused]] __device__ __forceinline__ float4 d2_sample(const warp::Staged& source, int c,
                                                             const QuadAxes& q) {
  const auto& axis_x = q.left;
  const auto& axis_y = q.above;
  const float a = glue::z_at(source, c, axis_y.i0, axis_x.i0),
              b = glue::z_at(source, c, axis_y.i0, axis_x.i1);
  const float d = glue::z_at(source, c, axis_y.i1, axis_x.i0),
              e = glue::z_at(source, c, axis_y.i1, axis_x.i1);
  return {warp::rounded(warp::lerp2(a, b, d, e, q.left, q.above)),
          warp::rounded(warp::lerp2(a, b, d, e, q.right, q.above)),
          warp::rounded(warp::lerp2(a, b, d, e, q.left, q.below)),
          warp::rounded(warp::lerp2(a, b, d, e, q.right, q.below))};
}

[[maybe_unused]] __device__ __forceinline__ float d2_average(float4 v) {
  return ((v.x + v.y) + (v.z + v.w)) * 0.25F;
}

// Chaque fil conserve ses quatre flots et le masque dans la case qu'il utilisera pour ses warps.
template <int K, bool Fixed, class Geo>
__device__ __forceinline__ void d2_stage_motion(__half* s, const FusedGlueParams& g,
                                                const warp::Staged (&src)[K], int x0, int y0) {
  for (int q = static_cast<int>(threadIdx.x); q < Geo::kIR * Geo::kIC; q += kThreads) {
    const int r = q / Geo::kIC, c = q % Geo::kIC, x = x0 + c, y = y0 + r;
    if (x < 0 || y < 0 || x >= g.out_width || y >= g.out_height) continue;
    auto* slot = s + (r * Geo::kIC + smem_col<2, Geo::kIC>(c)) * kLds;
    const auto last = d2_axes<Fixed>(g.source[K - 1], x, y, Fixed ? 4 : g.source[K - 1].scale);
    float4 flow[4];
#pragma unroll
    for (int i = 0; i < K; ++i) {
      int source_scale = g.source[i].scale;
      if constexpr (Fixed) source_scale = glue::last_scales<K + 1>(i);
      const auto axes = i == K - 1 ? last : d2_axes<Fixed>(g.source[i], x, y, source_scale);
      const auto scale = static_cast<float>(source_scale);
#pragma unroll
      for (int ch = 0; ch < 4; ++ch) {
        const float4 a = d2_sample(src[i], ch, axes);
        if (i == 0)
          flow[ch] = {a.x * scale, a.y * scale, a.z * scale, a.w * scale};
        else
          flow[ch] = {warp::rounded(flow[ch].x + a.x * scale), warp::rounded(flow[ch].y + a.y * scale),
                      warp::rounded(flow[ch].z + a.z * scale), warp::rounded(flow[ch].w + a.w * scale)};
      }
    }
    float avg[4];
#pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
      const float4 z = flow[ch];
      *reinterpret_cast<uint2*>(slot + 4 * ch) = make_uint2(half_bits(z.x, z.y), half_bits(z.z, z.w));
      avg[ch] = d2_average({warp::rounded(z.x * .5F), warp::rounded(z.y * .5F), warp::rounded(z.z * .5F),
                            warp::rounded(z.w * .5F)});
    }
    *reinterpret_cast<uint2*>(slot + 16) = make_uint2(half_bits(avg[0], avg[1]), half_bits(avg[2], avg[3]));
    slot[20] = __float2half_rn(d2_average(d2_sample(src[K - 1], 4, last)));
  }
}

template <bool Packed>
__device__ __forceinline__ void d2_warp_point(float (&v)[28], const FusedGlueParams& g, const __half* slot,
                                              int x, int y, int sample) {
  float fl[4];
#pragma unroll
  for (int ch = 0; ch < 4; ++ch) fl[ch] = __half2float(slot[4 * ch + sample]);
  glue::warp_images<Packed>(g, x, y, fl, v);
}

template <bool Packed, class Geo>
__device__ __forceinline__ void d2_warp_staged(__half* s, const FusedGlueParams& g, int x0, int y0) {
  for (int q = static_cast<int>(threadIdx.x); q < Geo::kIR * Geo::kIC; q += kThreads) {
    const int r = q / Geo::kIC, c = q % Geo::kIC, x = x0 + c, y = y0 + r;
    auto* slot = s + (r * Geo::kIC + smem_col<2, Geo::kIC>(c)) * kLds;
    float v[16]{};
    if (x >= 0 && y >= 0 && x < g.out_width && y < g.out_height) {
      float top[28], other[28], bottom[28];
      d2_warp_point<Packed>(top, g, slot, 2 * x, 2 * y, 0);
      d2_warp_point<Packed>(other, g, slot, 2 * x + 1, 2 * y, 1);
#pragma unroll
      for (int ch = 0; ch < 14; ++ch) top[ch] += other[ch];
      d2_warp_point<Packed>(bottom, g, slot, 2 * x, 2 * y + 1, 2);
      d2_warp_point<Packed>(other, g, slot, 2 * x + 1, 2 * y + 1, 3);
#pragma unroll
      for (int ch = 0; ch < 14; ++ch) v[ch] = (top[ch] + (bottom[ch] + other[ch])) * 0.25F;
      const float t = glue::instant(g);
      v[14] = ((t + t) + (t + t)) * 0.25F;
      v[15] = __half2float(slot[20]);
    }
    auto* to = reinterpret_cast<uint4*>(slot);
    to[0] = make_uint4(half_bits(v[0], v[1]), half_bits(v[2], v[3]), half_bits(v[4], v[5]),
                       half_bits(v[6], v[7]));
    to[1] = make_uint4(half_bits(v[8], v[9]), half_bits(v[10], v[11]), half_bits(v[12], v[13]),
                       half_bits(v[14], v[15]));
  }
}

// Les attributs et les flots moyens forment la seconde tranche du noyau D2.
template <int K, bool Fixed, class Geo>
__device__ __forceinline__ void d2_attributes(__half* s, const FusedGlueParams& g,
                                              const warp::Staged (&src)[K], int x0, int y0) {
  for (int q = static_cast<int>(threadIdx.x); q < Geo::kIR * Geo::kIC; q += kThreads) {
    const int r = q / Geo::kIC, c = q % Geo::kIC, x = x0 + c, y = y0 + r;
    auto* slot = s + (r * Geo::kIC + smem_col<2, Geo::kIC>(c)) * kLds;
    uint4 attributes{};
    uint2 flow{};
    if (x >= 0 && y >= 0 && x < g.out_width && y < g.out_height) {
      const auto last = d2_axes<Fixed>(g.source[K - 1], x, y, Fixed ? 4 : g.source[K - 1].scale);
      float values[8];
#pragma unroll
      for (int ch = 0; ch < 8; ++ch) values[ch] = d2_average(d2_sample(src[K - 1], 5 + ch, last));
      attributes = make_uint4(half_bits(values[0], values[1]), half_bits(values[2], values[3]),
                              half_bits(values[4], values[5]), half_bits(values[6], values[7]));
      flow = *reinterpret_cast<const uint2*>(slot + 16);
    }
    auto* to = reinterpret_cast<uint4*>(slot);
    to[0] = attributes;
    to[1] = make_uint4(flow.x, flow.y, 0, 0);
  }
}

// La couche : 32 canaux d'entrée (28 de X et 4 nuls) vers CT sorties, en une tuile (blockIdx.z = 0).
template <int K, int D, int CT, Acc A, bool Reuse, bool Packed, bool D2Taps = false>
__global__ void __launch_bounds__(kThreads, 2) glue_conv_kernel(Params p, FusedGlueParams g) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
  constexpr bool kTaps = D2Taps;
#else
  constexpr bool kTaps = false;
#endif
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
    const int scale = g.source[j].scale;
    src[j] = glue::stage_source(g.source[j], j == K - 1, px, py, relay);
    relay += (j == K - 1 ? glue::kLastStride : glue::kFlowStride) * glue_capacity(scale, D);
  }
  __syncthreads();

  Frag<A> acc[2][CT / 8];
#pragma unroll
  for (int f = 0; f < 2; ++f)
#pragma unroll
    for (int nt = 0; nt < CT / 8; ++nt) acc[f][nt].clear();
  // Les deux tranches, comme les deux tours de conv_kernel<16, …> : prises 0 à 8 de chacune.
  glue_half<K, D, 0, Reuse, Packed>(s, g, src, y0, x0);
  __syncthreads();
  accumulate<16, CT, 3, 2, 1, A>(acc, s, p.w, 0, 2, 0, 9, warp, lane);
  __syncthreads();
  if constexpr (kTaps)
    d2_attributes<K, false, G>(s, g, src, x0, y0);
  else
    glue_half<K, D, 1, Reuse, Packed>(s, g, src, y0, x0);
  __syncthreads();
  accumulate<16, CT, 3, 2, 1, A>(acc, s, p.w, 1, 2, 0, 9, warp, lane);
  __syncthreads();  // la tuile d'entrée n'est plus lue : la mémoire partagée reçoit la sortie
  stage<CT, 1, A>(acc, s, p.bias, warp, lane);
  __syncthreads();
  store<CT, 1, Out::kNhwc>(s, p, ty0, tx0, 0, 0);
}

// Les flots et le masque prennent cinq valeurs du rembourrage. Le warp repart de leurs valeurs
// déjà arrondies en FP16 et la seconde tranche garde le même cache de flot qu'avec la tuile large.
template <int K, class Geo, bool Fixed>
__device__ __forceinline__ void stage_motion(__half* s, const FusedGlueParams& g,
                                             const warp::Staged (&src)[K], int x0, int y0) {
  for (int q = static_cast<int>(threadIdx.x); q < Geo::kIR * Geo::kIC; q += kThreads) {
    const int row = q / Geo::kIC, col = q % Geo::kIC;
    const int y = y0 + row, x = x0 + col;
    const int h = g.height, w = g.width;
    if (x < 0 || x >= w || y < 0 || y >= h) continue;
    warp::Axis ax{}, ay{};
    float f[4];
    glue::flow_at<K, Fixed>(g, src, x, y, f, ax, ay);
    auto* slot = s + (row * Geo::kIC + smem_col<2, Geo::kIC>(col)) * kLds;
    const auto pack = make_uint2(half_bits(f[0], f[1]), half_bits(f[2], f[3]));
    *reinterpret_cast<uint2*>(slot + 16) = pack;
    slot[20] = __float2half_rn(glue::z_enlarge(src[K - 1], 4, ax, ay));
  }
}

template <class Geo, bool Packed>
__device__ __forceinline__ void warp_staged_tile(__half* s, const FusedGlueParams& g, int x0, int y0) {
  for (int q = static_cast<int>(threadIdx.x); q < Geo::kIR * Geo::kIC; q += kThreads) {
    const int r = q / Geo::kIC, c = q % Geo::kIC;
    const int x = x0 + c, y = y0 + r;
    auto* slot = s + (r * Geo::kIC + smem_col<2, Geo::kIC>(c)) * kLds;
    float v[28]{};
    if (x >= 0 && x < g.width && y >= 0 && y < g.height) {
      const auto pack = *reinterpret_cast<const uint2*>(slot + 16);
      const float2 a = half_pair(pack.x), b = half_pair(pack.y);
      const float flow[4]{a.x, a.y, b.x, b.y};
      glue::warp_images<Packed>(g, x, y, flow, v);
      v[14] = glue::instant(g);
      v[15] = __half2float(slot[20]);
    }
    auto* to = reinterpret_cast<uint4*>(slot);
    to[0] = make_uint4(half_bits(v[0], v[1]), half_bits(v[2], v[3]), half_bits(v[4], v[5]),
                       half_bits(v[6], v[7]));
    to[1] = make_uint4(half_bits(v[8], v[9]), half_bits(v[10], v[11]), half_bits(v[12], v[13]),
                       half_bits(v[14], v[15]));
  }
}

struct Narrow {
  static constexpr int kTH = kWarps, kWidth = kStagedGlueWidth, kIR = (kTH - 1) * 2 + 3,
                       kIC = (kWidth - 1) * 2 + 3;
  static constexpr int kEven = (kIC + 1) / 2;
};

// La tuile 16 × 4 conserve un seul fragment M16 par warp. Avec les calculs du warp et du flot dans
// deux étapes, leur consommation de registres n'est plus simultanée.
template <int K, bool Packed, bool Fixed = false, int D = 1>
__global__ void __launch_bounds__(kThreads, 2) glue_staged(Params p, FusedGlueParams g) {
  using N = Narrow;
  static_assert(D == 1 || D == 2);
  constexpr int CT = 16 * D, Rows = 1;
  constexpr int kNP = CT / 16, kO = CT + 8;
  extern __shared__ __align__(16) unsigned char raw[];
  auto* s = reinterpret_cast<__half*>(raw);
  auto* relay = s + N::kIR * N::kIC * kLds;
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int yo = static_cast<int>(blockIdx.y) * N::kTH, xo = static_cast<int>(blockIdx.x) * N::kWidth;
  const int y0 = 2 * yo - 1, x0 = 2 * xo - 1;
  const int xh = D == 1 ? g.height : g.out_height, xw = D == 1 ? g.width : g.out_width;
  const warp::Span px{D * max(x0, 0), D * min(x0 + N::kIC - 1, xw - 1) + D - 1};
  const warp::Span py{D * max(y0, 0), D * min(y0 + N::kIR - 1, xh - 1) + D - 1};
  warp::Staged src[K];
#pragma unroll
  for (int i = 0; i < K; ++i) {
    const auto& f = g.source[i];
    int scale = f.scale;
    if constexpr (Fixed) scale = glue::last_scales<K + D - 1>(i);
    src[i] = glue::stage_source(f, i == K - 1, px, py, relay, Fixed ? scale : 0);
    relay += (i == K - 1 ? glue::kLastStride : glue::kFlowStride) * glue_span(D * N::kIC - 1, scale) *
             glue_span(D * N::kIR - 1, scale);
  }
  __syncthreads();
  Frag<Acc::kF16> a[Rows][CT / 8];
#pragma unroll
  for (int row = 0; row < Rows; ++row)
#pragma unroll
    for (int c = 0; c < CT / 8; ++c) a[row][c].clear();
  const auto conv = [&](int sub) {
    const auto base = smem_u32(s) + static_cast<std::uint32_t>(((lane & 15) * kLds + (lane / 16) * 8) * 2);
#pragma unroll 1
    for (int tap = 0; tap < 9; ++tap) {
      const int dy = tap / 3, dx = tap % 3;
      const int col = ((dx & 1) == 0 ? 0 : N::kEven) + (dx / 2);
      uint4 weights[kNP];
#pragma unroll
      for (int n = 0; n < kNP; ++n) weights[n] = __ldg(p.w + ((tap * 2 + sub) * kNP + n) * 32 + lane);
#pragma unroll
      for (int row = 0; row < Rows; ++row) {
        const int y = (warp * Rows + row) * 2 + dy;
        std::uint32_t v[4];
        ldmatrix4(v, base + static_cast<std::uint32_t>((y * N::kIC + col) * kLds * 2));
#pragma unroll
        for (int n = 0; n < kNP; ++n) {
          a[row][2 * n].mma(v, weights[n].x, weights[n].y);
          a[row][2 * n + 1].mma(v, weights[n].z, weights[n].w);
        }
      }
    }
  };
  // Les deux boucles donnent les mêmes cases aux mêmes fils : chacun relit son propre flot.
  if constexpr (D == 1) {
    stage_motion<K, N, Fixed>(s, g, src, x0, y0);
    warp_staged_tile<N, Packed>(s, g, x0, y0);
  } else {
    d2_stage_motion<K, Fixed, N>(s, g, src, x0, y0);
    d2_warp_staged<Packed, N>(s, g, x0, y0);
  }
  __syncthreads();
  conv(0);
  __syncthreads();
  if constexpr (D == 1)
    glue_half<K, 1, 1, true, Packed, N, Fixed>(s, g, src, y0, x0);
  else
    d2_attributes<K, Fixed, N>(s, g, src, x0, y0);
  __syncthreads();
  conv(1);
  __syncthreads();
  const int t = lane & 3, g8 = lane / 4;
#pragma unroll
  for (int row = 0; row < Rows; ++row)
#pragma unroll
    for (int n = 0; n < CT / 8; ++n) {
      const int ch = n * 8 + 2 * t;
      const auto b = __ldg(reinterpret_cast<const float2*>(p.bias + ch));
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const auto v = a[row][n].pair(i);
        const int pixel = (warp * Rows + row) * N::kWidth + g8 + 8 * i;
        *reinterpret_cast<__half2*>(s + pixel * kO + ch) = to_half2(v.x + b.x, v.y + b.y);
      }
    }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.x); i < N::kTH * N::kWidth * (CT / 8); i += kThreads) {
    const int q = i / (CT / 8), ch = i % (CT / 8) * 8;
    const int y = yo + q / N::kWidth, x = xo + q % N::kWidth;
    if (y >= p.oh || x >= p.ow) continue;
    const auto from = *reinterpret_cast<const uint4*>(s + q * kO + ch);
    const long dst = static_cast<long>(y) * p.ow + x;
    *reinterpret_cast<uint4*>(p.out + dst * p.out_ld + ch) = finish8(from, p, dst, ch);
  }
}

template <int K, bool Fixed = false>
void staged_one(dim3 grid, int smem, cudaStream_t stream, const Params& p, const FusedGlueParams& g) {
  if (g.encode_hwc4)
    glue_staged<K, true, Fixed><<<grid, kThreads, static_cast<std::size_t>(smem), stream>>>(p, g);
  else
    glue_staged<K, false><<<grid, kThreads, static_cast<std::size_t>(smem), stream>>>(p, g);
}

template <int K, int D, int CT, Acc A>
void glue_one(dim3 grid, int smem, cudaStream_t stream, const Params& p, const FusedGlueParams& g) {
  if (g.encode_hwc4)
    glue_conv_kernel<K, D, CT, A, true, true>
        <<<grid, kThreads, static_cast<std::size_t>(smem), stream>>>(p, g);
  else
    glue_conv_kernel<K, D, CT, A, true, false>
        <<<grid, kThreads, static_cast<std::size_t>(smem), stream>>>(p, g);
}

template <int K, int D, Acc A>
GlueLaunch by_tile(int ct) {
  if (ct == 16) return glue_one<K, D, 16, A>;
  if (ct == 32) return glue_one<K, D, 32, A>;
  return nullptr;
}

// Les découpages de RIFE : 4.25 (8, 8, 4, 2, 1) et 4.22 lite (8, 4, 2, 1). Le dernier bloc après trois ou
// quatre flots, l'avant-dernier à moitié après deux ou trois.
template <int K, bool Fixed>
void d2_taps_one(dim3 grid, int smem, cudaStream_t stream, const Params& p, const FusedGlueParams& g) {
  if (g.encode_hwc4) {
    int halves = Narrow::kIR * Narrow::kIC * kLds;
    for (int j = 0; j < K; ++j) {
      const int scale = g.source[j].scale;
      halves += (j + 1 == K ? glue::kLastStride : glue::kFlowStride) * glue_span(2 * Narrow::kIC - 1, scale) *
                glue_span(2 * Narrow::kIR - 1, scale);
    }
    grid.x = static_cast<unsigned>((p.ow + Narrow::kWidth - 1) / Narrow::kWidth);
    glue_staged<K, true, Fixed, 2><<<grid, kThreads, static_cast<std::size_t>(halves) * 2, stream>>>(p, g);
  } else {
    // En plans, la tuile large garde l'évaluation d'origine du flot.
    glue_conv_kernel<K, 2, 32, Acc::kF16, true, false, true>
        <<<grid, kThreads, static_cast<std::size_t>(smem), stream>>>(p, g);
  }
}

template <int K>
bool match_scales(std::span<const int> scales) {
  for (std::size_t i = 0; i < scales.size(); ++i)
    if (scales[i] != glue::last_scales<K>(static_cast<int>(i))) return false;
  return true;
}

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

GlueLaunch pick_d2_taps(std::span<const int> scales) {
  if (scales.size() == 2) return match_scales<3>(scales) ? d2_taps_one<2, true> : d2_taps_one<2, false>;
  if (scales.size() == 3) return match_scales<4>(scales) ? d2_taps_one<3, true> : d2_taps_one<3, false>;
  return nullptr;
}

template <int K>
GlueLaunch staged_scales(std::span<const int> scales) {
  return match_scales<K>(scales) ? staged_one<K, true> : staged_one<K>;
}

GlueLaunch pick_staged_glue(std::span<const int> scales) {
  if (scales.size() == 3) return staged_scales<3>(scales);
  if (scales.size() == 4) return staged_scales<4>(scales);
  return nullptr;
}

int staged_glue_smem(std::span<const int> scales) {
  int halves = Narrow::kIR * Narrow::kIC * kLds;
  for (std::size_t j = 0; j < scales.size(); ++j)
    halves += (j + 1 == scales.size() ? glue::kLastStride : glue::kFlowStride) *
              glue_span(Narrow::kIC - 1, scales[j]) * glue_span(Narrow::kIR - 1, scales[j]);
  return halves * 2;
}

int glue_smem(std::span<const int> scales, int reduce) {
  int halves = G::kIR * G::kIC * kLds;
  for (std::size_t j = 0; j < scales.size(); ++j)
    halves +=
        (j + 1 == scales.size() ? glue::kLastStride : glue::kFlowStride) * glue_capacity(scales[j], reduce);
  return halves * 2;
}

}  // namespace kirari::gpu::cuda::rife_native
