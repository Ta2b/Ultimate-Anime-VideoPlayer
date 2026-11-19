#pragma once
// Noyaux du squelette CNN de GMFlow natif (docs/gmflow-natif.md), sur la famille de rife_native_kernels.cuh :
// mêmes tuiles (4 warps, Rows lignes par warp sur 32 colonnes), mêmes tranches de Kc canaux d'entrée, mêmes
// fragments et poids rangés, mêmes sommes du mma (accumulate) et même arrondi du biais (stage).
//
// La 7 × 7 de pas 2 lit les deux images de trois plans FP16 telles quelles : la normalisation d'ImageNet est
// dans ses poids et son biais (prepare/gmflow_backbone.h), un pixel hors de l'image vaut pad. Le morceau
// d'image est déplié en mémoire partagée en 160 valeurs par pixel de sortie, k = (dy · 7 + dx) · 3 + plan,
// zéros au-delà de 147 : une 1 × 1 de 160 canaux pour accumulate.
//
// Les 1 × 1 de pas 2 chargent un pixel sur deux dans la tuile d'une 1 × 1 de pas 1.
//
// InstanceNorm sans paramètres : l'épilogue ajoute Σx et Σx² de ce qu'il écrit (les valeurs FP16, après le
// biais) par image et par canal, en f32, une addition atomique par bloc et par canal. La convolution suivante
// normalise et applique la Relu à la tuile qu'elle charge (kNorm), en f32 puis arrondi FP16, comme le PReLU à
// la lecture de RIFE ; le bord hors de l'image reste nul.
//
// La fin de bloc résiduel se fait élément par élément : relu(raccourci + bloc), chacun normalisé s'il le
// faut, une seule écriture FP16.
//
// Le lot de deux images : blockIdx.z = image · tiles + tuile, les tenseurs d'une image à la suite de l'autre.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::gmflow_backbone {

namespace rn = rife_native;
using rn::Acc;
using rn::kThreads;
using rn::kTW;
using rn::kWarps;

// La lecture d'une convolution : NHWC tel quel (par cp.async), ou NHWC normalisé puis Relu (kNorm).
enum class In : int { kPlain, kNorm };

inline constexpr int kMaxChannels = 128;
inline constexpr int kStemK = 160;         // 7 · 7 · 3 = 147 produits, par 16
inline constexpr int kStemRows = 1;        // tuile de 4 × 32 pixels de sortie
inline constexpr int kStemPatchRows = 13;  // (4 - 1) · 2 + 7 lignes d'image
inline constexpr int kStemPatchCols = 69;  // (32 - 1) · 2 + 7 colonnes
inline constexpr int kStemLds = kStemK + 8;
inline constexpr int kReduceBytes = kThreads * 4 * 4;  // Σ et Σ² de deux canaux par fil

struct Params {
  const __half* in = nullptr;  // NHWC à in_ld canaux ; la 7 × 7 : les trois plans de chaque image
  int in_ld = 0;
  int ih = 0, iw = 0;
  const float* in_stats = nullptr;  // kNorm : [image][Σx, Σx²][in_ld]
  __half* out = nullptr;            // NHWC à out_ld canaux
  int out_ld = 0;
  int oh = 0, ow = 0;
  float* out_stats = nullptr;  // non nul : Σx et Σx² de la sortie ajoutés, [image][Σx, Σx²][out_ld]
  const uint4* w = nullptr;    // poids rangés, un paquet par tuile de sortie
  const float* bias = nullptr;
  int cin = 0;    // multiple de Kc (la 7 × 7 : kStemK)
  int tiles = 1;  // tuiles de CT canaux de sortie
  float eps = 1e-5F;
  float pad[3] = {0.F, 0.F, 0.F};  // la 7 × 7 : la valeur d'un pixel hors de l'image, par plan (FP16 exact)
};

// Moyenne et 1/√(variance + ε) du canal c, depuis les sommes de son image (sum : Σx, sum + ld : Σx²).
__device__ __forceinline__ float2 norm_of(const float* sum, int ld, int c, float inv_count, float eps) {
  const float m = sum[c] * inv_count;
  const float var = fmaxf(sum[ld + c] * inv_count - m * m, 0.F);
  return make_float2(m, rsqrtf(var + eps));
}

// Huit valeurs FP16 normalisées puis Relu, arrondies en FP16.
__device__ __forceinline__ uint4 normalize8(uint4 v, const float (&m)[8], const float (&r)[8]) {
  std::uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    __half2 h;
    *reinterpret_cast<std::uint32_t*>(&h) = w[j];
    const float2 f = __half22float2(h);
    h = __floats2half2_rn(fmaxf((f.x - m[2 * j]) * r[2 * j], 0.F),
                          fmaxf((f.y - m[2 * j + 1]) * r[2 * j + 1], 0.F));
    w[j] = *reinterpret_cast<std::uint32_t*>(&h);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}

// La tranche [c0, c0 + Kc) de la tuile d'entrée en mémoire partagée (pas Kc + 8), IR × IC pixels lus de Step
// en Step depuis (y0, x0), rangés comme rn::load_nhwc (colonnes paires puis impaires en pas 2, GS = 2) ;
// zéros hors de l'image. kNorm : normalisés et Relu, stats les sommes de l'image.
template <int Kc, int GS, int Step, int IR, int IC, In I>
__device__ __forceinline__ void load_tile(__half* s, const Params& p, const __half* in, const float* stats,
                                          int c0, int y0, int x0) {
  constexpr int kChunks = Kc / 8, kLds = Kc + 8;
  static_assert(kThreads % kChunks == 0, "chaque fil garde ses huit canaux");
  const int chunk = static_cast<int>(threadIdx.x) % kChunks;
  float m[8] = {}, r[8] = {};
  if constexpr (I == In::kNorm) {
    const float inv_count = 1.F / static_cast<float>(p.ih * p.iw);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float2 n = norm_of(stats, p.in_ld, c0 + chunk * 8 + j, inv_count, p.eps);
      m[j] = n.x;
      r[j] = n.y;
    }
  }
  const std::uint32_t base = rn::smem_u32(s);
  for (int i = static_cast<int>(threadIdx.x); i < IR * IC * kChunks; i += kThreads) {
    const int q = i / kChunks;
    const int row = q / IC, col = q - row * IC;
    const int y = y0 + row * Step, x = x0 + col * Step;
    const bool ok = y >= 0 && y < p.ih && x >= 0 && x < p.iw;
    const __half* from = ok ? in + (static_cast<long>(y) * p.iw + x) * p.in_ld + c0 + chunk * 8 : in;
    const int at = (row * IC + rn::smem_col<GS, IC>(col)) * kLds + chunk * 8;
    if constexpr (I == In::kNorm) {
      *reinterpret_cast<uint4*>(s + at) =
          ok ? normalize8(__ldg(reinterpret_cast<const uint4*>(from)), m, r) : make_uint4(0U, 0U, 0U, 0U);
    } else {
      rn::cp_async16(base + static_cast<std::uint32_t>(at * 2), from, ok);
    }
  }
}

// La tuile de sortie (mémoire partagée, pas CT + 8) vers out, en lignes entières.
template <int CT, int Rows>
__device__ __forceinline__ void store_tile(const __half* o, const Params& p, __half* out, int ty0, int tx0,
                                           int n0) {
  constexpr int kTH = kWarps * Rows, kLdo = CT + 8, kChunks = CT / 8;
  for (int i = static_cast<int>(threadIdx.x); i < kTH * kTW * kChunks; i += kThreads) {
    const int q = i / kChunks, c = i - q * kChunks;
    const int y = ty0 + q / kTW, x = tx0 + q % kTW;
    if (y >= p.oh || x >= p.ow) continue;
    *reinterpret_cast<uint4*>(out + (static_cast<long>(y) * p.ow + x) * p.out_ld + n0 + c * 8) =
        *reinterpret_cast<const uint4*>(o + q * kLdo + c * 8);
  }
}

// Σx et Σx² des canaux [n0, n0 + CT) de la tuile (pixels de l'image seulement), ajoutés à sum (Σx) et
// sum + out_ld (Σx²). Chaque fil somme deux canaux sur un pixel sur kGroups, red (kReduceBytes) rassemble.
template <int CT, int Rows>
__device__ __forceinline__ void add_stats(const __half* o, float4* red, const Params& p, float* sum, int ty0,
                                          int tx0, int n0) {
  constexpr int kTH = kWarps * Rows, kLdo = CT + 8, kPairs = CT / 2, kGroups = kThreads / kPairs;
  const int pair = static_cast<int>(threadIdx.x) % kPairs, group = static_cast<int>(threadIdx.x) / kPairs;
  float4 acc = make_float4(0.F, 0.F, 0.F, 0.F);
  for (int q = group; q < kTH * kTW; q += kGroups) {
    if (ty0 + q / kTW >= p.oh || tx0 + q % kTW >= p.ow) continue;
    const float2 v = __half22float2(*reinterpret_cast<const __half2*>(o + q * kLdo + 2 * pair));
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.x * v.x;
    acc.w += v.y * v.y;
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < CT) {
    const int n = static_cast<int>(threadIdx.x), odd = n & 1;
    float s = 0.F, s2 = 0.F;
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      const float4 v = red[g * kPairs + (n >> 1)];
      s += odd != 0 ? v.y : v.x;
      s2 += odd != 0 ? v.w : v.z;
    }
    atomicAdd(sum + n0 + n, s);
    atomicAdd(sum + p.out_ld + n0 + n, s2);
  }
}

// La fin d'une convolution : biais et arrondi FP16 en mémoire partagée (s), écriture, puis les statistiques.
template <int CT, int Rows, Acc A>
__device__ __forceinline__ void finish(const rn::Frag<A> (&acc)[Rows * 2][CT / 8], __half* s, const Params& p,
                                       int image, int tile, int ty0, int tx0, int warp, int lane) {
  constexpr int kTH = kWarps * Rows;
  const int n0 = tile * CT;
  rn::stage<CT, Rows, A>(acc, s, p.bias + n0, warp, lane);
  __syncthreads();
  store_tile<CT, Rows>(s, p, p.out + static_cast<long>(image) * p.oh * p.ow * p.out_ld, ty0, tx0, n0);
  if (p.out_stats != nullptr) {
    auto* red = reinterpret_cast<float4*>(s + kTH * kTW * (CT + 8));
    add_stats<CT, Rows>(s, red, p, p.out_stats + 2L * image * p.out_ld, ty0, tx0, n0);
  }
}

// Une convolution KH × KH (3 ou 1) de pas S, bord KH / 2. Une 1 × 1 de pas 2 lit un pixel sur deux dans une
// tuile de pas 1.
template <int Kc, int CT, int KH, int S, int Rows, In I, Acc A>
__global__ void __launch_bounds__(kThreads, 2) conv_kernel(Params p) {
  static_assert(Kc % 16 == 0 && CT % 16 == 0 && CT <= 64, "canaux par 16");
  static_assert(I == In::kPlain || (KH == 3 && S == 1), "normalisation à la lecture : 3 × 3 de pas 1");
  constexpr int kGS = KH == 1 ? 1 : S;
  constexpr int kStep = KH == 1 ? S : 1;
  constexpr int kTaps = KH * KH;
  using G = rn::Geometry<KH, kGS, Rows>;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* s = reinterpret_cast<__half*>(smem_raw);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int ty0 = static_cast<int>(blockIdx.y) * G::kTH, tx0 = static_cast<int>(blockIdx.x) * kTW;
  const int image = static_cast<int>(blockIdx.z) / p.tiles,
            tile = static_cast<int>(blockIdx.z) - image * p.tiles;
  const __half* in = p.in + static_cast<long>(image) * p.ih * p.iw * p.in_ld;
  const float* stats = I == In::kNorm ? p.in_stats + 2L * image * p.in_ld : nullptr;
  const int kcs = p.cin / 16;
  const uint4* w = p.w + static_cast<long>(tile) * kTaps * kcs * (CT / 16) * 32;
  const int y0 = ty0 * S - KH / 2, x0 = tx0 * S - KH / 2;

  rn::Frag<A> acc[Rows * 2][CT / 8];
#pragma unroll
  for (int f = 0; f < Rows * 2; ++f)
#pragma unroll
    for (int nt = 0; nt < CT / 8; ++nt) acc[f][nt].clear();
#pragma unroll 1
  for (int q = 0; q < p.cin / Kc; ++q) {
    if (q > 0) __syncthreads();  // la tranche d'avant n'est plus lue
    load_tile<Kc, kGS, kStep, G::kIR, G::kIC, I>(s, p, in, stats, q * Kc, y0, x0);
    if constexpr (I == In::kPlain) rn::cp_async_wait_all();
    __syncthreads();
    rn::accumulate<Kc, CT, KH, kGS, Rows, A>(acc, s, w, q, kcs, 0, kTaps, warp, lane);
  }
  __syncthreads();  // la tuile d'entrée n'est plus lue : la mémoire partagée reçoit la sortie
  finish<CT, Rows, A>(acc, s, p, image, tile, ty0, tx0, warp, lane);
}

// La 7 × 7 de pas 2 sur les trois plans de l'image : le morceau d'image (13 × 69 pixels pour une tuile de
// 4 × 32, pad hors de l'image), puis déplié, le fil q remplissant la ligne de son pixel de sortie.
template <int CT, Acc A>
__global__ void __launch_bounds__(kThreads, 2) stem_kernel(Params p) {
  using G = rn::Geometry<1, 1, kStemRows>;
  static_assert(G::kTH * kTW == kThreads, "un fil par pixel de sortie");
  constexpr int kPatch = kStemPatchRows * kStemPatchCols;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* s = reinterpret_cast<__half*>(smem_raw);
  __half* patch = s + kThreads * kStemLds;
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int ty0 = static_cast<int>(blockIdx.y) * G::kTH, tx0 = static_cast<int>(blockIdx.x) * kTW;
  const int image = static_cast<int>(blockIdx.z) / p.tiles,
            tile = static_cast<int>(blockIdx.z) - image * p.tiles;
  const long plane = static_cast<long>(p.ih) * p.iw;
  const __half* in = p.in + 3L * image * plane;
  const int y0 = ty0 * 2 - 3, x0 = tx0 * 2 - 3;
  for (int i = static_cast<int>(threadIdx.x); i < kPatch; i += kThreads) {
    const int y = y0 + i / kStemPatchCols, x = x0 + i % kStemPatchCols;
    const bool ok = y >= 0 && y < p.ih && x >= 0 && x < p.iw;
    const long at = ok ? static_cast<long>(y) * p.iw + x : 0;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      patch[c * kPatch + i] = ok ? in[c * plane + at] : __float2half_rn(p.pad[c]);
    }
  }
  __syncthreads();
  {
    const int py = static_cast<int>(threadIdx.x) / kTW, px = static_cast<int>(threadIdx.x) % kTW;
    const __half* from = patch + 2 * py * kStemPatchCols + 2 * px;
    __half* row = s + threadIdx.x * kStemLds;
    // La valeur k du pixel : prise k / 3 (ligne, colonne), plan k % 3 ; zéro au-delà des 147.
    const auto at = [&](int k) -> std::uint32_t {
      if (k >= 7 * 7 * 3) return 0U;
      const int tap = k / 3;
      return __half_as_ushort(from[(k % 3) * kPatch + (tap / 7) * kStemPatchCols + tap % 7]);
    };
#pragma unroll
    for (int k0 = 0; k0 < kStemK; k0 += 8) {
      std::uint32_t w[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) w[j] = at(k0 + 2 * j) | (at(k0 + 2 * j + 1) << 16U);
      *reinterpret_cast<uint4*>(row + k0) = make_uint4(w[0], w[1], w[2], w[3]);
    }
  }
  __syncthreads();
  rn::Frag<A> acc[kStemRows * 2][CT / 8];
#pragma unroll
  for (int f = 0; f < kStemRows * 2; ++f)
#pragma unroll
    for (int nt = 0; nt < CT / 8; ++nt) acc[f][nt].clear();
  const uint4* w = p.w + static_cast<long>(tile) * (kStemK / 16) * (CT / 16) * 32;
  rn::accumulate<kStemK, CT, 1, 1, kStemRows, A>(acc, s, w, 0, kStemK / 16, 0, 1, warp, lane);
  __syncthreads();
  finish<CT, kStemRows, A>(acc, s, p, image, tile, ty0, tx0, warp, lane);
}

// La fin d'un bloc résiduel : out = relu(skip' + in'), x' = (x - m) · r (puis Relu si demandé) quand x a des
// statistiques, x sinon. Un fil par pixel et huit canaux ; blockIdx.y : l'image.
struct EndParams {
  const __half* in = nullptr;
  const float* in_stats = nullptr;  // nul : lu tel quel
  bool in_relu = false;
  const __half* skip = nullptr;
  const float* skip_stats = nullptr;
  bool skip_relu = false;
  __half* out = nullptr;
  int channels = 0;  // au plus kMaxChannels, par 8
  long pixels = 0;   // d'une image
  float eps = 1e-5F;
};

__global__ void __launch_bounds__(256) block_end_kernel(EndParams p) {
  __shared__ float2 coef[2][kMaxChannels];  // (m, r) de in, puis de skip
  const int image = static_cast<int>(blockIdx.y), ld = p.channels;
  const float inv_count = 1.F / static_cast<float>(p.pixels);
  for (int c = static_cast<int>(threadIdx.x); c < ld; c += static_cast<int>(blockDim.x)) {
    const float2 none = make_float2(0.F, 1.F);
    coef[0][c] =
        p.in_stats != nullptr ? norm_of(p.in_stats + 2L * image * ld, ld, c, inv_count, p.eps) : none;
    coef[1][c] =
        p.skip_stats != nullptr ? norm_of(p.skip_stats + 2L * image * ld, ld, c, inv_count, p.eps) : none;
  }
  __syncthreads();
  const int groups = ld / 8;
  const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= p.pixels * groups) return;
  const long at = static_cast<long>(image) * p.pixels * ld + i * 8;
  const int c0 = static_cast<int>(i % groups) * 8;
  const uint4 a = __ldg(reinterpret_cast<const uint4*>(p.in + at));
  const uint4 b = __ldg(reinterpret_cast<const uint4*>(p.skip + at));
  const std::uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
  std::uint32_t w[4];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    __half2 ha, hb;
    *reinterpret_cast<std::uint32_t*>(&ha) = av[j / 2];
    *reinterpret_cast<std::uint32_t*>(&hb) = bv[j / 2];
    const float x = __half2float((j & 1) != 0 ? __high2half(ha) : __low2half(ha));
    const float y = __half2float((j & 1) != 0 ? __high2half(hb) : __low2half(hb));
    const float2 ca = coef[0][c0 + j], cb = coef[1][c0 + j];
    float xa = (x - ca.x) * ca.y, yb = (y - cb.x) * cb.y;
    if (p.in_relu) xa = fmaxf(xa, 0.F);
    if (p.skip_relu) yb = fmaxf(yb, 0.F);
    const __half h = __float2half_rn(fmaxf(xa + yb, 0.F));
    if ((j & 1) == 0) {
      w[j / 2] = __half_as_ushort(h);
    } else {
      w[j / 2] |= static_cast<std::uint32_t>(__half_as_ushort(h)) << 16U;
    }
  }
  *reinterpret_cast<uint4*>(p.out + at) = make_uint4(w[0], w[1], w[2], w[3]);
}

}  // namespace kirari::gpu::cuda::gmflow_backbone
