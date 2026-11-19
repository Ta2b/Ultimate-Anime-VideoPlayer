#pragma once
// Noyaux des couches d'attention du transformeur de GMFlow natif (docs/gmflow-natif.md, PR 2), sur les
// fragments de rife_native_kernels.cuh (mma.sync m16n8k16, deux m16n8k8 sur sm_75 ; ldmatrix ; cp.async dès
// sm_80).
//
// Les jetons sont rangés [lot][hauteur · largeur][128] FP16. Une couche, c'est deux lancements :
//  1. qkv_kernel : la GEMM 128 → 384 (q, k, v d'un coup), sommes selon Acc. q se lit dans query au lot
//     b, k et v dans keys au lot (b + key_shift) mod lot : en cross-attention, l'autre moitié du lot, par
//     l'indice, sans copie. Sortie [lot][jetons][384] FP16 (q | k | v).
//  2. attention_kernel : l'attention par fenêtres, fondue à la FlashAttention-2. Un bloc de 4 warps prend
//     64 requêtes d'une fenêtre (16 par warp) ; K et V de la fenêtre passent par tuiles de 64 clés en
//     mémoire partagée (cp.async, K et V chacun dans son tampon : V de la tuile se charge pendant S = QKᵀ,
//     K de la suivante pendant P · V). S en mma FP16 à sommes f32, facteur et masque en f32 (en base 2 :
//     exp2), maximum et somme courants en registres, P arrondi en FP16 pour le mma P · V à sommes f32. Le
//     décalage de Swin est une arithmétique d'adresses : le jeton t de la fenêtre (wy, wx) est à
//     ((wy · wh + t / ww + sh) mod H, (wx · ww + t mod ww + sw) mod W), la sortie s'écrit à la place de sa
//     requête ; le masque compare les régions (trois par axe, au bord moins une fenêtre et au bord moins
//     un décalage) calculées au chargement. Épilogue sans mémoire : O / l arrondi en FP16 devient le
//     fragment A de la fusion 128 × 128 (poids rangés pour le mma, lus en L1), la LayerNorm se fait sur la
//     ligne tenue par les quatre fils d'un groupe (moyenne, puis variance des écarts, en f32), puis le
//     résiduel (self) et une seule écriture FP16.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::gmflow_attention {

namespace rn = rife_native;
using rn::Acc;

inline constexpr int kD = 128;
inline constexpr int kQkv = 3 * kD;
inline constexpr int kThreads = 128;
inline constexpr int kRows = 64;        // requêtes d'un bloc, clés d'une tuile, jetons d'un bloc de la GEMM
inline constexpr int kLds = kD + 8;     // pas d'une ligne en mémoire partagée (sans conflit de banques)
inline constexpr int kSteps = kD / 16;  // pas k de 16 sur les canaux
inline constexpr int kPairs = kD / 16;  // paires de tuiles n8 sur les canaux
inline constexpr int kTileBytes = kRows * kLds * 2;
inline constexpr int kSmemBytes = 2 * kTileBytes + kRows * 4;  // K, V et les régions des clés
inline constexpr float kLog2e = 1.4426950408889634F;

struct QkvParams {
  const __half* query = nullptr;  // [lot][tokens][128]
  const __half* keys = nullptr;
  __half* out = nullptr;     // [lot][tokens][384]
  const uint4* w = nullptr;  // trois parties (q, k, v) de 128 sorties rangées pour le mma
  long tokens = 0;           // jetons d'une image
  int batch = 0;
  int key_shift = 0;  // 0, ou lot / 2 (cross-attention)
};

// La géométrie des fenêtres d'une échelle : jetons height × width par image, splits × splits fenêtres de wh ×
// ww, décalage (sh, sw) nul hors des couches décalées.
struct Grid {
  int height = 0, width = 0;
  int splits = 1;
  int wh = 0, ww = 0;
  int sh = 0, sw = 0;
  int length = 0;  // wh · ww
  bool shifted = false;
};

struct AttnParams {
  const __half* qkv = nullptr;     // [lot][tokens][384]
  const __half* source = nullptr;  // résiduel [lot][tokens][128], nul en cross-attention
  __half* out = nullptr;           // [lot][tokens][128]
  const uint4* merge = nullptr;    // fusion 128 × 128 rangée pour le mma
  const float* gamma = nullptr;
  const float* beta = nullptr;
  Grid grid;
  long tokens = 0;
  float scale_log2 = 0.F;  // log2(e) / √d
  float mask_log2 = 0.F;   // le masque (-100) · log2(e)
  float eps = 1e-5F;
};

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

__device__ __forceinline__ void ldmatrix4_trans(std::uint32_t (&a)[4], std::uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}

__device__ __forceinline__ std::uint32_t half2_bits(float lo, float hi) {
  const __half2 h = rn::to_half2(lo, hi);
  return *reinterpret_cast<const std::uint32_t*>(&h);
}

// Le jeton t de la fenêtre window : son indice dans l'image et sa région dans l'image décalée (0 hors
// décalage).
__device__ __forceinline__ int token_of(const Grid& g, int window, int t, int& region) {
  const int wy = window / g.splits, wx = window - wy * g.splits;
  const int r = t / g.ww, c = t - r * g.ww;
  const int yr = wy * g.wh + r, xr = wx * g.ww + c;
  int y = yr + g.sh, x = xr + g.sw;
  if (y >= g.height) y -= g.height;
  if (x >= g.width) x -= g.width;
  region = 0;
  if (g.shifted) {
    const int ry = (yr >= g.height - g.wh ? 1 : 0) + (yr >= g.height - g.sh ? 1 : 0);
    const int rx = (xr >= g.width - g.ww ? 1 : 0) + (xr >= g.width - g.sw ? 1 : 0);
    region = ry * 3 + rx;
  }
  return y * g.width + x;
}

// La GEMM QKV : un bloc de 64 jetons et une partie (q, k ou v : blockIdx.y), 16 jetons par warp sur les 128
// sorties de la partie. Les jetons dans la mémoire partagée (cp.async), les poids lus en L1.
template <Acc A>
__global__ void __launch_bounds__(kThreads) qkv_kernel(QkvParams p) {
  __shared__ __align__(16) __half s[kRows * kLds];
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const long first = static_cast<long>(blockIdx.x) * kRows;
  const int part = static_cast<int>(blockIdx.y), batch = static_cast<int>(blockIdx.z);
  const int from_batch = part == 0 ? batch : (batch + p.key_shift) % p.batch;
  const __half* in = (part == 0 ? p.query : p.keys) + static_cast<long>(from_batch) * p.tokens * kD;
  const int chunk = static_cast<int>(threadIdx.x) & 15;
  for (int row = static_cast<int>(threadIdx.x) >> 4; row < kRows; row += kThreads / 16) {
    const bool ok = first + row < p.tokens;
    rn::cp_async16(rn::smem_u32(s + row * kLds + chunk * 8), ok ? in + (first + row) * kD + chunk * 8 : in,
                   ok);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  rn::Frag<A> acc[2 * kPairs];
#pragma unroll
  for (int nt = 0; nt < 2 * kPairs; ++nt) acc[nt].clear();
  const std::uint32_t base =
      rn::smem_u32(s) + static_cast<std::uint32_t>(((warp * 16 + (lane & 15)) * kLds + (lane >> 4) * 8) * 2);
  const uint4* w = p.w + static_cast<long>(part) * kSteps * kPairs * 32 + lane;
#pragma unroll
  for (int ks = 0; ks < kSteps; ++ks) {
    std::uint32_t a[4];
    rn::ldmatrix4(a, base + static_cast<std::uint32_t>(ks * 32));
#pragma unroll
    for (int np = 0; np < kPairs; ++np) {
      const uint4 b = __ldg(w + (ks * kPairs + np) * 32);
      acc[2 * np].mma(a, b.x, b.y);
      acc[2 * np + 1].mma(a, b.z, b.w);
    }
#pragma unroll
    for (int nt = 0; nt < 2 * kPairs; ++nt) acc[nt].promote();
  }
  __syncthreads();  // la tuile d'entrée n'est plus lue : elle reçoit la sortie
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int nt = 0; nt < 2 * kPairs; ++nt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float2 v = acc[nt].pair(h);
      *reinterpret_cast<__half2*>(s + (warp * 16 + g + 8 * h) * kLds + nt * 8 + 2 * t) =
          rn::to_half2(v.x, v.y);
    }
  __syncthreads();
  __half* out = p.out + static_cast<long>(batch) * p.tokens * kQkv + part * kD;
  for (int row = static_cast<int>(threadIdx.x) >> 4; row < kRows; row += kThreads / 16) {
    if (first + row >= p.tokens) continue;
    *reinterpret_cast<uint4*>(out + (first + row) * kQkv + chunk * 8) =
        *reinterpret_cast<const uint4*>(s + row * kLds + chunk * 8);
  }
}

// 64 lignes de la partie part (0 : q, 1 : k, 2 : v) des jetons first.. de la fenêtre, en mémoire partagée
// (pas kLds), zéros au-delà de la fenêtre. regions (non nul) : la région de chaque clé, -1 au-delà.
__device__ __forceinline__ void load_rows(__half* s, int* regions, const AttnParams& p, const __half* qkv,
                                          int window, int first, int part) {
  const int chunk = static_cast<int>(threadIdx.x) & 15;
  for (int row = static_cast<int>(threadIdx.x) >> 4; row < kRows; row += kThreads / 16) {
    const int t = first + row;
    const bool ok = t < p.grid.length;
    int region = -1;
    const int token = ok ? token_of(p.grid, window, t, region) : 0;
    const __half* from = qkv + static_cast<long>(token) * kQkv + part * kD + chunk * 8;
    rn::cp_async16(rn::smem_u32(s + row * kLds + chunk * 8), from, ok);
    if (regions != nullptr && chunk == 0) regions[row] = ok ? region : -1;
  }
}

// S = Q Kᵀ d'une tuile pour les 16 requêtes du warp : q les fragments A de Q (8 pas sur les canaux), sk la
// tuile K.
__device__ __forceinline__ void scores(rn::Frag<Acc::kF32> (&s)[8], const std::uint32_t (&q)[kSteps][4],
                                       const __half* sk, int lane) {
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) s[nt].clear();
  // ldmatrix.x4 : clés 16 · np + (lane / 16) · 8 + lane mod 8, canaux 16 · ks + ((lane / 8) mod 2) · 8.
  const std::uint32_t base =
      rn::smem_u32(sk) +
      static_cast<std::uint32_t>((((lane >> 4) * 8 + (lane & 7)) * kLds + ((lane >> 3) & 1) * 8) * 2);
#pragma unroll
  for (int ks = 0; ks < kSteps; ++ks)
#pragma unroll
    for (int np = 0; np < 4; ++np) {
      std::uint32_t b[4];
      rn::ldmatrix4(b, base + static_cast<std::uint32_t>((np * 16 * kLds + ks * 16) * 2));
      s[2 * np].mma(q[ks], b[0], b[1]);
      s[2 * np + 1].mma(q[ks], b[2], b[3]);
    }
}

// Le softmax en ligne d'une tuile : facteur et masque en base 2, maximum m et somme partielle l du fil
// (rangées g et g + 8), O remis à l'échelle, P arrondi en FP16 dans les fragments A du mma P · V (4 pas de 16
// clés).
__device__ __forceinline__ void online_softmax(rn::Frag<Acc::kF32> (&s)[8],
                                               rn::Frag<Acc::kF32> (&o)[2 * kPairs],
                                               std::uint32_t (&pa)[4][4], float (&m)[2], float (&l)[2],
                                               const int (&qreg)[2], const int* kreg, const AttnParams& p,
                                               int lane) {
  const int t = lane & 3;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float x[16];
    float top = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const float2 v = s[nt].pair(h);
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int region = kreg[nt * 8 + 2 * t + e];
        float y = (e == 0 ? v.x : v.y) * p.scale_log2 + (region != qreg[h] ? p.mask_log2 : 0.F);
        y = region < 0 ? -INFINITY : y;
        x[2 * nt + e] = y;
        top = fmaxf(top, y);
      }
    }
    top = fmaxf(top, __shfl_xor_sync(0xffffffffU, top, 1));
    top = fmaxf(top, __shfl_xor_sync(0xffffffffU, top, 2));
    const float next = fmaxf(m[h], top);
    const float alpha = exp2f(m[h] - next);
    m[h] = next;
    float sum = 0.F;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      x[i] = exp2f(x[i] - next);
      sum += x[i];
    }
    l[h] = l[h] * alpha + sum;
#pragma unroll
    for (int nt = 0; nt < 2 * kPairs; ++nt) {
      o[nt].r[2 * h] *= alpha;
      o[nt].r[2 * h + 1] *= alpha;
    }
    // Fragment A de P : pas kk = tuiles n8 2kk (a0 rangée g, a1 rangée g + 8) et 2kk + 1 (a2, a3).
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      pa[kk][h] = half2_bits(x[4 * kk], x[4 * kk + 1]);
      pa[kk][2 + h] = half2_bits(x[4 * kk + 2], x[4 * kk + 3]);
    }
  }
}

// O += P V d'une tuile. ldmatrix.x4.trans : clés 16 · kk + ((lane / 8) mod 2) · 8 + lane mod 8, canaux
// 16 · cp + (lane / 16) · 8 ; b0, b1 de la tuile n8 2cp puis de 2cp + 1.
__device__ __forceinline__ void accumulate_pv(rn::Frag<Acc::kF32> (&o)[2 * kPairs],
                                              const std::uint32_t (&pa)[4][4], const __half* sv, int lane) {
  const std::uint32_t base =
      rn::smem_u32(sv) +
      static_cast<std::uint32_t>(((((lane >> 3) & 1) * 8 + (lane & 7)) * kLds + (lane >> 4) * 8) * 2);
#pragma unroll
  for (int kk = 0; kk < 4; ++kk)
#pragma unroll
    for (int cp = 0; cp < kPairs; ++cp) {
      std::uint32_t b[4];
      ldmatrix4_trans(b, base + static_cast<std::uint32_t>((kk * 16 * kLds + cp * 16) * 2));
      o[2 * cp].mma(pa[kk], b[0], b[1]);
      o[2 * cp + 1].mma(pa[kk], b[2], b[3]);
    }
}

// La somme sur les quatre fils d'un groupe (une rangée).
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffU, v, 1);
  return v + __shfl_xor_sync(0xffffffffU, v, 2);
}

// L'épilogue : O / l en FP16, fusion, LayerNorm, résiduel, écriture. tokens : l'indice dans l'image des
// rangées g et g + 8 du warp (-1 : au-delà de la fenêtre).
template <Acc A>
__device__ __forceinline__ void finish(rn::Frag<Acc::kF32> (&o)[2 * kPairs], const float (&l)[2],
                                       const int (&tokens)[2], const AttnParams& p, int batch, int lane) {
  const float inv[2] = {1.F / quad_sum(l[0]), 1.F / quad_sum(l[1])};
  std::uint32_t oa[kSteps][4];
#pragma unroll
  for (int ks = 0; ks < kSteps; ++ks)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      oa[ks][h] = half2_bits(o[2 * ks].r[2 * h] * inv[h], o[2 * ks].r[2 * h + 1] * inv[h]);
      oa[ks][2 + h] = half2_bits(o[2 * ks + 1].r[2 * h] * inv[h], o[2 * ks + 1].r[2 * h + 1] * inv[h]);
    }
  rn::Frag<A> y[2 * kPairs];
#pragma unroll
  for (int nt = 0; nt < 2 * kPairs; ++nt) y[nt].clear();
  const uint4* w = p.merge + lane;
#pragma unroll
  for (int ks = 0; ks < kSteps; ++ks) {
#pragma unroll
    for (int np = 0; np < kPairs; ++np) {
      const uint4 b = __ldg(w + (ks * kPairs + np) * 32);
      y[2 * np].mma(oa[ks], b.x, b.y);
      y[2 * np + 1].mma(oa[ks], b.z, b.w);
    }
#pragma unroll
    for (int nt = 0; nt < 2 * kPairs; ++nt) y[nt].promote();
  }
  const int t = lane & 3;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float sum = 0.F;
#pragma unroll
    for (int nt = 0; nt < 2 * kPairs; ++nt) sum += y[nt].pair(h).x + y[nt].pair(h).y;
    const float mean = quad_sum(sum) * (1.F / kD);
    float var = 0.F;
#pragma unroll
    for (int nt = 0; nt < 2 * kPairs; ++nt) {
      const float2 v = y[nt].pair(h);
      var += (v.x - mean) * (v.x - mean) + (v.y - mean) * (v.y - mean);
    }
    const float rstd = rsqrtf(quad_sum(var) * (1.F / kD) + p.eps);
    if (tokens[h] < 0) continue;
    const long at = (static_cast<long>(batch) * p.tokens + tokens[h]) * kD + 2 * t;
#pragma unroll
    for (int nt = 0; nt < 2 * kPairs; ++nt) {
      const int c = nt * 8;
      const float2 v = y[nt].pair(h);
      const float2 ga = __ldg(reinterpret_cast<const float2*>(p.gamma + c + 2 * t));
      const float2 be = __ldg(reinterpret_cast<const float2*>(p.beta + c + 2 * t));
      float2 r = make_float2((v.x - mean) * rstd * ga.x + be.x, (v.y - mean) * rstd * ga.y + be.y);
      if (p.source != nullptr) {
        const float2 src = __half22float2(__ldg(reinterpret_cast<const __half2*>(p.source + at + c)));
        r.x += src.x;
        r.y += src.y;
      }
      *reinterpret_cast<__half2*>(p.out + at + c) = rn::to_half2(r.x, r.y);
    }
  }
}

// Deux blocs par multiprocesseur sur sm_120 : sans ce plancher, ptxas de CUDA 13.4 y vise trois blocs (168
// registres) et déborde de 16 octets ; avec, 254 registres et rien en mémoire locale. Les autres
// architectures tiennent sans débordement à leur propre choix, qu'un plancher explicite changerait (même à un
// bloc, ptxas monte alors à 254 registres).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define KIRARI_GMFLOW_ATTN_BOUNDS __launch_bounds__(kThreads, 2)
#else
#define KIRARI_GMFLOW_ATTN_BOUNDS __launch_bounds__(kThreads)
#endif

// L'attention d'une couche : blockIdx.x les 64 requêtes, blockIdx.y la fenêtre, blockIdx.z le lot. Mémoire
// partagée dynamique : kSmemBytes.
template <Acc A>
__global__ void KIRARI_GMFLOW_ATTN_BOUNDS attention_kernel(AttnParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* sk = reinterpret_cast<__half*>(smem_raw);
  __half* sv = sk + kRows * kLds;
  int* kreg = reinterpret_cast<int*>(sv + kRows * kLds);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int window = static_cast<int>(blockIdx.y), batch = static_cast<int>(blockIdx.z);
  const int first = static_cast<int>(blockIdx.x) * kRows;
  const __half* qkv = p.qkv + static_cast<long>(batch) * p.tokens * kQkv;
  // Q dans le tampon de V, K de la première tuile dans le sien ; Q passe en registres.
  load_rows(sv, nullptr, p, qkv, window, first, 0);
  cp_async_commit();
  load_rows(sk, kreg, p, qkv, window, 0, 1);
  cp_async_commit();
  cp_async_wait<1>();
  __syncthreads();
  std::uint32_t q[kSteps][4];
  {
    const std::uint32_t base =
        rn::smem_u32(sv) +
        static_cast<std::uint32_t>(((warp * 16 + (lane & 15)) * kLds + (lane >> 4) * 8) * 2);
#pragma unroll
    for (int ks = 0; ks < kSteps; ++ks) rn::ldmatrix4(q[ks], base + static_cast<std::uint32_t>(ks * 32));
  }
  int qreg[2], tokens[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int t = first + warp * 16 + (lane >> 2) + 8 * h;
    qreg[h] = 0;
    tokens[h] = t < p.grid.length ? token_of(p.grid, window, t, qreg[h]) : -1;
  }
  __syncthreads();  // Q est en registres : le tampon de V est libre
  load_rows(sv, nullptr, p, qkv, window, 0, 2);
  cp_async_commit();
  rn::Frag<Acc::kF32> o[2 * kPairs];
#pragma unroll
  for (int nt = 0; nt < 2 * kPairs; ++nt) o[nt].clear();
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.F, 0.F};
  const int tiles = (p.grid.length + kRows - 1) / kRows;
#pragma unroll 1
  for (int j = 0; j < tiles; ++j) {
    cp_async_wait<1>();  // K de la tuile j (V peut encore venir)
    __syncthreads();
    rn::Frag<Acc::kF32> s[8];
    scores(s, q, sk, lane);
    std::uint32_t pa[4][4];
    online_softmax(s, o, pa, m, l, qreg, kreg, p, lane);
    __syncthreads();  // K et les régions de la tuile ne sont plus lus
    const bool more = j + 1 < tiles;
    if (more) {
      load_rows(sk, kreg, p, qkv, window, (j + 1) * kRows, 1);
      cp_async_commit();
      cp_async_wait<1>();  // V de la tuile j
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    accumulate_pv(o, pa, sv, lane);
    __syncthreads();  // V de la tuile n'est plus lu
    if (more) {
      load_rows(sv, nullptr, p, qkv, window, (j + 1) * kRows, 2);
      cp_async_commit();
    }
  }
  finish<A>(o, l, tokens, p, batch, lane);
}

}  // namespace kirari::gpu::cuda::gmflow_attention
