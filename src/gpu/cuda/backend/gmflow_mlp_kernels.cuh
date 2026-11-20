#pragma once
// Le MLP fondu des couches de cross-attention du transformeur de GMFlow natif (docs/gmflow-natif.md, « MLP
// natif (PR 3) ») : out = x + LayerNorm(W2 · GELU(W1 · [x, message])), par jeton, 128 canaux, couche cachée
// de hidden canaux (1024), sans biais aux deux GEMM, GELU exacte (erf), LayerNorm à échelle et décalage
// appris : ce que fait le graphe exporté, que prepare/gmflow_mlp.h vérifie nœud par nœud.
//
// Un bloc de W warps prend 16 · W jetons : [x, message] (256 canaux FP16) en mémoire partagée, chargés une
// fois. Chaque warp tient 16 jetons. Pour chaque tranche de 64 canaux de la couche cachée : H = X · W1
// (mma.sync m16n8k16, sommes A), GELU en f32 arrondie en FP16 directement dans les fragments A du mma suivant
// (la sortie m16n8 de deux tuiles voisines est le fragment A m16k16 : ni mémoire partagée ni mémoire
// globale), puis O += H · W2. O (16 jetons × 128 canaux en f32 par warp) reste en registres jusqu'à la fin :
// LayerNorm de chaque ligne (les quatre fils d'une ligne réduits par __shfl_xor), résiduel lu dans la tuile x
// en mémoire partagée, arrondi FP16, écriture en lignes entières. La couche cachée n'est jamais écrite.
//
// Les poids passent par un anneau de trois étages de 4 Ko en mémoire partagée (cp.async), communs aux warps
// du bloc : la L2 ne les donne qu'une fois par bloc. gmflow_mlp_net.cu les range en un flux d'étages dans
// l'ordre où le noyau les prend : pour chaque tranche, 8 étages de W1 (deux pas k de 16 entrées × 4 paires de
// tuiles n8 × 32 lanes) puis 4 étages de W2 (un pas k × 8 paires × 32 lanes) ; un uint4 par lane et par
// paire, b0 et b1 des deux tuiles, l'ordre des fragments B de rife_native::accumulate.
//
// Petites grilles (le huitième au demi-flot : 30 blocs) : la couche cachée se découpe entre splits blocs
// (blockIdx.y), chacun écrit sa part de O en f32 dans partial, et finish_kernel somme les parts dans un ordre
// fixe (déterministe), puis fait LayerNorm et résiduel.
//
// Sommes (Acc de rife_native_kernels.cuh) : gmflow_mlp_net.cu n'instancie que kF32 ; kMixed reporterait en
// f32 tous les quatre pas k de la première GEMM et à la fin de chaque tranche de la seconde (promote).
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::gmflow_mlp {

namespace rn = rife_native;
using rn::Acc;

inline constexpr int kChannels = 128;      // C
inline constexpr int kIn = 2 * kChannels;  // [x, message]
inline constexpr int kHiddenTile = 64;     // canaux cachés par tranche
inline constexpr int kRows = 16;           // jetons par warp
inline constexpr int kLdx = kIn + 8;       // pas d'une ligne de la tuile (ldmatrix sans conflit de banques)
inline constexpr int kK1 = kIn / 16;       // pas k de la première GEMM
inline constexpr int kNP1 = kHiddenTile / 16;  // paires de tuiles n8 de H
inline constexpr int kK2 = kHiddenTile / 16;   // pas k de la seconde GEMM par tranche
inline constexpr int kNP2 = kChannels / 16;    // paires de tuiles n8 de O
inline constexpr int kTiles = kChannels / 8;   // tuiles n8 de O
inline constexpr int kStageVec = 256;          // uint4 par étage (4 Ko)
inline constexpr int kStages = 3;              // étages de l'anneau
inline constexpr int kW1Stages = kK1 / 2;      // deux pas k de W1 par étage
inline constexpr int kChunkStages = kW1Stages + kK2;
inline constexpr float kInvSqrt2 = 0.70710678118654752F;
static_assert(2 * kNP1 * 32 == kStageVec && kNP2 * 32 == kStageVec, "un étage : 2 pas de W1 ou 1 pas de W2");

// La forme d'un bloc de Warps warps.
template <int Warps>
struct Block {
  static constexpr int kThreads = Warps * 32;
  static constexpr int kTokens = Warps * kRows;
  static constexpr int kTileBytes = kTokens * kLdx * 2;
  static constexpr int kSmemBytes = kTileBytes + kStages * kStageVec * 16;
  static constexpr int kWideSmemBytes = kTileBytes + 4 * kStageVec * 16;
};

struct Params {
  const __half* x = nullptr;        // [tokens][128] FP16 : la source (le résiduel)
  const __half* message = nullptr;  // [tokens][128] FP16 : la sortie de l'attention, après sa LayerNorm
  __half* out = nullptr;  // [tokens][128] FP16 ; peut être x (chaque bloc lit sa tuile avant d'écrire)
  long tokens = 0;
  int hidden = 0;                // multiple de kHiddenTile
  int splits = 1;                // parts de la couche cachée (blockIdx.y)
  float* partial = nullptr;      // splits > 1 : [splits][tokens][128] f32
  const uint4* w = nullptr;      // le flux d'étages de la couche (gmflow_mlp_net.cu)
  const float* gamma = nullptr;  // LayerNorm, 128 valeurs chacun
  const float* beta = nullptr;
  float eps = 1e-5F;
};

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
// Attendre que N groupes au plus restent en vol (sm_75 : les copies sont déjà faites).
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

// La tuile [x, message] des jetons [t0, t0 + Tokens) : ligne r, canaux 0-127 = x, 128-255 = message ; zéros
// au-delà de tokens.
template <int Threads, int Tokens>
__device__ __forceinline__ void load_tokens(__half* s, const Params& p, long t0) {
  constexpr int kChunks = kIn / 8;  // paquets de 16 octets par ligne
  const std::uint32_t base = rn::smem_u32(s);
  for (int i = static_cast<int>(threadIdx.x); i < Tokens * kChunks; i += Threads) {
    const int r = i / kChunks, c = i - r * kChunks;
    const long token = t0 + r;
    const bool ok = token < p.tokens;
    const __half* from = c < kChunks / 2 ? p.x + token * kChannels + c * 8
                                         : p.message + token * kChannels + (c - kChunks / 2) * 8;
    rn::cp_async16(base + static_cast<std::uint32_t>((r * kLdx + c * 8) * 2), ok ? from : p.x, ok);
  }
}

// L'anneau des poids : les étages [next, end) du flux restent à lancer.
struct Ring {
  const uint4* w;
  uint4* slots;
  int next;
  int end;
};

// Lance l'étage next dans sa place (un groupe cp.async, vide au-delà de la fin : le compte reste régulier).
template <int Threads>
__device__ __forceinline__ void issue(Ring& r) {
  if (r.next < r.end) {
    const uint4* from = r.w + static_cast<long>(r.next) * kStageVec;
    const std::uint32_t to = rn::smem_u32(r.slots + (r.next % kStages) * kStageVec);
    for (int i = static_cast<int>(threadIdx.x); i < kStageVec; i += Threads)
      rn::cp_async16(to + static_cast<std::uint32_t>(i * 16), from + i, true);
  }
  cp_async_commit();
  ++r.next;
}

// L'étage q, prêt pour tous les fils ; l'étage q + 2 part dans la place de q - 1, que tous ont finie.
template <int Threads>
__device__ __forceinline__ const uint4* acquire(Ring& r, int q) {
  cp_async_wait<kStages - 2>();
  __syncthreads();
  issue<Threads>(r);
  return r.slots + (q % kStages) * kStageVec;
}

// L'anneau large utilise deux cases de deux étages. L'ancienne case n'est réutilisée
// qu'après le point de rendez-vous qui publie la suivante ; une copie reste en vol pendant le calcul.
template <int Threads>
__device__ __forceinline__ void issue_wide(Ring& r) {
  if (r.next == r.end) return;
  constexpr int kVectors = 2 * kStageVec;
  const uint4* from = r.w + static_cast<long>(r.next) * kVectors;
  const std::uint32_t to = rn::smem_u32(r.slots + (r.next & 1) * kVectors);
  for (int i = static_cast<int>(threadIdx.x); i < kVectors; i += Threads)
    rn::cp_async16(to + static_cast<std::uint32_t>(i * 16), from + i, true);
  cp_async_commit();
  ++r.next;
}

template <int Threads>
__device__ __forceinline__ const uint4* acquire_wide(Ring& r, int q) {
  cp_async_wait<0>();
  __syncthreads();
  issue_wide<Threads>(r);
  return r.slots + (q & 1) * (2 * kStageVec);
}

// Un étage de W1 : les pas k 2i et 2i + 1 de H = X · W1. a_base : l'adresse ldmatrix de la lane (ligne lane %
// 16 du warp, demi-K lane / 16) ; slot : l'étage en mémoire partagée.
template <Acc A>
__device__ __forceinline__ void gemm1_stage(rn::Frag<A> (&h)[2 * kNP1], std::uint32_t a_base,
                                            const uint4* slot, int i, int lane) {
#pragma unroll
  for (int kl = 0; kl < 2; ++kl) {
    std::uint32_t a[4];
    rn::ldmatrix4(a, a_base + static_cast<std::uint32_t>((2 * i + kl) * 16 * 2));
#pragma unroll
    for (int np = 0; np < kNP1; ++np) {
      const uint4 b = slot[(kl * kNP1 + np) * 32 + lane];
      h[2 * np].mma(a, b.x, b.y);
      h[2 * np + 1].mma(a, b.z, b.w);
    }
  }
}

using InputStage = std::uint32_t[2][4];

__device__ __forceinline__ void load_input_stage(InputStage& a, std::uint32_t a_base, int i) {
#pragma unroll
  for (int kl = 0; kl < 2; ++kl)
    rn::ldmatrix4(a[kl], a_base + static_cast<std::uint32_t>((2 * i + kl) * 16 * 2));
}

template <Acc A>
__device__ __forceinline__ void gemm1_stage(rn::Frag<A> (&h)[2 * kNP1], const InputStage& a,
                                            const uint4* slot, int lane) {
#pragma unroll
  for (int kl = 0; kl < 2; ++kl) {
#pragma unroll
    for (int np = 0; np < kNP1; ++np) {
      const uint4 b = slot[(kl * kNP1 + np) * 32 + lane];
      h[2 * np].mma(a[kl], b.x, b.y);
      h[2 * np + 1].mma(a[kl], b.z, b.w);
    }
  }
}

// GELU exacte du graphe : x · (erf(x / √2) + 1) · 0,5.
__device__ __forceinline__ float gelu(float v) {
  return 0.5F * v * (1.F + erff(v * kInvSqrt2));
}

__device__ __forceinline__ std::uint32_t pack_half2(float lo, float hi) {
  const __half2 h = rn::to_half2(lo, hi);
  return *reinterpret_cast<const std::uint32_t*>(&h);
}

// GELU de H, arrondie en FP16, rangée dans les fragments A de H · W2 : le pas k j prend les tuiles 2j (canaux
// k 2t, 2t + 1 : a0 ligne g, a1 ligne g + 8) et 2j + 1 (canaux k 2t + 8, 2t + 9 : a2, a3).
template <Acc A>
__device__ __forceinline__ void gelu_stage(const rn::Frag<A> (&h)[2 * kNP1], int j, std::uint32_t (&a)[4]) {
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const float2 lo = h[2 * j + half].pair(0), hi = h[2 * j + half].pair(1);
    a[2 * half] = pack_half2(gelu(lo.x), gelu(lo.y));
    a[2 * half + 1] = pack_half2(gelu(hi.x), gelu(hi.y));
  }
}

template <Acc A>
__device__ __forceinline__ void gelu_to_a(const rn::Frag<A> (&h)[2 * kNP1], std::uint32_t (&a)[kK2][4]) {
#pragma unroll
  for (int j = 0; j < kK2; ++j)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float2 lo = h[2 * j + half].pair(0), hi = h[2 * j + half].pair(1);
      a[j][2 * half] = pack_half2(gelu(lo.x), gelu(lo.y));
      a[j][2 * half + 1] = pack_half2(gelu(hi.x), gelu(hi.y));
    }
}

// Un étage de W2 : O += GELU(H)[pas k j] · W2.
template <Acc A>
__device__ __forceinline__ void gemm2_stage(rn::Frag<A> (&o)[kTiles], const std::uint32_t (&a)[4],
                                            const uint4* slot, int lane) {
#pragma unroll
  for (int np = 0; np < kNP2; ++np) {
    const uint4 b = slot[np * 32 + lane];
    o[2 * np].mma(a, b.x, b.y);
    o[2 * np + 1].mma(a, b.z, b.w);
  }
}

// Une tranche cachée : 8 étages de W1, la GELU, 4 étages de W2.
template <int Threads, Acc A>
__device__ __forceinline__ void chunk(rn::Frag<A> (&o)[kTiles], Ring& ring, int& q, std::uint32_t a_base,
                                      int lane) {
  rn::Frag<A> h[2 * kNP1];
#pragma unroll
  for (int nt = 0; nt < 2 * kNP1; ++nt) h[nt].clear();
#pragma unroll
  for (int i = 0; i < kW1Stages; ++i) {
    gemm1_stage<A>(h, a_base, acquire<Threads>(ring, q++), i, lane);
    if ((i & 1) != 0) {
#pragma unroll
      for (int nt = 0; nt < 2 * kNP1; ++nt) h[nt].promote();
    }
  }
  std::uint32_t a[kK2][4];
  gelu_to_a<A>(h, a);
#pragma unroll
  for (int j = 0; j < kK2; ++j) gemm2_stage<A>(o, a[j], acquire<Threads>(ring, q++), lane);
#pragma unroll
  for (int nt = 0; nt < kTiles; ++nt) o[nt].promote();
}

template <int Threads>
__device__ __forceinline__ void paired_gemm2(rn::Frag<Acc::kF32> (&o)[kTiles],
                                            const rn::Frag<Acc::kF32> (&h)[2 * kNP1],
                                            Ring& ring, int& q, int lane) {
#pragma unroll
  for (int j = 0; j < kK2; j += 2) {
    std::uint32_t a[4];
    gelu_stage(h, j, a);
    const auto* slot = acquire_wide<Threads>(ring, q++);
    gemm2_stage(o, a, slot, lane);
    gelu_stage(h, j + 1, a);
    gemm2_stage(o, a, slot + kStageVec, lane);
  }
}

// Deux tranches cachées partagent chaque ldmatrix de X. Les sommes de W1 restent indépendantes et W2
// consomme toujours les deux tranches de gauche à droite. Sa GELU libère H un pas K à la fois.
template <int Threads>
__device__ __forceinline__ void chunk_pair(rn::Frag<Acc::kF32> (&o)[kTiles], Ring& ring, int& q,
                                           std::uint32_t a_base, int lane) {
  rn::Frag<Acc::kF32> h0[2 * kNP1], h1[2 * kNP1];
#pragma unroll
  for (int nt = 0; nt < 2 * kNP1; ++nt) {
    h0[nt].clear();
    h1[nt].clear();
  }
#pragma unroll
  for (int i = 0; i < kW1Stages; ++i) {
    const auto* first = acquire_wide<Threads>(ring, q++);
    InputStage a;
    load_input_stage(a, a_base, i);
    gemm1_stage(h0, a, first, lane);
    gemm1_stage(h1, a, first + kStageVec, lane);
  }
  paired_gemm2<Threads>(o, h0, ring, q, lane);
  paired_gemm2<Threads>(o, h1, ring, q, lane);
}

// Somme des quatre fils d'une ligne (même g = lane / 4).
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffU, v, 1);
  return v + __shfl_xor_sync(0xffffffffU, v, 2);
}

// LayerNorm et résiduel des 16 lignes du warp, en place dans la tuile x (mémoire partagée), puis l'écriture
// des lignes entières. Moyenne, puis variance des écarts (deux passages sur les registres), en f32.
template <Acc A>
__device__ __forceinline__ void finish_rows(const rn::Frag<A> (&o)[kTiles], __half* s, const Params& p,
                                            long t0, int warp, int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float sum = 0.F;
#pragma unroll
    for (int nt = 0; nt < kTiles; ++nt) {
      const float2 v = o[nt].pair(h);
      sum += v.x + v.y;
    }
    const float mean = quad_sum(sum) * (1.F / kChannels);
    float sq = 0.F;
#pragma unroll
    for (int nt = 0; nt < kTiles; ++nt) {
      const float2 v = o[nt].pair(h);
      sq += (v.x - mean) * (v.x - mean) + (v.y - mean) * (v.y - mean);
    }
    const float r = rsqrtf(quad_sum(sq) * (1.F / kChannels) + p.eps);
    __half* row = s + (warp * kRows + g + 8 * h) * kLdx;
#pragma unroll
    for (int nt = 0; nt < kTiles; ++nt) {
      const int ch = nt * 8 + 2 * t;
      const float2 v = o[nt].pair(h);
      const float2 gm = __ldg(reinterpret_cast<const float2*>(p.gamma + ch));
      const float2 bt = __ldg(reinterpret_cast<const float2*>(p.beta + ch));
      auto* at = reinterpret_cast<__half2*>(row + ch);
      const float2 x = __half22float2(*at);
      *at = rn::to_half2(x.x + ((v.x - mean) * r * gm.x + bt.x), x.y + ((v.y - mean) * r * gm.y + bt.y));
    }
  }
  __syncwarp();
  constexpr int kChunks = kChannels / 8;
  for (int i = lane; i < kRows * kChunks; i += 32) {
    const int r = i / kChunks, c = i - r * kChunks;
    const long token = t0 + warp * kRows + r;
    if (token >= p.tokens) continue;
    *reinterpret_cast<uint4*>(p.out + token * kChannels + c * 8) =
        *reinterpret_cast<const uint4*>(s + (warp * kRows + r) * kLdx + c * 8);
  }
}

// La part blockIdx.y de O, en f32, vers partial (splits > 1).
template <Acc A>
__device__ __forceinline__ void spill_rows(const rn::Frag<A> (&o)[kTiles], const Params& p, long t0, int warp,
                                           int lane) {
  const int g = lane >> 2, t = lane & 3;
  float* base = p.partial + static_cast<long>(blockIdx.y) * p.tokens * kChannels + 2 * t;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const long token = t0 + warp * kRows + g + 8 * h;
    if (token >= p.tokens) continue;
#pragma unroll
    for (int nt = 0; nt < kTiles; ++nt)
      *reinterpret_cast<float2*>(base + token * kChannels + nt * 8) = o[nt].pair(h);
  }
}

// Une couche du MLP sur 16 · Warps jetons (blockIdx.x), pour les tranches cachées de la part blockIdx.y.
template <int Warps, Acc A, bool Paired = false>
__global__ void __launch_bounds__(Warps * 32, Warps == 4 ? 2 : 1) mlp_kernel(Params p) {
  static_assert(Warps == 4 || Warps == 8, "64 ou 128 jetons par bloc");
  using B = Block<Warps>;
  constexpr int kGroupsPerChunk = kChunkStages / (Paired ? 2 : 1);
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* s = reinterpret_cast<__half*>(smem_raw);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const long t0 = static_cast<long>(blockIdx.x) * B::kTokens;
  const int chunks = p.hidden / kHiddenTile;
  const int c0 = static_cast<int>(blockIdx.y) * chunks / p.splits;
  const int c1 = (static_cast<int>(blockIdx.y) + 1) * chunks / p.splits;

  load_tokens<B::kThreads, B::kTokens>(s, p, t0);
  cp_async_commit();
  Ring ring{.w = p.w,
            .slots = reinterpret_cast<uint4*>(smem_raw + B::kTileBytes),
            .next = c0 * kGroupsPerChunk,
            .end = c1 * kGroupsPerChunk};
  if constexpr (Paired) {
    issue_wide<B::kThreads>(ring);
  } else {
    issue<B::kThreads>(ring);
    issue<B::kThreads>(ring);
  }

  const std::uint32_t a_base =
      rn::smem_u32(s) +
      static_cast<std::uint32_t>(((warp * kRows + (lane & 15)) * kLdx + (lane >> 4) * 8) * 2);
  rn::Frag<A> o[kTiles];
#pragma unroll
  for (int nt = 0; nt < kTiles; ++nt) o[nt].clear();
  int q = c0 * kGroupsPerChunk;
  if constexpr (Paired) {
    static_assert(A == Acc::kF32);
#pragma unroll 1
    for (int c = c0; c < c1; c += 2) chunk_pair<B::kThreads>(o, ring, q, a_base, lane);
  } else {
#pragma unroll 1
    for (int c = c0; c < c1; ++c) chunk<B::kThreads, A>(o, ring, q, a_base, lane);
  }
  if (p.splits > 1) {
    spill_rows<A>(o, p, t0, warp, lane);
    return;
  }
  // La sortie de W2 est Half à l'entrée de la LayerNorm TensorRT, après l'accumulation entière.
  if constexpr (A != Acc::kF16) {
#pragma unroll
    for (auto& tile : o) {
#pragma unroll
      for (float& value : tile.r) value = __half2float(__float2half_rn(value));
    }
  }
  finish_rows<A>(o, s, p, t0, warp, lane);
}

// La fin d'une couche à couche cachée découpée : un warp par jeton, quatre canaux par fil. Somme des parts
// dans l'ordre, LayerNorm (deux passages), résiduel, arrondi FP16.
__global__ void __launch_bounds__(256) finish_kernel(Params p) {
  const long token = static_cast<long>(blockIdx.x) * 8 + (threadIdx.x >> 5);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (token >= p.tokens) return;
  const long at = token * kChannels + 4 * lane;
  float4 v = make_float4(0.F, 0.F, 0.F, 0.F);
  for (int k = 0; k < p.splits; ++k) {
    const float4 part = *reinterpret_cast<const float4*>(p.partial + k * p.tokens * kChannels + at);
    v.x += part.x, v.y += part.y, v.z += part.z, v.w += part.w;
  }
  v.x = __half2float(__float2half_rn(v.x));
  v.y = __half2float(__float2half_rn(v.y));
  v.z = __half2float(__float2half_rn(v.z));
  v.w = __half2float(__float2half_rn(v.w));
  float sum = v.x + v.y + v.z + v.w;
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) sum += __shfl_xor_sync(0xffffffffU, sum, m);
  const float mean = sum * (1.F / kChannels);
  const float dx = v.x - mean, dy = v.y - mean, dz = v.z - mean, dw = v.w - mean;
  float sq = dx * dx + dy * dy + dz * dz + dw * dw;
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) sq += __shfl_xor_sync(0xffffffffU, sq, m);
  const float r = rsqrtf(sq * (1.F / kChannels) + p.eps);
  const float4 gm = __ldg(reinterpret_cast<const float4*>(p.gamma) + lane);
  const float4 bt = __ldg(reinterpret_cast<const float4*>(p.beta) + lane);
  const uint2 xr = __ldg(reinterpret_cast<const uint2*>(p.x + at));
  const float2 x01 = rn::half_pair(xr.x), x23 = rn::half_pair(xr.y);
  uint2 w;
  w.x = pack_half2(x01.x + (dx * r * gm.x + bt.x), x01.y + (dy * r * gm.y + bt.y));
  w.y = pack_half2(x23.x + (dz * r * gm.z + bt.z), x23.y + (dw * r * gm.w + bt.w));
  *reinterpret_cast<uint2*>(p.out + at) = w;
}

}  // namespace kirari::gpu::cuda::gmflow_mlp
