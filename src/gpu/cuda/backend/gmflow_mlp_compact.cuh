// MLP au quart sur Ada : chaque warp garde ses 16 jetons. La dernière couche accumule en FP32.
#pragma once
#include "gpu/cuda/backend/gmflow_mlp_kernels.cuh"

namespace kirari::gpu::cuda::gmflow_mlp {

using CompactBlock = Block<6>;
inline constexpr int kCompactHidden = 1024;
inline constexpr int kChunks = kChannels / 8;
inline constexpr int kStagesPerPlane = kW1Stages / 2;
using MessageFragments = InputStage[kStagesPerPlane];

inline constexpr int kCompactSmem = CompactBlock::kTokens * kChannels * 2 + 4 * kStageVec * 16;

template <int Threads, int Rows>
__device__ __forceinline__ void load_residual(__half* s, const Params& p, long first) {
  for (int i = static_cast<int>(threadIdx.x); i < Rows * kChunks; i += Threads) {
    const int row = i / kChunks, ch = (i % kChunks) * 8;
    const bool valid = first + row < p.tokens;
    const auto* ptr = valid ? p.x + (first + row) * kChannels + ch : p.x;
    rn::cp_async16(rn::smem_u32(residual_at<true>(s, row, ch)), ptr, valid);
  }
}

__device__ __forceinline__ void residual_stage(InputStage& a, __half* s, int warp, int lane, int i) {
  const int row = warp * kRows + (lane & 15), chunk = (lane >> 4) * 8;
#pragma unroll
  for (int kl = 0; kl < 2; ++kl) {
    const int ch = (2 * i + kl) * 16 + chunk;
    rn::ldmatrix4(a[kl], rn::smem_u32(residual_at<true>(s, row, ch)));
  }
}

// Même fragment A m16n8k16 que ldmatrix, lu directement puis conservé pendant toute la couche.
__device__ __forceinline__ void message_stage(InputStage& a, const Params& p, long first, int warp, int lane,
                                              int i) {
  const long row = first + warp * kRows + (lane >> 2);
  const int ch0 = i * 32 + 2 * (lane & 3);
#pragma unroll
  for (int kl = 0; kl < 2; ++kl) {
    const int ch = ch0 + 16 * kl;
#pragma unroll
    for (int hi = 0; hi < 2; ++hi) {
      const long t = row + 8 * hi;
      const auto* from = t < p.tokens ? p.message + t * kChannels + ch : p.message;
      a[kl][hi] = t < p.tokens ? __ldg(reinterpret_cast<const std::uint32_t*>(from)) : 0U;
      a[kl][2 + hi] = t < p.tokens ? __ldg(reinterpret_cast<const std::uint32_t*>(from + 8)) : 0U;
    }
  }
}

template <int Threads, Acc A>
__device__ __forceinline__ void compact_chunk(rn::Frag<A> (&o)[kTiles], Ring& ring, int& q, __half* s,
                                              int warp, int lane, const MessageFragments& cached) {
  rn::Frag<A> h0[2 * kNP1], h1[2 * kNP1];
#pragma unroll
  for (int nt = 0; nt < 2 * kNP1; ++nt) {
    h0[nt].clear();
    h1[nt].clear();
  }
#pragma unroll 1
  for (int i = 0; i < kStagesPerPlane; ++i) {
    const auto* slot = acquire_wide<Threads>(ring, q++);
    InputStage a;
    residual_stage(a, s, warp, lane, i);
    gemm1_stage(h0, a, slot, lane);
    gemm1_stage(h1, a, slot + kStageVec, lane);
  }
#pragma unroll
  for (int i = 0; i < kStagesPerPlane; ++i) {
    const auto* slot = acquire_wide<Threads>(ring, q++);
    gemm1_stage(h0, cached[i], slot, lane);
    gemm1_stage(h1, cached[i], slot + kStageVec, lane);
  }
  paired_gemm2<Threads>(o, h0, ring, q, lane);
  paired_gemm2<Threads>(o, h1, ring, q, lane);
}

// Uniquement pour 1 024 canaux cachés, en une part et avec les sommes FP16.
__global__ void __launch_bounds__(CompactBlock::kThreads, 2) mlp_compact96(Params p) {
  using B = CompactBlock;
  constexpr int kGroupsPerChunk = kChunkStages / 2;
  constexpr int kChunks = kCompactHidden / kHiddenTile;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* s = reinterpret_cast<__half*>(smem_raw);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const long t0 = static_cast<long>(blockIdx.x) * B::kTokens;

  load_residual<B::kThreads, B::kTokens>(s, p, t0);
  cp_async_commit();
  Ring ring{.w = p.w,
            .slots = reinterpret_cast<uint4*>(s + B::kTokens * kChannels),
            .next = 0,
            .end = kChunks * kGroupsPerChunk};
  issue_wide<B::kThreads>(ring);
  MessageFragments cached{};
#pragma unroll
  for (int i = 0; i < kStagesPerPlane; ++i) message_stage(cached[i], p, t0, warp, lane, i);
  rn::Frag<Acc::kF16> o[kTiles];
#pragma unroll
  for (auto& tile : o) tile.clear();
  int q = 0;
#pragma unroll 1
  for (int c = 0; c < kChunks; c += 2) compact_chunk<B::kThreads>(o, ring, q, s, warp, lane, cached);
  finish_rows<Acc::kF16, true>(o, s, p, t0, warp, lane);
}

using StrictFrag = rn::Frag<Acc::kF32>;
inline constexpr int kCompactStrictSmem = Block<4>::kTokens * kChannels * 2 + 4 * kStageVec * 16;

// Une seule tranche cachée en vol. Le flux non apparié met deux étages W1 successifs dans une case large.
__device__ __forceinline__ void compact_strict_chunk(StrictFrag (&o)[kTiles], Ring& ring, int& q, __half* s,
                                                     int warp, int lane, const MessageFragments& cached) {
  constexpr int kThreads = Block<4>::kThreads;
  StrictFrag h[2 * kNP1];
#pragma unroll
  for (auto& tile : h) tile.clear();
#pragma unroll 1
  for (int i = 0; i < kStagesPerPlane; i += 2) {
    const auto* slot = acquire_wide<kThreads>(ring, q++);
    InputStage a;
    residual_stage(a, s, warp, lane, i);
    gemm1_stage(h, a, slot, lane);
    residual_stage(a, s, warp, lane, i + 1);
    gemm1_stage(h, a, slot + kStageVec, lane);
  }
#pragma unroll
  for (int i = 0; i < kStagesPerPlane; i += 2) {
    const auto* slot = acquire_wide<kThreads>(ring, q++);
    gemm1_stage(h, cached[i], slot, lane);
    gemm1_stage(h, cached[i + 1], slot + kStageVec, lane);
  }
  paired_gemm2<kThreads>(o, h, ring, q, lane);
}

// Seule Ada emprunte ce chemin. Les autres binaires compilent sans contraindre les registres à trois CTA.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 890
inline constexpr int kCompactStrictBlocks = 2;
#else
inline constexpr int kCompactStrictBlocks = 3;
#endif

// Dernier MLP au quart, 1 024 canaux cachés et une seule part.
__global__ void __launch_bounds__(Block<4>::kThreads, kCompactStrictBlocks) mlp_compact_strict64(Params p) {
  using B = Block<4>;
  constexpr int kHiddenChunks = kCompactHidden / kHiddenTile;
  constexpr int kGroupsPerChunk = kChunkStages / 2;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* s = reinterpret_cast<__half*>(smem_raw);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const long t0 = static_cast<long>(blockIdx.x) * B::kTokens;

  load_residual<B::kThreads, B::kTokens>(s, p, t0);
  cp_async_commit();
  Ring ring{.w = p.w,
            .slots = reinterpret_cast<uint4*>(s + B::kTokens * kChannels),
            .next = 0,
            .end = kHiddenChunks * kGroupsPerChunk};
  issue_wide<B::kThreads>(ring);
  MessageFragments cached{};
#pragma unroll
  for (int i = 0; i < kStagesPerPlane; ++i) message_stage(cached[i], p, t0, warp, lane, i);
  StrictFrag o[kTiles];
#pragma unroll
  for (auto& tile : o) tile.clear();
  int q = 0;
#pragma unroll 1
  for (int c = 0; c < kHiddenChunks; ++c) compact_strict_chunk(o, ring, q, s, warp, lane, cached);
  round_mlp_output(o);
  finish_rows<Acc::kF32, true>(o, s, p, t0, warp, lane);
}

}  // namespace kirari::gpu::cuda::gmflow_mlp
