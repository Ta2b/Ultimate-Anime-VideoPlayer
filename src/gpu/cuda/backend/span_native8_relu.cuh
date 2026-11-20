#pragma once
// Les 3×3 C48 et C64 avec ReLU du SPAN FP8 natif (c1/c2). Le K aplati de la convolution est
// celui du kFewChannels de CUTLASS. L'épilogue partage l'écriture E4M3 et FP16 du lecteur.
#include <cstdint>
#include <vector>

#include "gpu/cuda/backend/span_fp8_kernels.cuh"
#include "gpu/cuda/backend/span_native16_kernels.cuh"

namespace kirari::gpu::cuda::span_native8 {

namespace sk = span_fp8_kernels;
namespace sn = span_native;
using Byte = std::uint8_t;
using U32 = std::uint32_t;
inline constexpr int kSW = sn::kTW + 2;
inline constexpr int kPixels = (sn::kTH + 2) * kSW;

template <int C>
struct Tile {
  static_assert(C == 48 || C == 64);
  static constexpr int kK = 9 * C;
  static constexpr int kSteps = (kK + 31) / 32;
  static constexpr int kChunks = C / 16;
  // Les pixels successifs avancent d'un nombre impair de secteurs de 16 octets en mémoire partagée.
  static constexpr int kStride = C == 64 ? C + 16 : C;
};

// Les deux fragments B de chaque paire n8, mots de quatre E4M3 dans l'ordre des canaux.
template <int C>
std::vector<U32> pack(const std::vector<Byte>& weights) {
  constexpr int kSteps = Tile<C>::kSteps, kPairs = Tile<C>::kChunks, kK = Tile<C>::kK;
  std::vector<U32> out;
  out.reserve(kSteps * kPairs * 32 * 4);
  for (int step = 0; step < kSteps; ++step)
    for (int np = 0; np < kPairs; ++np)
      for (int lane = 0; lane < 32; ++lane)
        for (int tile = 0; tile < 2; ++tile) {
          const int n = 8 * (2 * np + tile) + lane / 4;
          for (int half = 0; half < 2; ++half) {
            const int k = step * 32 + half * 16 + 4 * (lane % 4);
            U32 word = 0;
            for (int i = 0; i < 4; ++i)
              if (k + i < kK) word |= U32{weights[n * kK + k + i]} << (8 * i);
            out.push_back(word);
          }
        }
  return out;
}

__device__ __forceinline__ void mma(sn::Frag<sn::Acc::kF16>& frag, const U32 (&a)[4], U32 b0, U32 b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
      : "+r"(frag.r[0]), "+r"(frag.r[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma(sn::Frag<sn::Acc::kF32>& frag, const U32 (&a)[4], U32 b0, U32 b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(frag.r[0]), "+f"(frag.r[1]), "+f"(frag.r[2]), "+f"(frag.r[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void add_stage(sn::Frag<sn::Acc::kF16>& sum,
                                          const sn::Frag<sn::Acc::kF16>& stage) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const __half2 a = *reinterpret_cast<const __half2*>(&sum.r[i]);
    const __half2 b = *reinterpret_cast<const __half2*>(&stage.r[i]);
    const __half2 result = __hadd2(a, b);
    sum.r[i] = *reinterpret_cast<const U32*>(&result);
  }
}

__device__ __forceinline__ void add_stage(sn::Frag<sn::Acc::kF32>& sum,
                                          const sn::Frag<sn::Acc::kF32>& stage) {
#pragma unroll
  for (int i = 0; i < 4; ++i) sum.r[i] = __fadd_rn(sum.r[i], stage.r[i]);
}

struct Params {
  const Byte* input;
  const uint4* packed;
  sk::KP epilogue;
  int height, width;
};

// Les quatre fragments A pour un MMA. La moitié au-delà de K a des poids nuls ; son adresse reste chargée.
template <int C>
__device__ __forceinline__ void load_a(U32 (&a)[sn::kMF][4], U32 tile, int row, int lane, int step) {
  const int half = step * 32 + (lane >> 4) * 16;
  const int k = half < Tile<C>::kK ? half : Tile<C>::kK - C;
  const int tap = k / C, c = k % C;
  const int from = row + (tap / 3) * kSW + tap % 3;
#pragma unroll
  for (int f = 0; f < sn::kMF; ++f)
    sn::ldmatrix4(a[f],
                  tile + static_cast<U32>((from + (f >> 1) * kSW + 16 * (f & 1)) * Tile<C>::kStride + c));
}

template <int C, bool Pair, sn::Acc A>
__device__ __forceinline__ void mma_group(sn::Frag<A> (&sum)[sn::kMF][C / 8], U32 tile, const uint4* weights,
                                          int row, int lane, int step) {
  constexpr int kPairs = Tile<C>::kChunks;
  U32 a0[sn::kMF][4], a1[sn::kMF][4];
  load_a<C>(a0, tile, row, lane, step);
  if constexpr (Pair) load_a<C>(a1, tile, row, lane, step + 1);
#pragma unroll
  for (int np = 0; np < kPairs; ++np) {
    const auto* w = weights + (step * kPairs + np) * 32 + lane;
    const uint4 b0 = __ldg(w);
    uint4 b1{};
    if constexpr (Pair) b1 = __ldg(w + kPairs * 32);
#pragma unroll
    for (int f = 0; f < sn::kMF; ++f) {
      sn::Frag<A> left, right;
      left.clear();
      right.clear();
      mma(left, a0[f], b0.x, b0.y);
      mma(right, a0[f], b0.z, b0.w);
      if constexpr (Pair) {
        mma(left, a1[f], b1.x, b1.y);
        mma(right, a1[f], b1.z, b1.w);
      }
      add_stage(sum[f][2 * np], left);
      add_stage(sum[f][2 * np + 1], right);
    }
  }
}

template <int C, sn::Acc A>
__global__ void __launch_bounds__(sn::kThreads) relu_conv(Params p) {
  constexpr int kChunks = Tile<C>::kChunks, kStride = Tile<C>::kStride;
  __shared__ __align__(16) Byte tile[kPixels * kStride];
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int y0 = static_cast<int>(blockIdx.y) * sn::kTH, x0 = static_cast<int>(blockIdx.x) * sn::kTW;
  const U32 base = sn::smem_u32(tile);
  for (int i = tid; i < kPixels * kChunks; i += sn::kThreads) {
    const int pixel = i / kChunks, chunk = i % kChunks;
    const int y = y0 - 1 + pixel / kSW, x = x0 - 1 + pixel % kSW;
    const bool valid = y >= 0 && y < p.height && x >= 0 && x < p.width;
    const auto* from = valid ? p.input + (static_cast<long>(y) * p.width + x) * C + chunk * 16 : p.input;
    sn::cp_async16(base + static_cast<U32>(pixel * kStride + chunk * 16), from, valid);
  }
  sn::cp_async_wait_all();
  __syncthreads();

  sn::Frag<A> sum[sn::kMF][C / 8];
#pragma unroll
  for (int f = 0; f < sn::kMF; ++f)
#pragma unroll
    for (int n = 0; n < C / 8; ++n) sum[f][n].clear();
  const int row = warp * sn::kRows * kSW + (lane & 15);
  // CUTLASS TK64 verse le temporaire au premier MMA de chaque tuile K, et au dernier MMA du calcul.
  // Les groupes sont [0], [1,2], …, [dernier pas], avec addition au type de l'accumulateur.
  mma_group<C, false>(sum, base, p.packed, row, lane, 0);
#pragma unroll 1
  for (int step = 1; step + 1 < Tile<C>::kSteps; step += 2)
    mma_group<C, true>(sum, base, p.packed, row, lane, step);
  mma_group<C, false>(sum, base, p.packed, row, lane, Tile<C>::kSteps - 1);
  bool overflow = false;
#pragma unroll
  for (int n = 0; n < C / 8; ++n) {
    const int c = n * 8 + 2 * (lane & 3);
    const float2 b = __ldg(reinterpret_cast<const float2*>(p.epilogue.bias + c));
    const float2 s = __ldg(reinterpret_cast<const float2*>(p.epilogue.alpha_col + c));
#pragma unroll
    for (int f = 0; f < sn::kMF; ++f) {
      const int y = y0 + warp * sn::kRows + (f >> 1);
      const int x0f = x0 + 16 * (f & 1) + (lane >> 2);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float2 v = sn::sums(sum[f][n], h, overflow);
        const int x = x0f + 8 * h;
        if (y < p.height && x < p.width)
          sk::store_pair<false>(p.epilogue, C, static_cast<long>(y) * p.width + x, c,
                                fmaxf(fmaf(v.x, s.x, b.x), 0.F), fmaxf(fmaf(v.y, s.y, b.y), 0.F));
      }
    }
  }
  if constexpr (A == sn::Acc::kF16) sk::note_overflow(p.epilogue, overflow);
}

template <int Channels>
void launch_relu(cudaStream_t stream, const Params& p, bool f16) {
  const dim3 grid(static_cast<unsigned>((p.width + sn::kTW - 1) / sn::kTW),
                  static_cast<unsigned>((p.height + sn::kTH - 1) / sn::kTH));
  if (f16)
    relu_conv<Channels, sn::Acc::kF16><<<grid, sn::kThreads, 0, stream>>>(p);
  else
    relu_conv<Channels, sn::Acc::kF32><<<grid, sn::kThreads, 0, stream>>>(p);
}

}  // namespace kirari::gpu::cuda::span_native8
