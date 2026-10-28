#pragma once
// Prototype CUTLASS : fin de bloc SPAN fusionnée dans l'épilogue d'une convolution 3x3 64->64 FP8 E4M3.
//   c3  = Conv3x3(entrée FP8) + biais                      (accumulateur f32)
//   s   = c3 + saut x                                      (x en FP8, déquantifié par son échelle)
//   att = Tanh(Conv1x1(c3) + b1)                           (1x1 64->64 : mma f16 sur les registres de
//   l'accumulateur) out = s * att, requantifié en E4M3 par échelle statique
// La boucle principale est celle de CUTLASS (Mma de DefaultConv2dFprop) ; l'épilogue est écrit à la main
// parce que le bloc de sortie d'un warp contient les 64 canaux : l'accumulateur m16n8 d'une paire de tuiles n
// est, registre pour registre, l'opérande A d'un mma m16n8k16 f16 (même astuce que FlashAttention). Aucun
// aller-retour mémoire pour c3.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <functional>
#include <string>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/numeric_types.h"

namespace span {

struct TailParams {
  const uint8_t* skip;  // saut x, NHWC E4M3
  float skip_scale;     // échelle de déquantification du saut
  float alpha;          // échelle entrée*poids : accumulateur -> unités réelles de c3
  const float* b3;      // biais de la conv 3x3 (unités réelles)
  const __half* w1;     // poids 1x1 [sortie j][entrée k], FP16, 64x64
  const float* b1;      // biais de la 1x1
  uint8_t* out;         // sortie NHWC E4M3
  float inv_sy;         // 1 / échelle de requantification de la sortie
};

__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Mode 0 : conv + biais + Relu -> E4M3 (mesure du coût de la boucle principale seule dans ce cadre)
// Mode 1 : fin de bloc SPAN complète
template <class Base, int Mode>
__global__ void __launch_bounds__(Base::kThreadCount)
    span_tail_kernel(typename Base::Params params, TailParams tp) {
  using Mma = typename Base::Mma;
  constexpr int kWarpM = Mma::Policy::Operator::Shape::kM;  // lignes par warp
  constexpr int kMI = kWarpM / 16;
  static_assert(Mma::WarpCount::kN == 1 && Mma::WarpCount::kK == 1, "un warp doit porter les 64 canaux");
  static_assert(Mma::Shape::kN == 64, "le bloc doit couvrir les 64 canaux");
  constexpr int kLd = 72;  // pas de ligne de W1 en smem (en demi-flottants) : sans conflit de banques

  extern __shared__ __align__(16) unsigned char smem[];
  auto& ss = *reinterpret_cast<typename Mma::SharedStorage*>(smem);
  __half* w1s = reinterpret_cast<__half*>(smem + ((sizeof(typename Mma::SharedStorage) + 15) / 16) * 16);

  typename Base::ThreadblockSwizzle swizzle;
  cutlass::gemm::GemmCoord tile = swizzle.get_tile_offset(params.swizzle_log_tile);
  if (params.grid_tiled_shape.m() <= tile.m() || params.grid_tiled_shape.n() <= tile.n()) return;

  int thread_idx = threadIdx.x;
  if constexpr (Mode == 1) {
    for (int i = thread_idx; i < 64 * 8; i += Base::kThreadCount) {
      int r = i >> 3, c8 = i & 7;
      *reinterpret_cast<uint4*>(w1s + r * kLd + c8 * 8) =
          *reinterpret_cast<const uint4*>(tp.w1 + r * 64 + c8 * 8);
    }
  }
  __syncthreads();

  typename Mma::IteratorA iterator_A(
      params.iterator_A, params.problem_size, params.ptr_A, thread_idx,
      cutlass::MatrixCoord(tile.m() * Mma::Shape::kM, tile.k() * Mma::Shape::kK));
  typename Mma::IteratorB iterator_B(
      params.iterator_B, params.problem_size, params.ptr_B, thread_idx,
      cutlass::MatrixCoord(tile.k() * Mma::Shape::kK, tile.n() * Mma::Shape::kN));
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane = threadIdx.x % 32;
  Mma mma(ss, thread_idx, warp_idx, lane);
  typename Mma::FragmentC acc;
  acc.clear();
  mma(params.gemm_k_iterations, acc, iterator_A, iterator_B, acc, params.gemm_k_iterations_per_channel);

  // ---- épilogue en registres. Tuile d'accumulateur (m, n) = acc[(m + n*kMI)*4 + e] :
  //      e=0,1 -> ligne g, colonnes 2t, 2t+1 ; e=2,3 -> ligne g+8 (g = lane/4, t = lane%4), colonnes de la
  //      tuile n*8.
  const float* a = acc.data();
  const int g = lane >> 2, t = lane & 3;
  const int warp_m = warp_idx % Mma::WarpCount::kM;
  const long row_base = long(tile.m()) * Mma::Shape::kM + warp_m * kWarpM;

#pragma unroll
  for (int m = 0; m < kMI; ++m) {
    const long r0 = row_base + m * 16 + g, r1 = r0 + 8;
    float v[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      float2 bb = __ldg(reinterpret_cast<const float2*>(tp.b3 + n * 8 + 2 * t));
      v[n][0] = a[(m + n * kMI) * 4 + 0] * tp.alpha + bb.x;
      v[n][1] = a[(m + n * kMI) * 4 + 1] * tp.alpha + bb.y;
      v[n][2] = a[(m + n * kMI) * 4 + 2] * tp.alpha + bb.x;
      v[n][3] = a[(m + n * kMI) * 4 + 3] * tp.alpha + bb.y;
    }
    if constexpr (Mode == 0) {
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        float2 p0 = make_float2(fmaxf(v[n][0], 0.f) * tp.inv_sy, fmaxf(v[n][1], 0.f) * tp.inv_sy);
        float2 p1 = make_float2(fmaxf(v[n][2], 0.f) * tp.inv_sy, fmaxf(v[n][3], 0.f) * tp.inv_sy);
        *reinterpret_cast<uint16_t*>(tp.out + r0 * 64 + n * 8 + 2 * t) =
            __nv_cvt_float2_to_fp8x2(p0, __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<uint16_t*>(tp.out + r1 * 64 + n * 8 + 2 * t) =
            __nv_cvt_float2_to_fp8x2(p1, __NV_SATFINITE, __NV_E4M3);
      }
    } else {
      // saut x : chargé tôt pour cacher la latence derrière les mma de la 1x1
      uint16_t sk[8][2];
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        sk[n][0] = *reinterpret_cast<const uint16_t*>(tp.skip + r0 * 64 + n * 8 + 2 * t);
        sk[n][1] = *reinterpret_cast<const uint16_t*>(tp.skip + r1 * 64 + n * 8 + 2 * t);
      }
      // opérande A (c3 en FP16) : 4 pas de k=16, chacun formé de deux tuiles n adjacentes
      uint32_t A[4][4];
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        A[ks][0] = pack_half2(v[2 * ks][0], v[2 * ks][1]);
        A[ks][1] = pack_half2(v[2 * ks][2], v[2 * ks][3]);
        A[ks][2] = pack_half2(v[2 * ks + 1][0], v[2 * ks + 1][1]);
        A[ks][3] = pack_half2(v[2 * ks + 1][2], v[2 * ks + 1][3]);
      }
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          const __half* wp = w1s + (nt * 8 + g) * kLd + ks * 16 + 2 * t;
          mma_f16(d, A[ks], *reinterpret_cast<const uint32_t*>(wp),
                  *reinterpret_cast<const uint32_t*>(wp + 8));
        }
        float2 bb1 = __ldg(reinterpret_cast<const float2*>(tp.b1 + nt * 8 + 2 * t));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          __nv_fp8x2_e4m3 q;
          q.__x = sk[nt][h];
          float2 skf = float2(q);
          float s0 = v[nt][2 * h] + skf.x * tp.skip_scale, s1 = v[nt][2 * h + 1] + skf.y * tp.skip_scale;
          float o0 = s0 * tanh_approx(d[2 * h] + bb1.x) * tp.inv_sy;
          float o1 = s1 * tanh_approx(d[2 * h + 1] + bb1.y) * tp.inv_sy;
          *reinterpret_cast<uint16_t*>(tp.out + (h ? r1 : r0) * 64 + nt * 8 + 2 * t) =
              __nv_cvt_float2_to_fp8x2(make_float2(o0, o1), __NV_SATFINITE, __NV_E4M3);
        }
      }
    }
  }
}

template <int TM, int TK, int WM, int Stages>
struct TailCfg {
  using E = cutlass::float_e4m3_t;
  using Dummy =
      cutlass::epilogue::thread::LinearCombination<E, 8, float, float>;  // jamais exécuté : types de Params
  using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
      E, cutlass::layout::TensorNHWC, E, cutlass::layout::TensorNHWC, E, cutlass::layout::TensorNHWC, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89, cutlass::gemm::GemmShape<TM, 64, TK>,
      cutlass::gemm::GemmShape<WM, 64, TK>, cutlass::gemm::GemmShape<16, 8, 32>, Dummy,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
};

// Lancement d'un noyau : construit les Params CUTLASS une fois, puis lance (compatible capture de graphe
// CUDA).
template <class C, int Mode>
struct Launcher {
  using Base = typename C::Base;
  using E = typename C::E;
  typename Base::Params params;
  TailParams tp;
  dim3 grid, block;
  size_t smem = 0;

  void init(int H, int W, const E* in, const E* w, E* dummy_out, const TailParams& t) {
    cutlass::conv::Conv2dProblemSize ps({1, H, W, 64}, {64, 3, 3, 64}, {1, 1, 1, 1}, {1, 1}, {1, 1},
                                        cutlass::conv::Mode::kCrossCorrelation, 1);
    typename Base::Arguments args(ps, {const_cast<E*>(in), {64, 64 * W, 64 * W * H}},
                                  {const_cast<E*>(w), {64, 64 * 3, 64 * 3 * 3}},
                                  {dummy_out, {64, 64 * W, 64 * W * H}},
                                  {dummy_out, {64, 64 * W, 64 * W * H}}, typename C::Dummy::Params(1.f, 0.f));
    params = typename Base::Params(args, nullptr);
    tp = t;
    grid = typename Base::ThreadblockSwizzle().get_grid_shape(params.grid_tiled_shape);
    block = dim3(Base::kThreadCount);
    smem = ((sizeof(typename Base::Mma::SharedStorage) + 15) / 16) * 16 + 64 * 72 * 2;
    if (smem > 48 * 1024)
      cudaFuncSetAttribute(span_tail_kernel<Base, Mode>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           int(smem));
  }
  void launch(cudaStream_t s) const { span_tail_kernel<Base, Mode><<<grid, block, smem, s>>>(params, tp); }
};

}  // namespace span
