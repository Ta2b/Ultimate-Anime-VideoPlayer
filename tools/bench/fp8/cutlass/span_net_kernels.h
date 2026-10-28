#pragma once
// Noyaux du SPAN entier (version généralisée de span_kernels.h) : une seule boucle principale CUTLASS
// (convolution implicit GEMM, entrée FP8 E4M3 ou FP16) et un épilogue écrit à la main qui sait :
//   Mode 0 : biais (+ Relu) puis écriture E4M3 (sortie FP8, échelle statique) et/ou FP16 (tranche d'un tampon
//            de concaténation : pas de ligne ld16 en éléments)
//   Mode 1 : fin de bloc SPAN (saut x + Tanh(conv 1x1 de c3) + produit), mêmes sorties
// Le warp porte les colonnes de sortie complètes (jusqu'à 64) : l'accumulateur m16n8 de deux tuiles voisines
// est l'opérande A d'un mma m16n8k16 f16 pour la conv 1x1.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/numeric_types.h"

namespace net {

struct KP {
  const uint8_t* skip = nullptr;  // saut x, NHWC E4M3 (mode 1)
  float skip_scale = 0.f;
  float alpha = 1.f;            // accumulateur -> unités réelles
  const float* bias = nullptr;  // biais de la conv 3x3 (unités réelles)
  const __half* w1 = nullptr;   // poids 1x1 [sortie][entrée], FP16, 64x64 (mode 1)
  const float* b1 = nullptr;    // biais de la 1x1 (mode 1)
  uint8_t* out8 = nullptr;      // sortie E4M3 NHWC (64 canaux) ou nullptr
  float inv_s8 = 1.f;           // 1 / échelle de la sortie E4M3
  __half* out16 = nullptr;      // sortie FP16 (déjà décalée à la tranche de canaux) ou nullptr
  int ld16 = 0;                 // pas de ligne de out16, en éléments
  int ncols = 64;               // colonnes de sortie utiles (64 ou 48)
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

// Écrit une paire de canaux (2t, 2t+1 de la tuile n) d'une ligne, dans les sorties demandées.
__device__ __forceinline__ void store_pair(const KP& p, long row, int col, float x0, float x1) {
  if (p.out8)
    *reinterpret_cast<uint16_t*>(p.out8 + row * 64 + col) =
        __nv_cvt_float2_to_fp8x2(make_float2(x0 * p.inv_s8, x1 * p.inv_s8), __NV_SATFINITE, __NV_E4M3);
  if (p.out16) *reinterpret_cast<__half2*>(p.out16 + row * p.ld16 + col) = __floats2half2_rn(x0, x1);
}

template <class Base, int Mode, bool Relu>
__global__ void __launch_bounds__(Base::kThreadCount) net_kernel(typename Base::Params params, KP tp) {
  using Mma = typename Base::Mma;
  constexpr int kWarpM = Mma::Policy::Operator::Shape::kM;
  constexpr int kMI = kWarpM / 16;
  static_assert(Mma::WarpCount::kN == 1 && Mma::WarpCount::kK == 1,
                "un warp doit porter toutes les colonnes");
  static_assert(Mma::Shape::kN == 64, "le bloc doit couvrir 64 colonnes");
  constexpr int kLd = 72;  // pas de ligne de W1 en smem (demi-flottants) : sans conflit de banques

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

  // Tuile d'accumulateur (m, n) = acc[(m + n*kMI)*4 + e] : e=0,1 -> ligne g, colonnes 2t,2t+1 ; e=2,3 -> g+8.
  const auto* a = acc.data();
  const int g = lane >> 2, t = lane & 3;
  const int warp_m = warp_idx % Mma::WarpCount::kM;
  const long row_base = long(tile.m()) * Mma::Shape::kM + warp_m * kWarpM;
  const int nt_used = tp.ncols >> 3;

#pragma unroll
  for (int m = 0; m < kMI; ++m) {
    const long r0 = row_base + m * 16 + g, r1 = r0 + 8;
    float v[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      float2 bb = n < nt_used ? __ldg(reinterpret_cast<const float2*>(tp.bias + n * 8 + 2 * t))
                              : make_float2(0.f, 0.f);
      v[n][0] = float(a[(m + n * kMI) * 4 + 0]) * tp.alpha + bb.x;
      v[n][1] = float(a[(m + n * kMI) * 4 + 1]) * tp.alpha + bb.y;
      v[n][2] = float(a[(m + n * kMI) * 4 + 2]) * tp.alpha + bb.x;
      v[n][3] = float(a[(m + n * kMI) * 4 + 3]) * tp.alpha + bb.y;
    }
    if constexpr (Mode == 0) {
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        if (n >= nt_used) continue;
        if constexpr (Relu) {
          v[n][0] = fmaxf(v[n][0], 0.f);
          v[n][1] = fmaxf(v[n][1], 0.f);
          v[n][2] = fmaxf(v[n][2], 0.f);
          v[n][3] = fmaxf(v[n][3], 0.f);
        }
        store_pair(tp, r0, n * 8 + 2 * t, v[n][0], v[n][1]);
        store_pair(tp, r1, n * 8 + 2 * t, v[n][2], v[n][3]);
      }
    } else {
      uint16_t sk[8][2];
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        sk[n][0] = *reinterpret_cast<const uint16_t*>(tp.skip + r0 * 64 + n * 8 + 2 * t);
        sk[n][1] = *reinterpret_cast<const uint16_t*>(tp.skip + r1 * 64 + n * 8 + 2 * t);
      }
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
          float o0 = s0 * tanh_approx(d[2 * h] + bb1.x);
          float o1 = s1 * tanh_approx(d[2 * h + 1] + bb1.y);
          store_pair(tp, h ? r1 : r0, nt * 8 + 2 * t, o0, o1);
        }
      }
    }
  }
}

// EIn : float_e4m3_t ou half_t. Acc : float ou half_t (accumulateur du mma).
template <class EIn, int TM, int TK, int WM, int Stages, class Acc = float,
          cutlass::conv::IteratorAlgorithm Alg = cutlass::conv::IteratorAlgorithm::kOptimized>
struct Cfg {
  using E = EIn;
  static constexpr bool kFp8 = cutlass::sizeof_bits<EIn>::value == 8;
  using Inst = typename std::conditional<kFp8, cutlass::gemm::GemmShape<16, 8, 32>,
                                         cutlass::gemm::GemmShape<16, 8, 16>>::type;
  using Dummy = cutlass::epilogue::thread::LinearCombination<EIn, 8, Acc, float>;  // jamais exécuté
  using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
      EIn, cutlass::layout::TensorNHWC, EIn, cutlass::layout::TensorNHWC, EIn, cutlass::layout::TensorNHWC,
      Acc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89, cutlass::gemm::GemmShape<TM, 64, TK>,
      cutlass::gemm::GemmShape<WM, 64, TK>, Inst, Dummy,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, cutlass::arch::OpMultiplyAdd,
      Alg>::Kernel;
};

// Un lancement : paramètres CUTLASS construits une fois, lancement compatible avec la capture de graphe.
template <class C, int Mode, bool Relu>
struct Launcher {
  using Base = typename C::Base;
  using E = typename C::E;
  typename Base::Params params;
  KP kp;
  dim3 grid, block;
  size_t smem = 0;

  // Convolution RxR (remplissage pad), Cin canaux d'entrée, Kout canaux de sortie, image HxW.
  void init(int H, int W, int R, int pad, int Cin, int Kout, const E* in, const E* w, const KP& k) {
    cutlass::conv::Conv2dProblemSize ps({1, H, W, Cin}, {Kout, R, R, Cin}, {pad, pad, pad, pad}, {1, 1},
                                        {1, 1}, cutlass::conv::Mode::kCrossCorrelation, 1);
    E* dummy = reinterpret_cast<E*>(0x1000);  // jamais déréférencé : l'épilogue standard n'est pas exécuté
    typename Base::Arguments args(
        ps, {const_cast<E*>(in), {Cin, Cin * W, Cin * W * H}},
        {const_cast<E*>(w), {Cin, Cin * R, Cin * R * R}}, {dummy, {Kout, Kout * W, Kout * W * H}},
        {dummy, {Kout, Kout * W, Kout * W * H}}, typename C::Dummy::Params(1.f, 0.f));
    params = typename Base::Params(args, nullptr);
    kp = k;
    kp.ncols = Kout;
    grid = typename Base::ThreadblockSwizzle().get_grid_shape(params.grid_tiled_shape);
    block = dim3(Base::kThreadCount);
    smem = ((sizeof(typename Base::Mma::SharedStorage) + 15) / 16) * 16 + 64 * 72 * 2;
    if (smem > 48 * 1024)
      cudaFuncSetAttribute(net_kernel<Base, Mode, Relu>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           int(smem));
  }
  void launch(cudaStream_t s) const { net_kernel<Base, Mode, Relu><<<grid, block, smem, s>>>(params, kp); }
};

}  // namespace net
