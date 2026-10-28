#pragma once
// Noyau conv 3x3 FP8 E4M3 pour RIFE v4.25 (IFNet) : boucle principale CUTLASS (DefaultConv2dFprop,
// implicit GEMM) et épilogue écrit à la main, par registres de l'accumulateur m16n8, qui fait la couche
// résiduelle « ResConv » de l'IFNet en une passe :
//     y = LeakyRelu( beta * (conv(x) + biais) + résiduel )         (pente 0,2 ; beta par canal)
// et écrit y en FP8 E4M3 (entrée de la couche suivante, échelle statique) et/ou en FP16 (flot résiduel,
// non quantifié : l'erreur ne s'accumule pas sur les 8 couches). Sans résiduel/beta : biais + LeakyRelu seul
// (convs strided d'entrée du bloc). Le bloc de sortie du CTA couvre TN canaux (32 ou 64) ; le canal de départ
// est tile.n * TN, donc Cout = 96/128/192 se fait avec 3/2/3 colonnes de CTA.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/numeric_types.h"

namespace rife {

struct KP {
  float alpha = 1.f;            // accumulateur -> unités réelles (échelle entrée * échelle poids)
  const float* bias = nullptr;  // biais (unités réelles), Cout
  const float* beta = nullptr;  // bêta par canal (nullptr : 1)
  const __half* res = nullptr;  // résiduel FP16 NHWC (pas de ligne = Cout) ou nullptr
  float slope = 0.2f;           // pente du LeakyRelu (1,0 : pas d'activation)
  uint8_t* out8 = nullptr;      // sortie E4M3 NHWC ou nullptr
  float inv_s8 = 1.f;           // 1 / échelle de la sortie E4M3
  __half* out16 = nullptr;      // sortie FP16 NHWC ou nullptr
  int cout = 0;                 // canaux de sortie
  long m = 0;                   // lignes de sortie (N*P*Q)
};

template <class Base, int TN>
__global__ void __launch_bounds__(Base::kThreadCount) rife_conv_kernel(typename Base::Params params, KP tp) {
  using Mma = typename Base::Mma;
  constexpr int kWarpM = Mma::Policy::Operator::Shape::kM;
  constexpr int kMI = kWarpM / 16;
  constexpr int kNT = TN / 8;
  static_assert(Mma::WarpCount::kN == 1 && Mma::WarpCount::kK == 1,
                "un warp doit porter toutes les colonnes");
  static_assert(Mma::Shape::kN == TN, "le bloc doit couvrir TN colonnes");

  extern __shared__ __align__(16) unsigned char smem[];
  auto& ss = *reinterpret_cast<typename Mma::SharedStorage*>(smem);

  typename Base::ThreadblockSwizzle swizzle;
  cutlass::gemm::GemmCoord tile = swizzle.get_tile_offset(params.swizzle_log_tile);
  if (params.grid_tiled_shape.m() <= tile.m() || params.grid_tiled_shape.n() <= tile.n()) return;

  int thread_idx = threadIdx.x;
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
  const int col_base = tile.n() * TN;

#pragma unroll
  for (int m = 0; m < kMI; ++m) {
    const long r0 = row_base + m * 16 + g, r1 = r0 + 8;
#pragma unroll
    for (int n = 0; n < kNT; ++n) {
      const int c = col_base + n * 8 + 2 * t;
      if (c >= tp.cout) continue;
      const float2 bb = __ldg(reinterpret_cast<const float2*>(tp.bias + c));
      float2 be = make_float2(1.f, 1.f);
      if (tp.beta) be = __ldg(reinterpret_cast<const float2*>(tp.beta + c));
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const long r = h ? r1 : r0;
        if (r >= tp.m) continue;
        float x0 = (float(a[(m + n * kMI) * 4 + 2 * h]) * tp.alpha + bb.x) * be.x;
        float x1 = (float(a[(m + n * kMI) * 4 + 2 * h + 1]) * tp.alpha + bb.y) * be.y;
        if (tp.res) {
          const float2 rr = __half22float2(*reinterpret_cast<const __half2*>(tp.res + r * tp.cout + c));
          x0 += rr.x;
          x1 += rr.y;
        }
        x0 = x0 > 0.f ? x0 : x0 * tp.slope;
        x1 = x1 > 0.f ? x1 : x1 * tp.slope;
        if (tp.out8)
          *reinterpret_cast<uint16_t*>(tp.out8 + r * tp.cout + c) = __nv_cvt_float2_to_fp8x2(
              make_float2(x0 * tp.inv_s8, x1 * tp.inv_s8), __NV_SATFINITE, __NV_E4M3);
        if (tp.out16) *reinterpret_cast<__half2*>(tp.out16 + r * tp.cout + c) = __floats2half2_rn(x0, x1);
      }
    }
  }
}

// EIn : float_e4m3_t ou half_t. TN : colonnes de sortie par CTA (32 ou 64).
template <class EIn, int TM, int TN, int TK, int WM, int Stages,
          cutlass::conv::IteratorAlgorithm Alg = cutlass::conv::IteratorAlgorithm::kOptimized>
struct Cfg {
  using E = EIn;
  static constexpr int kTN = TN;
  static constexpr bool kFp8 = cutlass::sizeof_bits<EIn>::value == 8;
  using Inst = typename std::conditional<kFp8, cutlass::gemm::GemmShape<16, 8, 32>,
                                         cutlass::gemm::GemmShape<16, 8, 16>>::type;
  using Dummy = cutlass::epilogue::thread::LinearCombination<EIn, 8, float, float>;  // jamais exécuté
  using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
      EIn, cutlass::layout::TensorNHWC, EIn, cutlass::layout::TensorNHWC, EIn, cutlass::layout::TensorNHWC,
      float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89, cutlass::gemm::GemmShape<TM, TN, TK>,
      cutlass::gemm::GemmShape<WM, TN, TK>, Inst, Dummy,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, cutlass::arch::OpMultiplyAdd,
      Alg>::Kernel;
};

// Un lancement : paramètres CUTLASS construits une fois ; compatible avec la capture de graphe CUDA.
template <class C>
struct Launcher {
  using Base = typename C::Base;
  using E = typename C::E;
  typename Base::Params params;
  KP kp;
  dim3 grid, block;
  size_t smem = 0;

  // Convolution RxR, remplissage pad, pas stride ; Cin canaux d'entrée (multiple de 16), Kout de sortie.
  void init(int H, int W, int R, int pad, int stride, int Cin, int Kout, const E* in, const E* w,
            const KP& k) {
    cutlass::conv::Conv2dProblemSize ps({1, H, W, Cin}, {Kout, R, R, Cin}, {pad, pad, pad, pad},
                                        {stride, stride}, {1, 1}, cutlass::conv::Mode::kCrossCorrelation, 1);
    E* dummy = reinterpret_cast<E*>(0x1000);  // jamais déréférencé : l'épilogue standard n'est pas exécuté
    int P = ps.P, Q = ps.Q;
    typename Base::Arguments args(
        ps, {const_cast<E*>(in), {Cin, Cin * W, Cin * W * H}},
        {const_cast<E*>(w), {Cin, Cin * R, Cin * R * R}}, {dummy, {Kout, Kout * Q, Kout * Q * P}},
        {dummy, {Kout, Kout * Q, Kout * Q * P}}, typename C::Dummy::Params(1.f, 0.f));
    params = typename Base::Params(args, nullptr);
    kp = k;
    kp.cout = Kout;
    kp.m = long(P) * Q;
    grid = typename Base::ThreadblockSwizzle().get_grid_shape(params.grid_tiled_shape);
    block = dim3(Base::kThreadCount);
    smem = ((sizeof(typename Base::Mma::SharedStorage) + 15) / 16) * 16;
    if (smem > 48 * 1024)
      cudaFuncSetAttribute(rife_conv_kernel<Base, C::kTN>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           int(smem));
  }
  void launch(cudaStream_t s) const { rife_conv_kernel<Base, C::kTN><<<grid, block, smem, s>>>(params, kp); }
};

}  // namespace rife
