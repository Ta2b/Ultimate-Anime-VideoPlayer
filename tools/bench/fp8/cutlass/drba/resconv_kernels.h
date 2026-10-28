#pragma once
// Prototype CUTLASS : ResConv de DistilDRBA fusionné dans l'épilogue d'une convolution 3x3 C->C FP8 E4M3.
//   y = LeakyRelu_0,2( (Conv3x3(x) + b) * beta + r )
// x : activation NHWC E4M3 ; r : flot résiduel NHWC FP16 (la somme des sauts reste en FP16, seule l'entrée
// de la conv est quantifiée) ; sorties : y en FP16 (résiduel suivant) et y en E4M3 (entrée de la conv
// suivante, échelle statique). La boucle principale est celle de CUTLASS (Mma de DefaultConv2dFprop) ;
// l'épilogue est écrit à la main sur les registres de l'accumulateur m16n8, sans passage par la mémoire.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/numeric_types.h"

namespace drba {

struct ResParams {
  const __half* res;  // flot résiduel d'entrée, NHWC FP16
  const float* ba;    // alpha * beta[c] : accumulateur -> unités réelles, multiplié par beta
  const float* bb;    // b[c] * beta[c]
  __half* out16;      // résiduel de sortie, NHWC FP16
  uint8_t* out8;      // sortie requantifiée, NHWC E4M3
  float inv_sy;       // 1 / échelle de requantification
  int rows;           // H * W
};

template <class Base, int C>
__global__ void __launch_bounds__(Base::kThreadCount)
    resconv_kernel(typename Base::Params params, ResParams rp) {
  using Mma = typename Base::Mma;
  constexpr int kWarpM = Mma::Policy::Operator::Shape::kM;
  constexpr int kWarpN = Mma::Policy::Operator::Shape::kN;
  constexpr int kMI = kWarpM / 16;
  constexpr int kNI = kWarpN / 8;

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

  const float* a = acc.data();
  const int g = lane >> 2, t = lane & 3;
  const int wmn = warp_idx % (Mma::WarpCount::kM * Mma::WarpCount::kN);
  const int warp_m = wmn % Mma::WarpCount::kM, warp_n = wmn / Mma::WarpCount::kM;
  const long row_base = long(tile.m()) * Mma::Shape::kM + warp_m * kWarpM;
  const int col_base = tile.n() * Mma::Shape::kN + warp_n * kWarpN;

#pragma unroll
  for (int m = 0; m < kMI; ++m) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const long r = row_base + m * 16 + g + 8 * h;
      if (r >= rp.rows) continue;
#pragma unroll
      for (int n = 0; n < kNI; ++n) {
        const int c = col_base + n * 8 + 2 * t;
        const float2 ba = __ldg(reinterpret_cast<const float2*>(rp.ba + c));
        const float2 bb = __ldg(reinterpret_cast<const float2*>(rp.bb + c));
        const __half2 rs = *reinterpret_cast<const __half2*>(rp.res + r * C + c);
        float y0 = a[(m + n * kMI) * 4 + 2 * h + 0] * ba.x + bb.x + __low2float(rs);
        float y1 = a[(m + n * kMI) * 4 + 2 * h + 1] * ba.y + bb.y + __high2float(rs);
        y0 = y0 > 0.f ? y0 : 0.2f * y0;
        y1 = y1 > 0.f ? y1 : 0.2f * y1;
        *reinterpret_cast<__half2*>(rp.out16 + r * C + c) = __floats2half2_rn(y0, y1);
        *reinterpret_cast<uint16_t*>(rp.out8 + r * C + c) =
            __nv_cvt_float2_to_fp8x2(make_float2(y0 * rp.inv_sy, y1 * rp.inv_sy), __NV_SATFINITE, __NV_E4M3);
      }
    }
  }
}

template <int C, int TM, int TN, int TK, int WM, int WN, int Stages>
struct ResCfg {
  using E = cutlass::float_e4m3_t;
  using Dummy =
      cutlass::epilogue::thread::LinearCombination<E, 8, float, float>;  // types de Params seulement
  using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
      E, cutlass::layout::TensorNHWC, E, cutlass::layout::TensorNHWC, E, cutlass::layout::TensorNHWC, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89, cutlass::gemm::GemmShape<TM, TN, TK>,
      cutlass::gemm::GemmShape<WM, WN, TK>, cutlass::gemm::GemmShape<16, 8, 32>, Dummy,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
};

// Lancement : construit les Params une fois, puis lance (compatible capture de graphe CUDA).
template <class Cfg, int C>
struct ResLauncher {
  using Base = typename Cfg::Base;
  using E = typename Cfg::E;
  typename Base::Params params;
  ResParams rp;
  dim3 grid, block;
  size_t smem = 0;

  void init(int H, int W, const E* in, const E* w, E* dummy_out, const ResParams& r) {
    cutlass::conv::Conv2dProblemSize ps({1, H, W, C}, {C, 3, 3, C}, {1, 1, 1, 1}, {1, 1}, {1, 1},
                                        cutlass::conv::Mode::kCrossCorrelation, 1);
    typename Base::Arguments args(ps, {const_cast<E*>(in), {C, C * W, C * W * H}},
                                  {const_cast<E*>(w), {C, C * 3, C * 3 * 3}},
                                  {dummy_out, {C, C * W, C * W * H}}, {dummy_out, {C, C * W, C * W * H}},
                                  typename Cfg::Dummy::Params(1.f, 0.f));
    params = typename Base::Params(args, nullptr);
    rp = r;
    grid = typename Base::ThreadblockSwizzle().get_grid_shape(params.grid_tiled_shape);
    block = dim3(Base::kThreadCount);
    smem = sizeof(typename Base::Mma::SharedStorage);
    if (smem > 48 * 1024)
      cudaFuncSetAttribute(resconv_kernel<Base, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  }
  void launch(cudaStream_t s) const { resconv_kernel<Base, C><<<grid, block, smem, s>>>(params, rp); }
};

}  // namespace drba
