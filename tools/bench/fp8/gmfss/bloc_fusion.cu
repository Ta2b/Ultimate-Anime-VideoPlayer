// Prototype : bloc résiduel plein résolution du réseau « fusion » de GMFSS (64 canaux, 384x640, FP16)
// avec les opérations élémentaires fusionnées dans l'épilogue des convolutions 3x3.
//
//   noyau 1 : q   = PReLU(Conv3x3(p) + b1)         (PReLU = relu(x) - a*relu(-x), a scalaire)
//   noyau 2 : out = Conv3x3(q) + b2 + x            (saut résiduel)
//
// La boucle principale est celle de CUTLASS (Conv2dFprop, Mma tensor core sm80 qui tourne sur Ada) ;
// l'épilogue est écrit à la main sur les registres de l'accumulateur : biais, activation, saut, écriture
// FP16, sans aller-retour mémoire du tenseur intermédiaire.
//
// Banc d'essai : lit p, x, ref (sortie d'ONNX Runtime CPU sur des entrées réelles, voir prep_bloc.py),
// vérifie les deux noyaux et mesure le temps GPU de chaque configuration ; compare aussi à une version
// NON fusionnée (convolution seule + noyau élémentaire séparé) pour isoler le gain de la fusion.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/half.h"
#include "cutlass/numeric_types.h"

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

namespace {

constexpr int kC = 64;  // canaux d'entrée et de sortie

struct EpiParams {
  const float* bias;   // biais (kC)
  const __half* skip;  // saut résiduel NHWC (mode 1)
  __half* out;         // sortie NHWC
  float slope;         // pente négative du PReLU (mode 0)
  long m_total;        // nombre de pixels
};

// Mode 0 : out = PReLU(acc + biais)   Mode 1 : out = acc + biais + saut   Mode 2 : out = acc + biais
template <class Base, class Acc, int Mode>
__global__ void __launch_bounds__(Base::kThreadCount)
    conv_epi_kernel(typename Base::Params params, EpiParams ep) {
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

  // Épilogue en registres. Tuile d'accumulateur (m, n) = acc[(m + n*kMI)*4 + e] :
  // e = 0, 1 -> ligne g, colonnes 2t, 2t+1 ; e = 2, 3 -> ligne g+8 (g = lane/4, t = lane%4).
  const auto* a = acc.data();
  const int g = lane >> 2, t = lane & 3;
  const int warp_mn = warp_idx % (Mma::WarpCount::kM * Mma::WarpCount::kN);
  const int warp_m = warp_mn % Mma::WarpCount::kM;
  const int warp_n = warp_mn / Mma::WarpCount::kM;
  const long row_base = long(tile.m()) * Mma::Shape::kM + warp_m * kWarpM;
  const int col_base = tile.n() * Mma::Shape::kN + warp_n * kWarpN;

#pragma unroll
  for (int m = 0; m < kMI; ++m) {
    const long r0 = row_base + m * 16 + g, r1 = r0 + 8;
#pragma unroll
    for (int n = 0; n < kNI; ++n) {
      const int c = col_base + n * 8 + 2 * t;
      const float2 bb = __ldg(reinterpret_cast<const float2*>(ep.bias + c));
      float v[4] = {float(a[(m + n * kMI) * 4 + 0]) + bb.x, float(a[(m + n * kMI) * 4 + 1]) + bb.y,
                    float(a[(m + n * kMI) * 4 + 2]) + bb.x, float(a[(m + n * kMI) * 4 + 3]) + bb.y};
      if constexpr (Mode == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = v[i] > 0.f ? v[i] : v[i] * ep.slope;
      }
      if constexpr (Mode == 1) {
        if (r0 < ep.m_total) {
          const float2 s = __half22float2(*reinterpret_cast<const __half2*>(ep.skip + r0 * kC + c));
          v[0] += s.x;
          v[1] += s.y;
        }
        if (r1 < ep.m_total) {
          const float2 s = __half22float2(*reinterpret_cast<const __half2*>(ep.skip + r1 * kC + c));
          v[2] += s.x;
          v[3] += s.y;
        }
      }
      if (r0 < ep.m_total) *reinterpret_cast<__half2*>(ep.out + r0 * kC + c) = __floats2half2_rn(v[0], v[1]);
      if (r1 < ep.m_total) *reinterpret_cast<__half2*>(ep.out + r1 * kC + c) = __floats2half2_rn(v[2], v[3]);
    }
  }
}

// Noyau élémentaire séparé (référence « non fusionnée ») : PReLU en place sur un tenseur FP16.
__global__ void prelu_kernel(const __half2* in, __half2* out, long n2, float slope) {
  long i = blockIdx.x * long(blockDim.x) + threadIdx.x;
  if (i >= n2) return;
  float2 v = __half22float2(in[i]);
  v.x = v.x > 0.f ? v.x : v.x * slope;
  v.y = v.y > 0.f ? v.y : v.y * slope;
  out[i] = __floats2half2_rn(v.x, v.y);
}

template <int TM, int TK, int WM, int WN, int Stages, class Acc>
struct Cfg {
  using E = cutlass::half_t;
  using Dummy = cutlass::epilogue::thread::LinearCombination<E, 8, Acc, Acc>;  // types de Params seulement
  using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
      E, cutlass::layout::TensorNHWC, E, cutlass::layout::TensorNHWC, E, cutlass::layout::TensorNHWC, Acc,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TM, 64, TK>,
      cutlass::gemm::GemmShape<WM, WN, TK>, cutlass::gemm::GemmShape<16, 8, 16>, Dummy,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
};

// Un lancement préparé : Params CUTLASS construits une fois, lancement compatible capture de graphe CUDA.
template <class C, class Acc, int Mode>
struct Launcher {
  using Base = typename C::Base;
  typename Base::Params params;
  EpiParams ep{};
  dim3 grid, block;
  size_t smem = 0;

  void init(int H, int W, const __half* in, const __half* w, const EpiParams& e) {
    using E = typename C::E;
    cutlass::conv::Conv2dProblemSize ps({1, H, W, kC}, {kC, 3, 3, kC}, {1, 1, 1, 1}, {1, 1}, {1, 1},
                                        cutlass::conv::Mode::kCrossCorrelation, 1);
    E* dummy = reinterpret_cast<E*>(e.out);
    typename Base::Arguments args(ps,
                                  {reinterpret_cast<E*>(const_cast<__half*>(in)), {kC, kC * W, kC * W * H}},
                                  {reinterpret_cast<E*>(const_cast<__half*>(w)), {kC, kC * 3, kC * 3 * 3}},
                                  {dummy, {kC, kC * W, kC * W * H}}, {dummy, {kC, kC * W, kC * W * H}},
                                  typename C::Dummy::Params(Acc(1.f), Acc(0.f)));
    params = typename Base::Params(args, nullptr);
    ep = e;
    grid = typename Base::ThreadblockSwizzle().get_grid_shape(params.grid_tiled_shape);
    block = dim3(Base::kThreadCount);
    smem = ((sizeof(typename Base::Mma::SharedStorage) + 15) / 16) * 16;
    if (smem > 48 * 1024)
      CK(cudaFuncSetAttribute(conv_epi_kernel<Base, Acc, Mode>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              int(smem)));
  }
  void launch(cudaStream_t s) const {
    conv_epi_kernel<Base, Acc, Mode><<<grid, block, smem, s>>>(params, ep);
  }
};

std::vector<char> lire(const std::string& chemin) {
  std::ifstream f(chemin, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "fichier introuvable : %s\n", chemin.c_str());
    std::exit(1);
  }
  return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

template <class T>
T* vers_gpu(const std::vector<char>& v) {
  T* p;
  CK(cudaMalloc(&p, v.size()));
  CK(cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice));
  return p;
}

struct Ecart {
  double max_abs, psnr;
};
Ecart comparer(const __half* d_out, const std::vector<char>& ref, size_t n) {
  std::vector<__half> h(n);
  CK(cudaMemcpy(h.data(), d_out, n * 2, cudaMemcpyDeviceToHost));
  const __half* r = reinterpret_cast<const __half*>(ref.data());
  double mx = 0, se = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = std::fabs(double(__half2float(h[i])) - double(__half2float(r[i])));
    mx = std::max(mx, d);
    se += d * d;
  }
  return {mx, 10 * std::log10(1.0 / std::max(se / double(n), 1e-20))};
}

// Médiane et minimum (ms) d'une séquence lancée `iter` fois, mesurée par événements CUDA.
std::pair<float, float> chronometrer(const std::function<void(cudaStream_t)>& f, int iter = 300) {
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  f(s);
  CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  for (int i = 0; i < 50; ++i) CK(cudaGraphLaunch(ge, s));
  CK(cudaStreamSynchronize(s));
  std::vector<float> ts;
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  for (int i = 0; i < iter; ++i) {
    CK(cudaEventRecord(a, s));
    CK(cudaGraphLaunch(ge, s));
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    ts.push_back(ms);
  }
  std::sort(ts.begin(), ts.end());
  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroy(g));
  CK(cudaStreamDestroy(s));
  return {ts[ts.size() / 2], ts[0]};
}

}  // namespace

template <class C, class Acc>
void essai(const char* nom, const std::string& dir, int H, int W) {
  const size_t n = size_t(H) * W * kC;
  auto vp = lire(dir + "/p.bin"), vx = lire(dir + "/x.bin"), vref = lire(dir + "/ref.bin");
  auto vw1 = lire(dir + "/w1.bin"), vw2 = lire(dir + "/w2.bin"), vb1 = lire(dir + "/b1.bin"),
       vb2 = lire(dir + "/b2.bin");
  float slope = std::strtof(std::string(lire(dir + "/params.txt").data()).c_str(), nullptr);
  auto* p = vers_gpu<__half>(vp);
  auto* x = vers_gpu<__half>(vx);
  auto* w1 = vers_gpu<__half>(vw1);
  auto* w2 = vers_gpu<__half>(vw2);
  auto* b1 = vers_gpu<float>(vb1);
  auto* b2 = vers_gpu<float>(vb2);
  __half *q, *out, *tmp;
  CK(cudaMalloc(&q, n * 2));
  CK(cudaMalloc(&out, n * 2));
  CK(cudaMalloc(&tmp, n * 2));
  const long M = long(H) * W;

  Launcher<C, Acc, 0> k1;  // conv1 + biais + PReLU
  Launcher<C, Acc, 1> k2;  // conv2 + biais + saut
  Launcher<C, Acc, 2> c1;  // conv1 + biais seule (référence non fusionnée)
  k1.init(H, W, p, w1, {b1, nullptr, q, slope, M});
  k2.init(H, W, q, w2, {b2, x, out, 0.f, M});
  c1.init(H, W, p, w1, {b1, nullptr, tmp, 0.f, M});

  k1.launch(0);
  k2.launch(0);
  CK(cudaDeviceSynchronize());
  Ecart e = comparer(out, vref, n);
  std::printf("%-34s correction vs ONNX Runtime CPU : max %.4g, PSNR %.1f dB\n", nom, e.max_abs, e.psnr);

  auto fus = chronometrer([&](cudaStream_t s) {
    k1.launch(s);
    k2.launch(s);
  });
  auto nonf = chronometrer([&](cudaStream_t s) {
    c1.launch(s);
    prelu_kernel<<<(unsigned)((n / 2 + 255) / 256), 256, 0, s>>>(
        reinterpret_cast<const __half2*>(tmp), reinterpret_cast<__half2*>(q), long(n / 2), slope);
    k2.launch(s);
  });
  auto k1t = chronometrer([&](cudaStream_t s) { k1.launch(s); });
  auto c1t = chronometrer([&](cudaStream_t s) { c1.launch(s); });
  auto k2t = chronometrer([&](cudaStream_t s) { k2.launch(s); });
  const double gflop = 2.0 * 2 * M * 64 * 64 * 9 * 1e-9;
  std::printf(
      "%-34s bloc fusionné : méd %.4f ms (min %.4f) %.1f TFLOPS | non fusionné (conv + noyau PReLU + conv) : "
      "méd %.4f ms (min %.4f)\n",
      nom, fus.first, fus.second, gflop / fus.first, nonf.first, nonf.second);
  std::printf("%-34s détail : noyau1 fusionné %.4f, conv1 seule %.4f, noyau2 fusionné %.4f ms\n", nom,
              k1t.first, c1t.first, k2t.first);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage : bloc_fusion dossier_bloc [H W]\n");
    return 1;
  }
  const std::string dir = argv[1];
  const int H = argc > 2 ? std::atoi(argv[2]) : 384, W = argc > 3 ? std::atoi(argv[3]) : 640;
  using F = float;
  using H16 = cutlass::half_t;
  essai<Cfg<128, 32, 64, 32, 4, F>, F>("128x64x32 w64x32 s4 acc f32", dir, H, W);
  essai<Cfg<128, 32, 64, 32, 4, H16>, H16>("128x64x32 w64x32 s4 acc f16", dir, H, W);
  essai<Cfg<128, 32, 64, 64, 3, F>, F>("128x64x32 w64x64 s3 acc f32", dir, H, W);
  essai<Cfg<128, 32, 64, 64, 3, H16>, H16>("128x64x32 w64x64 s3 acc f16", dir, H, W);
  essai<Cfg<256, 32, 64, 64, 3, H16>, H16>("256x64x32 w64x64 s3 acc f16", dir, H, W);
  return 0;
}
