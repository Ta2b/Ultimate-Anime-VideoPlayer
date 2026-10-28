// Variante FP8 E4M3 de bloc_fusion.cu : même bloc résiduel plein résolution du réseau « fusion » de GMFSS
// (64 canaux, 384x640), activations et poids en E4M3, accumulation f32, épilogue fusionnée.
//
//   noyau 1 : q8  = E4M3( PReLU(Conv3x3(p8) * alpha1[c] + b1[c]) / sq )      alpha1[c] = sp * sw1[c]
//   noyau 2 : out = Conv3x3(q8) * alpha2[c] + b2[c] + x                      (saut et sortie en FP16)
//
// Échelles statiques : sp, sq par tenseur (amax du banc de calibration = les entrées réelles), poids par
// canal de sortie. Vérifié contre la référence ONNX Runtime CPU (FP16) sur les mêmes entrées réelles.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
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

constexpr int kC = 64;

struct EpiParams {
  const float* alpha;  // échelle entrée * poids, par canal de sortie
  const float* bias;   // biais en unités réelles
  const __half* skip;  // saut résiduel FP16 (mode 1)
  void* out;           // E4M3 (mode 0) ou FP16 (mode 1)
  float slope;         // pente négative du PReLU
  float inv_sq;        // 1 / échelle de requantification de la sortie (mode 0)
  long m_total;
};

// Mode 0 : E4M3(PReLU(acc*alpha + biais) * inv_sq)    Mode 1 : FP16(acc*alpha + biais + saut)
template <class Base, int Mode>
__global__ void __launch_bounds__(Base::kThreadCount)
    conv8_kernel(typename Base::Params params, EpiParams ep) {
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
      const float2 al = __ldg(reinterpret_cast<const float2*>(ep.alpha + c));
      const float2 bb = __ldg(reinterpret_cast<const float2*>(ep.bias + c));
      float v[4] = {a[(m + n * kMI) * 4 + 0] * al.x + bb.x, a[(m + n * kMI) * 4 + 1] * al.y + bb.y,
                    a[(m + n * kMI) * 4 + 2] * al.x + bb.x, a[(m + n * kMI) * 4 + 3] * al.y + bb.y};
      if constexpr (Mode == 0) {
        uint8_t* out = reinterpret_cast<uint8_t*>(ep.out);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = (v[i] > 0.f ? v[i] : v[i] * ep.slope) * ep.inv_sq;
        if (r0 < ep.m_total)
          *reinterpret_cast<uint16_t*>(out + r0 * kC + c) =
              __nv_cvt_float2_to_fp8x2(make_float2(v[0], v[1]), __NV_SATFINITE, __NV_E4M3);
        if (r1 < ep.m_total)
          *reinterpret_cast<uint16_t*>(out + r1 * kC + c) =
              __nv_cvt_float2_to_fp8x2(make_float2(v[2], v[3]), __NV_SATFINITE, __NV_E4M3);
      } else {
        __half* out = reinterpret_cast<__half*>(ep.out);
        if (r0 < ep.m_total) {
          const float2 s = __half22float2(*reinterpret_cast<const __half2*>(ep.skip + r0 * kC + c));
          *reinterpret_cast<__half2*>(out + r0 * kC + c) = __floats2half2_rn(v[0] + s.x, v[1] + s.y);
        }
        if (r1 < ep.m_total) {
          const float2 s = __half22float2(*reinterpret_cast<const __half2*>(ep.skip + r1 * kC + c));
          *reinterpret_cast<__half2*>(out + r1 * kC + c) = __floats2half2_rn(v[2] + s.x, v[3] + s.y);
        }
      }
    }
  }
}

template <int TM, int TK, int WM, int WN, int Stages>
struct Cfg {
  using E = cutlass::float_e4m3_t;
  using Dummy = cutlass::epilogue::thread::LinearCombination<E, 8, float, float>;
  using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
      E, cutlass::layout::TensorNHWC, E, cutlass::layout::TensorNHWC, E, cutlass::layout::TensorNHWC, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89, cutlass::gemm::GemmShape<TM, 64, TK>,
      cutlass::gemm::GemmShape<WM, WN, TK>, cutlass::gemm::GemmShape<16, 8, 32>, Dummy,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
};

template <class C, int Mode>
struct Launcher {
  using Base = typename C::Base;
  typename Base::Params params;
  EpiParams ep{};
  dim3 grid, block;
  size_t smem = 0;
  void init(int H, int W, const uint8_t* in, const uint8_t* w, const EpiParams& e, void* dummy_out) {
    using E = typename C::E;
    cutlass::conv::Conv2dProblemSize ps({1, H, W, kC}, {kC, 3, 3, kC}, {1, 1, 1, 1}, {1, 1}, {1, 1},
                                        cutlass::conv::Mode::kCrossCorrelation, 1);
    E* dummy = reinterpret_cast<E*>(dummy_out);
    typename Base::Arguments args(ps,
                                  {reinterpret_cast<E*>(const_cast<uint8_t*>(in)), {kC, kC * W, kC * W * H}},
                                  {reinterpret_cast<E*>(const_cast<uint8_t*>(w)), {kC, kC * 3, kC * 3 * 3}},
                                  {dummy, {kC, kC * W, kC * W * H}}, {dummy, {kC, kC * W, kC * W * H}},
                                  typename C::Dummy::Params(1.f, 0.f));
    params = typename Base::Params(args, nullptr);
    ep = e;
    grid = typename Base::ThreadblockSwizzle().get_grid_shape(params.grid_tiled_shape);
    block = dim3(Base::kThreadCount);
    smem = ((sizeof(typename Base::Mma::SharedStorage) + 15) / 16) * 16;
    if (smem > 48 * 1024)
      CK(cudaFuncSetAttribute(conv8_kernel<Base, Mode>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              int(smem)));
  }
  void launch(cudaStream_t s) const { conv8_kernel<Base, Mode><<<grid, block, smem, s>>>(params, ep); }
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
T* vers_gpu(const void* h, size_t bytes) {
  T* p;
  CK(cudaMalloc(&p, bytes));
  CK(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice));
  return p;
}

uint8_t quant(float v) {
  return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

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

template <class C>
void essai(const char* nom, const std::string& dir, int H, int W) {
  const size_t n = size_t(H) * W * kC;
  const long M = long(H) * W;
  auto vp = lire(dir + "/p.bin"), vx = lire(dir + "/x.bin"), vref = lire(dir + "/ref.bin");
  auto vw1 = lire(dir + "/w1.bin"), vw2 = lire(dir + "/w2.bin"), vb1 = lire(dir + "/b1.bin"),
       vb2 = lire(dir + "/b2.bin");
  const float slope = std::strtof(std::string(lire(dir + "/params.txt").data()).c_str(), nullptr);
  const __half* hp = reinterpret_cast<const __half*>(vp.data());
  const __half* hw1 = reinterpret_cast<const __half*>(vw1.data());
  const __half* hw2 = reinterpret_cast<const __half*>(vw2.data());
  const float* b1 = reinterpret_cast<const float*>(vb1.data());
  const float* b2 = reinterpret_cast<const float*>(vb2.data());

  // Calibration statique sur les entrées réelles : amax -> 448 (E4M3).
  float amax_p = 0;
  for (size_t i = 0; i < n; ++i) amax_p = std::max(amax_p, std::fabs(__half2float(hp[i])));
  const float sp = amax_p / 448.f;
  std::vector<uint8_t> p8(n), w1_8(kC * 9 * kC), w2_8(kC * 9 * kC);
  for (size_t i = 0; i < n; ++i) p8[i] = quant(__half2float(hp[i]) / sp);
  std::vector<float> sw1(kC), sw2(kC);
  for (int k = 0; k < kC; ++k) {
    float m1 = 0, m2 = 0;
    for (int i = 0; i < 9 * kC; ++i) {
      m1 = std::max(m1, std::fabs(__half2float(hw1[k * 9 * kC + i])));
      m2 = std::max(m2, std::fabs(__half2float(hw2[k * 9 * kC + i])));
    }
    sw1[k] = m1 / 448.f;
    sw2[k] = m2 / 448.f;
    for (int i = 0; i < 9 * kC; ++i) {
      w1_8[k * 9 * kC + i] = quant(__half2float(hw1[k * 9 * kC + i]) / sw1[k]);
      w2_8[k * 9 * kC + i] = quant(__half2float(hw2[k * 9 * kC + i]) / sw2[k]);
    }
  }
  // sq : amax de q = PReLU(conv1(p) + b1) sur les entrées réelles, par un calcul FP32 de conv1 sur un
  // échantillon de lignes (CPU), avec 25 % de marge : suffisant pour une échelle statique de banc d'essai.
  float amax_q = 0;
  {
    const int rows[] = {50, 120, 200, 300};
    for (int y : rows)
      for (int x = 0; x < W; x += 3)
        for (int k = 0; k < kC; ++k) {
          float acc = b1[k];
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              int yy = y + dy, xx = x + dx;
              if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
              for (int c = 0; c < kC; ++c)
                acc += __half2float(hp[(size_t(yy) * W + xx) * kC + c]) *
                       __half2float(hw1[k * 9 * kC + ((dy + 1) * 3 + (dx + 1)) * kC + c]);
            }
          acc = acc > 0 ? acc : acc * slope;
          amax_q = std::max(amax_q, std::fabs(acc));
        }
  }
  amax_q *= 1.25f;  // marge pour les pixels hors échantillon
  const float sq = amax_q / 448.f;
  std::printf("%-30s échelles : sp %.3g (amax %.3f), sq %.3g (amax ~%.3f)\n", nom, sp, amax_p, sq, amax_q);

  std::vector<float> al1(kC), al2(kC);
  for (int k = 0; k < kC; ++k) {
    al1[k] = sp * sw1[k];
    al2[k] = sq * sw2[k];
  }
  auto* d_p8 = vers_gpu<uint8_t>(p8.data(), n);
  auto* d_w1 = vers_gpu<uint8_t>(w1_8.data(), w1_8.size());
  auto* d_w2 = vers_gpu<uint8_t>(w2_8.data(), w2_8.size());
  auto* d_al1 = vers_gpu<float>(al1.data(), kC * 4);
  auto* d_al2 = vers_gpu<float>(al2.data(), kC * 4);
  auto* d_b1 = vers_gpu<float>(b1, kC * 4);
  auto* d_b2 = vers_gpu<float>(b2, kC * 4);
  auto* d_x = vers_gpu<__half>(vx.data(), n * 2);
  uint8_t* d_q8;
  __half* d_out;
  CK(cudaMalloc(&d_q8, n));
  CK(cudaMalloc(&d_out, n * 2));

  Launcher<C, 0> k1;
  Launcher<C, 1> k2;
  k1.init(H, W, d_p8, d_w1, {d_al1, d_b1, nullptr, d_q8, slope, 1.f / sq, M}, d_q8);
  k2.init(H, W, d_q8, d_w2, {d_al2, d_b2, d_x, d_out, 0.f, 0.f, M}, d_q8);
  k1.launch(0);
  k2.launch(0);
  CK(cudaDeviceSynchronize());
  std::vector<__half> h(n);
  CK(cudaMemcpy(h.data(), d_out, n * 2, cudaMemcpyDeviceToHost));
  const __half* r = reinterpret_cast<const __half*>(vref.data());
  double mx = 0, se = 0, se_skip = 0;
  const __half* hx = reinterpret_cast<const __half*>(vx.data());
  for (size_t i = 0; i < n; ++i) {
    double d = double(__half2float(h[i])) - double(__half2float(r[i]));
    mx = std::max(mx, std::fabs(d));
    se += d * d;
    double e0 = double(__half2float(hx[i])) - double(__half2float(r[i]));
    se_skip += e0 * e0;
  }
  std::printf(
      "%-30s FP8 vs ONNX Runtime CPU : max %.4g, PSNR %.1f dB (la sortie du bloc sans le résiduel : %.1f "
      "dB)\n",
      nom, mx, 10 * std::log10(1.0 / std::max(se / double(n), 1e-20)),
      10 * std::log10(1.0 / std::max(se_skip / double(n), 1e-20)));

  auto fus = chronometrer([&](cudaStream_t s) {
    k1.launch(s);
    k2.launch(s);
  });
  auto t1 = chronometrer([&](cudaStream_t s) { k1.launch(s); });
  auto t2 = chronometrer([&](cudaStream_t s) { k2.launch(s); });
  const double gflop = 2.0 * 2 * M * 64 * 64 * 9 * 1e-9;
  std::printf("%-30s bloc FP8 fusionné : méd %.4f ms (min %.4f) %.1f TFLOPS | noyau1 %.4f, noyau2 %.4f ms\n",
              nom, fus.first, fus.second, gflop / fus.first, t1.first, t2.first);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage : bloc_fp8 dossier_bloc [H W]\n");
    return 1;
  }
  const std::string dir = argv[1];
  const int H = argc > 2 ? std::atoi(argv[2]) : 384, W = argc > 3 ? std::atoi(argv[3]) : 640;
  essai<Cfg<128, 64, 64, 32, 3>>("fp8 128x64x64 w64x32 s3", dir, H, W);
  essai<Cfg<128, 64, 32, 64, 3>>("fp8 128x64x64 w32x64 s3", dir, H, W);
  essai<Cfg<64, 64, 32, 32, 5>>("fp8 64x64x64 w32x32 s5", dir, H, W);
  essai<Cfg<256, 64, 64, 32, 3>>("fp8 256x64x64 w64x32 s3", dir, H, W);
  return 0;
}
