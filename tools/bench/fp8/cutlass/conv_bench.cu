// Prototype CUTLASS : convolution 3x3 64->64 (NHWC) avec épilogue fusionné « échelle + biais + Relu »,
// en FP16 (accumulateur f32) et en FP8 E4M3 (accumulateur f32, sortie requantifiée par échelle statique).
// Mesure (graphe CUDA, événements, médiane) et correction (référence fp32 GPU, validée par une référence
// CPU). Prototype hors lecteur : CUTLASS est lu depuis un dépôt externe (voir build.sh).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/numeric_types.h"

namespace {

// ---------------------------------------------------------------- noyaux CUTLASS ----

template <class EIn, int TM, int TN, int TK, int WM, int WN, int Stages, class Acc = float>
struct Cfg {
  static constexpr bool kFp8 = cutlass::sizeof_bits<EIn>::value == 8;
  using EOut = typename std::conditional<kFp8, cutlass::float_e4m3_t, cutlass::half_t>::type;
  using Inst = typename std::conditional<kFp8, cutlass::gemm::GemmShape<16, 8, 32>,
                                         cutlass::gemm::GemmShape<16, 8, 16>>::type;
  static constexpr int kAccess =
      8;  // 8 éléments par accès d'épilogue (16 en FP8 ne tient pas dans la tuile 128x64)
  // Z = Relu(alpha * acc + biais) ; alpha porte l'échelle déquantification entrée*poids / requantification
  // sortie
  using EpiOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
      EOut, Acc, float, EOut, EOut, kAccess, cutlass::epilogue::thread::ReLu<float>, cutlass::plus<float>,
      /*StoreT=*/false, /*ElementVector=*/float>;
  using Kernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
      EIn, cutlass::layout::TensorNHWC, EIn, cutlass::layout::TensorNHWC, EOut, cutlass::layout::TensorNHWC,
      Acc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89, cutlass::gemm::GemmShape<TM, TN, TK>,
      cutlass::gemm::GemmShape<WM, WN, TK>, Inst, EpiOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
  using Conv = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};

struct Result {
  std::string name;
  bool ok = false;
  double ms = 0, ms_min = 0, tflops = 0;
  Err e_kernel, e_true;  // contre la référence des mêmes valeurs quantifiées / contre fp32 d'origine
};

double median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <class C>
Result run(const char* name, const Data& d, bool timing) {
  using EIn = typename C::Conv::ElementA;
  using EOut = typename C::EOut;
  constexpr bool fp8 = C::kFp8;
  Result res;
  res.name = name;
  const int H = d.H, W = d.W;
  const size_t nx = d.x.size(), ny = size_t(H) * W * kC;

  // échelles statiques (une par tenseur) : amax -> 448 en FP8 ; 1 en FP16
  auto amax = [](const std::vector<float>& v) {
    float m = 0;
    for (float f : v) m = std::max(m, std::fabs(f));
    return m;
  };
  float sx = fp8 ? amax(d.x) / 448.f : 1.f, sw = fp8 ? amax(d.w) / 448.f : 1.f;
  std::vector<EIn> hx(nx), hw(d.w.size());
  std::vector<float> xq(nx), wq(d.w.size());  // valeurs déquantifiées exactes vues par le noyau
  for (size_t i = 0; i < nx; ++i) {
    hx[i] = EIn(d.x[i] / sx);
    xq[i] = float(hx[i]) * sx;
  }
  for (size_t i = 0; i < d.w.size(); ++i) {
    hw[i] = EIn(d.w[i] / sw);
    wq[i] = float(hw[i]) * sw;
  }
  std::vector<float> ref_true, ref_q;
  if (H * W <= 64 * 64) {  // petit tenseur : la référence GPU est elle-même validée par le CPU
    std::vector<float> cpu;
    conv_cpu(d.x, d.w, d.b, cpu, H, W);
    conv_gpu_ref(d.x, d.w, d.b, ref_true, H, W);
    Err chk = compare(cpu, ref_true);
    std::printf("  [validation référence GPU contre CPU double : erreur max %.3g]\n", chk.max_abs);
  } else {
    conv_gpu_ref(d.x, d.w, d.b, ref_true, H, W);
  }
  conv_gpu_ref(xq, wq, d.b, ref_q, H, W);
  float sy = fp8 ? amax(ref_true) / 448.f : 1.f;
  if (fp8)  // la référence « noyau » est arrondie en E4M3 comme la sortie du noyau : l'écart attendu est d'un
            // ulp au plus
    for (auto& v : ref_q) v = float(EOut(v / sy)) * sy;
  float alpha = sx * sw / sy;
  std::vector<float> bias(kC);
  for (int k = 0; k < kC; ++k) bias[k] = d.b[k] / sy;

  EIn *dx, *dw;
  EOut* dy;
  float* db;
  CK(cudaMalloc(&dx, nx * sizeof(EIn)));
  CK(cudaMalloc(&dw, hw.size() * sizeof(EIn)));
  CK(cudaMalloc(&dy, ny * sizeof(EOut)));
  CK(cudaMalloc(&db, kC * 4));
  CK(cudaMemcpy(dx, hx.data(), nx * sizeof(EIn), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw, hw.data(), hw.size() * sizeof(EIn), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, bias.data(), kC * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dy, 0, ny * sizeof(EOut)));

  cutlass::conv::Conv2dProblemSize ps({1, H, W, kC}, {kC, 3, 3, kC}, {1, 1, 1, 1}, {1, 1}, {1, 1},
                                      cutlass::conv::Mode::kCrossCorrelation, 1);
  typename C::Conv op;
  typename C::Conv::Arguments args(ps, {dx, {kC, kC * W, kC * W * H}}, {dw, {kC, kC * 3, kC * 3 * 3}},
                                   {dy, {kC, kC * W, kC * W * H}}, {dy, {kC, kC * W, kC * W * H}},
                                   {alpha, 0.f}, cutlass::conv::SplitKMode::kSerial, db, nullptr, 0, kC);
  size_t ws = C::Conv::get_workspace_size(args);
  void* dws = nullptr;
  if (ws) CK(cudaMalloc(&dws, ws));
  cutlass::Status st = op.can_implement(args);
  if (st != cutlass::Status::kSuccess) {
    std::printf("  %s : can_implement refuse (%d)\n", name, int(st));
    return res;
  }
  st = op.initialize(args, dws);
  if (st != cutlass::Status::kSuccess) {
    std::printf("  %s : initialize refuse (%d)\n", name, int(st));
    return res;
  }
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  st = op.run(s);
  CK(cudaStreamSynchronize(s));
  CK(cudaGetLastError());
  if (st != cutlass::Status::kSuccess) {
    std::printf("  %s : run refuse\n", name);
    return res;
  }

  std::vector<EOut> hy(ny);
  CK(cudaMemcpy(hy.data(), dy, ny * sizeof(EOut), cudaMemcpyDeviceToHost));
  std::vector<float> got(ny);
  for (size_t i = 0; i < ny; ++i) got[i] = float(hy[i]) * sy;
  res.e_kernel = compare(ref_q, got);
  res.e_true = compare(ref_true, got);
  res.ok = true;

  if (timing) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    op.run(s);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 200; ++i) CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    std::vector<cudaEvent_t> ev(2 * 200);
    for (auto& e : ev) CK(cudaEventCreate(&e));
    for (int i = 0; i < 200; ++i) {
      CK(cudaEventRecord(ev[2 * i], s));
      CK(cudaGraphLaunch(ge, s));
      CK(cudaEventRecord(ev[2 * i + 1], s));
    }
    CK(cudaStreamSynchronize(s));
    std::vector<float> t(200);
    for (int i = 0; i < 200; ++i) CK(cudaEventElapsedTime(&t[i], ev[2 * i], ev[2 * i + 1]));
    res.ms = median(t);
    res.ms_min = *std::min_element(t.begin(), t.end());
    double flop = 2.0 * H * W * kC * kC * 9;
    res.tflops = flop / (res.ms * 1e-3) / 1e12;
    for (auto& e : ev) cudaEventDestroy(e);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
  }
  cudaStreamDestroy(s);
  cudaFree(dx);
  cudaFree(dw);
  cudaFree(dy);
  cudaFree(db);
  if (dws) cudaFree(dws);
  return res;
}

using F16 = cutlass::half_t;
using E4 = cutlass::float_e4m3_t;

struct Entry {
  const char* name;
  std::function<Result(const Data&, bool)> fn;
};

#define ENTRY(NAME, ...)                                                          \
  Entry {                                                                         \
    NAME, [](const Data& d, bool t) { return run<Cfg<__VA_ARGS__>>(NAME, d, t); } \
  }

std::vector<Entry> entries() {
  return {
      ENTRY("fp16 128x64x32 w64x32 s4", F16, 128, 64, 32, 64, 32, 4),
      ENTRY("fp16 128x64x64 w64x32 s3", F16, 128, 64, 64, 64, 32, 3),
      ENTRY("fp16 256x64x32 w64x64 s3", F16, 256, 64, 32, 64, 64, 3),
      ENTRY("fp8 128x64x64 w64x32 s4", E4, 128, 64, 64, 64, 32, 4),
      ENTRY("fp16 128x64x32 w64x32 s4 acc16", F16, 128, 64, 32, 64, 32, 4, F16),
      ENTRY("fp16 128x64x32 w64x32 s5 acc16", F16, 128, 64, 32, 64, 32, 5, F16),
      ENTRY("fp16 256x64x32 w64x64 s3 acc16", F16, 256, 64, 32, 64, 64, 3, F16),
      ENTRY("fp16 128x64x64 w64x32 s3 acc16", F16, 128, 64, 64, 64, 32, 3, F16),
      ENTRY("fp8 128x64x64 w64x32 s3", E4, 128, 64, 64, 64, 32, 3),
      ENTRY("fp8 128x64x64 w64x32 s5", E4, 128, 64, 64, 64, 32, 5),
      ENTRY("fp8 256x64x64 w64x32 s3", E4, 256, 64, 64, 64, 32, 3),
      ENTRY("fp8 64x64x64 w32x32 s5", E4, 64, 64, 64, 32, 32, 5),
  };
}

}  // namespace

int main(int argc, char** argv) {
  std::string only = argc > 1 ? argv[1] : "";
  auto es = entries();
  std::printf("== Petit tenseur 1x64x64x64 : correction ==\n");
  Data small = make_data(64, 64, 1);
  for (auto& e : es) {
    if (!only.empty() && std::string(e.name).find(only) == std::string::npos) continue;
    Result r = e.fn(small, false);
    if (r.ok)
      std::printf("  %-28s noyau: max %.3g PSNR %.1f dB | vs fp32 d'origine: max %.3g PSNR %.1f dB\n", e.name,
                  r.e_kernel.max_abs, r.e_kernel.psnr, r.e_true.max_abs, r.e_true.psnr);
  }
  std::printf("== Grand tenseur 1x64x544x960 ==\n");
  Data big = make_data(544, 960, 2);
  std::printf("%-28s %9s %9s %8s | %-22s | %-22s\n", "config", "ms méd.", "ms min", "TFLOPS",
              "noyau max / PSNR", "vs fp32 max / PSNR");
  for (auto& e : es) {
    if (!only.empty() && std::string(e.name).find(only) == std::string::npos) continue;
    Result r = e.fn(big, true);
    if (r.ok)
      std::printf("%-28s %9.4f %9.4f %8.1f | %9.3g / %6.1f dB | %9.3g / %6.1f dB\n", e.name, r.ms, r.ms_min,
                  r.tflops, r.e_kernel.max_abs, r.e_kernel.psnr, r.e_true.max_abs, r.e_true.psnr);
  }
  return 0;
}
