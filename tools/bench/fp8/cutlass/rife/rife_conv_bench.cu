// Mesure isolée des convolutions 3x3 de RIFE v4.25 (IFNet) en FP8 E4M3 CUTLASS avec épilogue fusionné
// (biais, bêta, résiduel, LeakyRelu 0,2, requantification), aux résolutions réelles des 5 blocs
// pour une image de travail de 768x1280. Chaque cas : temps par lancement (graphe de 200 lancements,
// tenseurs résidents en L2 comme dans le réseau), erreur contre une référence GPU fp32 sur les valeurs
// déquantifiées, et temps TensorRT FP16 de la même couche (profil --dumpProfile, moyenne par inférence).
//   rife_conv_bench
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "rife_kernels.h"

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(2);                                                                        \
    }                                                                                      \
  } while (0)

using namespace rife;
using E4 = cutlass::float_e4m3_t;

// Référence : conv RxR, pas stride, remplissage pad, sur des valeurs fp32 NHWC, filtre KRSC.
__global__ void ref_kernel(const float* x, const float* w, const float* b, const float* beta,
                           const float* res, float* y, int H, int W, int P, int Q, int Cin, int Cout, int R,
                           int pad, int stride, float slope) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(P) * Q * Cout) return;
  int k = idx % Cout;
  size_t p = idx / Cout;
  int q = p % Q, pp = p / Q;
  float acc = 0.f;
  for (int r = 0; r < R; ++r)
    for (int s = 0; s < R; ++s) {
      int ih = pp * stride + r - pad, iw = q * stride + s - pad;
      if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
      const float* xp = x + (size_t(ih) * W + iw) * Cin;
      const float* wp = w + ((size_t(k) * R + r) * R + s) * Cin;
      for (int c = 0; c < Cin; ++c) acc = fmaf(xp[c], wp[c], acc);
    }
  acc = (acc + b[k]) * (beta ? beta[k] : 1.f);
  if (res) acc += res[idx];
  y[idx] = acc > 0.f ? acc : acc * slope;
}

template <class T>
static T* up(const std::vector<T>& v) {
  T* d;
  CK(cudaMalloc(&d, v.size() * sizeof(T)));
  CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

struct Row {
  const char* name;
  int count;      // nombre de couches identiques dans le réseau
  double ms_cut;  // ms par lancement CUTLASS
  double ms_trt;  // ms par couche TensorRT FP16 (profil)
  double psnr;
};
static std::vector<Row> rows;

// Un cas. Cin_real : canaux réels de l'entrée (le reste est rempli de zéros jusqu'à Cin, multiple de 32).
template <class C>
static void run_case(const char* name, int count, int H, int W, int Cin, int Cout, int stride, bool residual,
                     double trt_ms) {
  const int P = (H + 2 - 3) / stride + 1, Q = (W + 2 - 3) / stride + 1;
  std::mt19937 g(7);
  std::uniform_real_distribution<float> u(0.f, 1.f), ub(-0.1f, 0.1f), ur(0.f, 1.f);
  std::normal_distribution<float> n(0.f, std::sqrt(2.f / (9 * Cin)));
  std::vector<float> x(size_t(H) * W * Cin), w(size_t(Cout) * 9 * Cin), b(Cout), beta(Cout),
      res(size_t(P) * Q * Cout);
  for (auto& v : x) v = u(g);
  for (auto& v : w) v = n(g);
  for (auto& v : b) v = ub(g);
  for (auto& v : beta) v = 0.3f + 0.1f * u(g);
  for (auto& v : res) v = ur(g);
  float sx = 1.f / 448.f, sw = 0;
  for (float v : w) sw = std::max(sw, std::fabs(v));
  sw /= 448.f;
  std::vector<E4> xq(x.size()), wq(w.size());
  std::vector<float> xd(x.size()), wd(w.size());
  for (size_t i = 0; i < x.size(); ++i) {
    xq[i] = E4(x[i] / sx);
    xd[i] = float(xq[i]) * sx;
  }
  for (size_t i = 0; i < w.size(); ++i) {
    wq[i] = E4(w[i] / sw);
    wd[i] = float(wq[i]) * sw;
  }
  std::vector<__half> resh(res.size());
  for (size_t i = 0; i < res.size(); ++i) resh[i] = __float2half(res[i]);
  for (size_t i = 0; i < res.size(); ++i) res[i] = __half2float(resh[i]);

  E4 *d_x = up(xq), *d_w = up(wq);
  float *d_b = up(b), *d_beta = up(beta), *d_xd = up(xd), *d_wd = up(wd), *d_res = up(res), *d_ref;
  __half* d_resh = up(resh);
  CK(cudaMalloc(&d_ref, res.size() * 4));
  __half* d_o16;
  uint8_t* d_o8;
  CK(cudaMalloc(&d_o16, res.size() * 2));
  CK(cudaMalloc(&d_o8, res.size()));

  float sy = 4.f / 448.f;  // échelle de sortie E4M3 (la sortie est de l'ordre de 1)
  KP k;
  k.alpha = sx * sw;
  k.bias = d_b;
  if (residual) {
    k.beta = d_beta;
    k.res = d_resh;
  }
  k.out8 = d_o8;
  k.inv_s8 = 1.f / sy;
  k.out16 = d_o16;
  static Launcher<C> L;
  L.init(H, W, 3, 1, stride, Cin, Cout, d_x, d_w, k);

  ref_kernel<<<(res.size() + 255) / 256, 256>>>(d_xd, d_wd, d_b, residual ? d_beta : nullptr,
                                                residual ? d_res : nullptr, d_ref, H, W, P, Q, Cin, Cout, 3,
                                                1, stride, 0.2f);
  L.launch(0);
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  std::vector<float> ref(res.size());
  std::vector<__half> o16(res.size());
  std::vector<uint8_t> o8(res.size());
  CK(cudaMemcpy(ref.data(), d_ref, ref.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o16.data(), d_o16, o16.size() * 2, cudaMemcpyDeviceToHost));
  double mse = 0, peak = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double d = double(__half2float(o16[i])) - ref[i];
    mse += d * d;
    peak = std::max(peak, double(std::fabs(ref[i])));
  }
  mse /= ref.size();
  double psnr = mse > 0 ? 10 * std::log10(peak * peak / mse) : 200;

  // temps : graphe de 200 lancements
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cudaGraph_t gr;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < 200; ++i) L.launch(s);
  CK(cudaStreamEndCapture(s, &gr));
  CK(cudaGraphInstantiate(&ge, gr, 0));
  for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, s));
  CK(cudaStreamSynchronize(s));
  std::vector<float> t(15);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int i = 0; i < 15; ++i) {
    CK(cudaEventRecord(e0, s));
    CK(cudaGraphLaunch(ge, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&t[i], e0, e1));
  }
  std::sort(t.begin(), t.end());
  double ms = t[7] / 200.0;
  rows.push_back({name, count, ms, trt_ms, psnr});
  std::printf("%-34s %dx%d->%dx%d  C %3d->%3d  grille %3dx%d  CUTLASS %.4f ms  TRT %.4f ms  PSNR %.1f dB\n",
              name, H, W, P, Q, Cin, Cout, L.grid.x, L.grid.y, ms, trt_ms, psnr);
  std::fflush(stdout);
  cudaFree(d_x);
  cudaFree(d_w);
  cudaFree(d_b);
  cudaFree(d_beta);
  cudaFree(d_xd);
  cudaFree(d_wd);
  cudaFree(d_res);
  cudaFree(d_ref);
  cudaFree(d_resh);
  cudaFree(d_o16);
  cudaFree(d_o8);
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(gr);
}

int main() {
  // (TM, TN, TK, WM, étages)
  using A64 = Cfg<E4, 128, 64, 64, 32, 3>;  // Cout multiple de 64, Cin >= 64
  using B32 = Cfg<E4, 128, 32, 64, 32, 3>;  // Cout multiple de 32, Cin multiple de 32
  using D48 = Cfg<E4, 128, 48, 64, 32, 3>;  // Cout 48
  // Couches résiduelles 3x3 (bloc k : 8 couches), temps TRT = moyenne mesurée dans le moteur
  run_case<A64>("blk0 res 192->192 @24x40", 8, 24, 40, 192, 192, 1, true, 0.0116);
  run_case<A64>("blk1 res 128->128 @24x40", 8, 24, 40, 128, 128, 1, true, 0.0088);
  run_case<B32>("blk2 res  96-> 96 @48x80", 8, 48, 80, 96, 96, 1, true, 0.0118);
  run_case<A64>("blk3 res  64-> 64 @96x160", 8, 96, 160, 64, 64, 1, true, 0.0142);
  run_case<B32>("blk4 res  32-> 32 @192x320", 8, 192, 320, 32, 32, 1, true, 0.0212);
  // Convs strided d'entrée (Relu), Cin rempli à 32 minimum
  run_case<B32>("blk0 conv0.1  96->192 s2 @48x80", 1, 48, 80, 96, 192, 2, false, 0.0116);
  run_case<B32>("blk1 conv0.1  64->128 s2 @48x80", 1, 48, 80, 64, 128, 2, false, 0.0084);
  run_case<D48>("blk2 conv0.1  48-> 96 s2 @96x160", 1, 96, 160, 48, 96, 2, false, 0.0114);
  run_case<B32>("blk3 conv0.1  32-> 64 s2 @192x320", 1, 192, 320, 32, 64, 2, false, 0.0109);
  run_case<B32>("blk4 conv0.1  16-> 32 s2 @384x640 (Cin 32)", 1, 384, 640, 32, 32, 2, false, 0.0199);
  run_case<B32>("blk0 conv0.0  32-> 96 s2 @96x160 (Cin 32)", 1, 96, 160, 32, 96, 2, false, 0.0082);
  run_case<B32>("blk3 conv0.0  32-> 32 s2 @384x640 (Cin 32)", 1, 384, 640, 32, 32, 2, false, 0.0263);
  run_case<B32>("blk4 conv0.0  28-> 16 s2 @768x1280 (Cin 32)", 1, 768, 1280, 32, 32, 2, false, 0.2161);

  double c = 0, t = 0;
  for (auto& r : rows) {
    c += r.ms_cut * r.count;
    t += r.ms_trt * r.count;
  }
  std::printf("\nsomme pondérée (nombre de couches) : CUTLASS %.3f ms, TensorRT FP16 %.3f ms\n", c, t);
  return 0;
}
