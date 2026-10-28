// Banc isolé d'un ResConv FP8 de DistilDRBA (conv 3x3 C->C + beta + saut + LeakyRelu, requantifié) aux trois
// formes réelles de 1920x1088, contre la même suite de 8 ResConv sous TensorRT FP16 (resconv_C*.onnx).
//   resconv_bench : exactitude (référence fp32 GPU) puis ms de la suite de 8 dans un graphe CUDA.
#include <cstdio>
#include <vector>

#include "resconv_kernels.h"

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(2);                                                                        \
    }                                                                                      \
  } while (0)

// Référence : accumulateur fp32 sur les valeurs E4M3 décodées.
template <int C>
__global__ void ref_kernel(const __nv_fp8_e4m3* x, const __nv_fp8_e4m3* w, const __half* res, const float* ba,
                           const float* bb, float* y, int H, int W) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(H) * W * C) return;
  int k = idx % C;
  size_t p = idx / C;
  int ww = p % W, h = p / W;
  float acc = 0.f;
  for (int r = 0; r < 3; ++r)
    for (int s = 0; s < 3; ++s) {
      int ih = h + r - 1, iw = ww + s - 1;
      if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
      for (int c = 0; c < C; ++c)
        acc = fmaf(float(x[(size_t(ih) * W + iw) * C + c]), float(w[((size_t(k) * 3 + r) * 3 + s) * C + c]),
                   acc);
    }
  float v = acc * ba[k] + bb[k] + __half2float(res[idx]);
  y[idx] = v > 0.f ? v : 0.2f * v;
}

template <int C, class Cfg>
static void run(const char* name, int H, int W) {
  using L = drba::ResLauncher<Cfg, C>;
  const size_t n = size_t(H) * W * C;
  std::vector<__nv_fp8_e4m3> hx(n), hw(size_t(C) * 9 * C);
  std::vector<__half> hr(n);
  std::vector<float> hba(C), hbb(C);
  unsigned s = 12345;
  auto rnd = [&]() {
    s = s * 1664525u + 1013904223u;
    return (s >> 8) / float(1 << 24);
  };
  for (auto& v : hx) v = __nv_fp8_e4m3(rnd() * 2.f - 1.f);
  const float ws = std::sqrt(3.f * 2.f / (9 * C)) * 0.5f;
  for (auto& v : hw) v = __nv_fp8_e4m3((rnd() * 2.f - 1.f) * ws);
  for (auto& v : hr) v = __float2half(rnd() * 2.f - 1.f);
  for (int c = 0; c < C; ++c) {
    hba[c] = 0.8f + 0.4f * rnd();
    hbb[c] = (rnd() - 0.5f) * 0.1f;
  }
  __nv_fp8_e4m3 *dx, *dw;
  __half *dr, *dy16;
  uint8_t* dy8;
  float *dba, *dbb, *dref;
  CK(cudaMalloc(&dx, n));
  CK(cudaMalloc(&dw, hw.size()));
  CK(cudaMalloc(&dr, n * 2));
  CK(cudaMalloc(&dy16, n * 2));
  CK(cudaMalloc(&dy8, n));
  CK(cudaMalloc(&dba, C * 4));
  CK(cudaMalloc(&dbb, C * 4));
  CK(cudaMalloc(&dref, n * 4));
  CK(cudaMemcpy(dx, hx.data(), n, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dr, hr.data(), n * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dba, hba.data(), C * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbb, hbb.data(), C * 4, cudaMemcpyHostToDevice));

  drba::ResParams rp{dr, dba, dbb, dy16, dy8, 1.f, H * W};
  L l;
  l.init(H, W, reinterpret_cast<const cutlass::float_e4m3_t*>(dx),
         reinterpret_cast<const cutlass::float_e4m3_t*>(dw), nullptr, rp);
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  l.launch(st);
  CK(cudaStreamSynchronize(st));
  CK(cudaGetLastError());
  ref_kernel<C><<<(n + 255) / 256, 256>>>(dx, dw, dr, dba, dbb, dref, H, W);
  CK(cudaDeviceSynchronize());
  std::vector<float> ref(n);
  std::vector<__half> got(n);
  CK(cudaMemcpy(ref.data(), dref, n * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(got.data(), dy16, n * 2, cudaMemcpyDeviceToHost));
  double mx = 0, mse = 0, peak = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = std::fabs(double(__half2float(got[i])) - ref[i]);
    mx = std::max(mx, d);
    mse += d * d;
    peak = std::max(peak, double(std::fabs(ref[i])));
  }
  mse /= n;

  // suite de 8 noyaux dans un graphe : l'entrée est toujours la même (seul le temps compte)
  cudaGraph_t gr;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < 8; ++i) l.launch(st);
  CK(cudaStreamEndCapture(st, &gr));
  CK(cudaGraphInstantiate(&ge, gr, 0));
  for (int i = 0; i < 200; ++i) CK(cudaGraphLaunch(ge, st));
  CK(cudaStreamSynchronize(st));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  const int N = 1000;
  float best = 1e9, sum = 0;
  for (int rep = 0; rep < 5; ++rep) {
    cudaEventRecord(e0, st);
    for (int i = 0; i < N; ++i) CK(cudaGraphLaunch(ge, st));
    cudaEventRecord(e1, st);
    CK(cudaEventSynchronize(e1));
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best = std::min(best, ms / N);
    sum += ms / N;
  }
  std::printf(
      "  C=%d %-26s grille %3dx%d  8 ResConv: min %.1f us  moy %.1f us (%.1f us/noyau) | max err %.4f PSNR "
      "%.1f dB\n",
      C, name, l.grid.x, l.grid.y, best * 1000, sum / 5 * 1000, best * 1000 / 8, mx,
      mse > 0 ? 10 * std::log10(peak * peak / mse) : 200.0);
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(gr);
  cudaStreamDestroy(st);
  cudaFree(dx);
  cudaFree(dw);
  cudaFree(dr);
  cudaFree(dy16);
  cudaFree(dy8);
  cudaFree(dba);
  cudaFree(dbb);
  cudaFree(dref);
}

#define CFG(C, TM, TN, TK, WM, WN, ST, H, W) \
  run<C, drba::ResCfg<C, TM, TN, TK, WM, WN, ST>>(#TM "x" #TN "x" #TK " w" #WM "x" #WN " s" #ST, H, W)

int main() {
  std::printf("TensorRT FP16 de référence (8 ResConv) : C=192 64.5 us, C=128 84.1 us, C=96 164.7 us\n");
  // C=192 sur 17x30
  CFG(192, 32, 64, 64, 32, 32, 4, 17, 30);
  CFG(192, 64, 64, 64, 32, 32, 4, 17, 30);
  CFG(192, 64, 96, 64, 32, 48, 4, 17, 30);
  CFG(192, 64, 192, 64, 32, 96, 3, 17, 30);
  CFG(192, 128, 64, 64, 64, 32, 4, 17, 30);
  // C=128 sur 34x60
  CFG(128, 64, 64, 64, 32, 32, 4, 34, 60);
  CFG(128, 64, 128, 64, 32, 64, 4, 34, 60);
  CFG(128, 128, 64, 64, 64, 32, 4, 34, 60);
  CFG(128, 128, 128, 64, 64, 64, 3, 34, 60);
  // C=96 sur 68x120
  CFG(96, 64, 96, 64, 32, 48, 4, 68, 120);
  CFG(96, 128, 96, 64, 64, 48, 4, 68, 120);
  CFG(96, 128, 96, 64, 64, 48, 3, 68, 120);
  CFG(96, 64, 96, 64, 32, 48, 4, 68, 120);
  CFG(96, 128, 96, 64, 64, 96, 3, 68, 120);
  return 0;
}
