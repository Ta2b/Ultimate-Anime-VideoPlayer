// Banc isolé du noyau fusionné (fp8_fused.cu) : correction contre une référence CPU en double,
// bande passante contre la borne mémoire de la RTX 4070 Ti (504 Go/s). Tenseur 544x960x64.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_fp8.h>
#include <random>
#include <vector>

#include "../../../../src/gpu/cuda/backend/fp8_fused.cuh"

using namespace ajp::gpu::cuda;

static float fp8_to_f(uint8_t b) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
  return __half2float(__half(h));
}
#define CK(x)                                                          \
  do {                                                                 \
    cudaError_t e = (x);                                               \
    if (e != cudaSuccess) {                                            \
      printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      return 1;                                                        \
    }                                                                  \
  } while (0)

int main() {
  const size_t n = 544ull * 960 * 64;
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<uint8_t> c3(n), sk(n), att8(n);
  std::vector<__half> att(n);
  Fp8SpanTailScales sc{0.02f, 0.03f, 1.f / 0.05f};
  const float s_att = 0.01f;
  for (size_t i = 0; i < n; ++i) {
    c3[i] = __nv_cvt_float_to_fp8(nd(rng) * 0.3f / 0.02f, __NV_SATFINITE, __NV_E4M3);
    sk[i] = __nv_cvt_float_to_fp8(nd(rng) * 0.3f / 0.03f, __NV_SATFINITE, __NV_E4M3);
    att[i] = __float2half(nd(rng) * 1.5f);
    att8[i] = __nv_cvt_float_to_fp8(nd(rng) * 1.5f / s_att, __NV_SATFINITE, __NV_E4M3);
  }
  uint8_t *dc, *ds, *da8, *dout;
  __half* da;
  CK(cudaMalloc(&dc, n));
  CK(cudaMalloc(&ds, n));
  CK(cudaMalloc(&da8, n));
  CK(cudaMalloc(&dout, n));
  CK(cudaMalloc(&da, n * 2));
  CK(cudaMemcpy(dc, c3.data(), n, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(ds, sk.data(), n, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(da8, att8.data(), n, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(da, att.data(), n * 2, cudaMemcpyHostToDevice));
  std::vector<uint8_t> out(n);

  auto check = [&](const char* nom, bool att_is8) {
    cudaMemcpy(out.data(), dout, n, cudaMemcpyDeviceToHost);
    double mx = 0, se = 0, sig = 0;
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i) {
      double a = att_is8 ? fp8_to_f(att8[i]) * (double)s_att : (double)__half2float(att[i]);
      double ref = (fp8_to_f(c3[i]) * (double)sc.c3 + fp8_to_f(sk[i]) * (double)sc.skip) * std::tanh(a);
      double got = fp8_to_f(out[i]) / sc.out_inv;
      // Référence idéale = arrondi E4M3 de la valeur exacte ; un octet d'écart n'est pas une erreur.
      uint8_t q = __nv_cvt_double_to_fp8(ref * sc.out_inv, __NV_SATFINITE, __NV_E4M3);
      double refq = fp8_to_f(q) / sc.out_inv;
      mx = std::max(mx, std::fabs(got - refq));
      diff += (q != out[i]);
      se += (got - ref) * (got - ref);
      sig += ref * ref;
    }
    double mse = se / n, rms = std::sqrt(sig / n);
    printf(
        "%-22s max|got-refQ|=%.3e  octets différents de l'arrondi exact=%.4f %%  erreur rms/rms signal=%.4f  "
        "PSNR(pic=4 rms)=%.1f dB\n",
        nom, mx, 100.0 * diff / n, std::sqrt(mse) / rms, 10 * std::log10(16 * rms * rms / mse));
  };
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  // L2 de 48 Mo sur AD104 : on la vide (écriture de 256 Mo) avant chaque passe mesurée.
  uint8_t* flush;
  size_t fl = 256ull << 20;
  CK(cudaMalloc(&flush, fl));
  auto bench = [&](const char* nom, double bytes, auto fn) {
    for (int i = 0; i < 20; ++i) fn();
    std::vector<float> t;
    for (int r = 0; r < 100; ++r) {
      cudaMemsetAsync(flush, r, fl, st);
      cudaEventRecord(e0, st);
      fn();
      cudaEventRecord(e1, st);
      cudaEventSynchronize(e1);
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    double med = t[t.size() / 2];
    printf("%-22s médiane %.4f ms  min %.4f  %.0f Go/s  (borne mémoire %.4f ms à 504 Go/s)\n", nom, med, t[0],
           bytes / med / 1e6, bytes / 504e6);
  };
  CK(launch_fp8_span_tail(dc, da, ds, dout, n, sc, true, st));
  CK(cudaStreamSynchronize(st));
  check("att fp16, tanh rapide", false);
  CK(launch_fp8_span_tail(dc, da, ds, dout, n, sc, false, st));
  CK(cudaStreamSynchronize(st));
  check("att fp16, tanhf", false);
  CK(launch_fp8_span_tail_att8(dc, da8, ds, dout, n, sc, s_att, st));
  CK(cudaStreamSynchronize(st));
  check("att fp8", true);
  bench("att fp16 (5 o/elt)", 5.0 * n, [&] { launch_fp8_span_tail(dc, da, ds, dout, n, sc, true, st); });
  bench("att fp16 tanhf", 5.0 * n, [&] { launch_fp8_span_tail(dc, da, ds, dout, n, sc, false, st); });
  bench("att fp8 (4 o/elt)", 4.0 * n,
        [&] { launch_fp8_span_tail_att8(dc, da8, ds, dout, n, sc, s_att, st); });
  return 0;
}
