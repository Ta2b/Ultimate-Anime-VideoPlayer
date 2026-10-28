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

#include "common.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/numeric_types.h"
#include "span_kernels.h"

namespace {
using namespace span;

struct Result {
  bool ok = false;
  double ms = 0, ms_min = 0, tflops = 0;
  Err e_kernel, e_true;
};

double median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Données du bloc : entrée de la conv (c2 après Relu), saut x, poids 3x3, 1x1, biais.
struct Block {
  int H, W;
  Data d;  // d.x = entrée de la conv, d.w = poids 3x3, d.b = biais 3x3
  std::vector<float> skip, w1, b1;
};

Block make_block(int H, int W, unsigned seed) {
  Block b;
  b.H = H;
  b.W = W;
  b.d = make_data(H, W, seed);
  std::mt19937 g(seed + 100);
  std::uniform_real_distribution<float> u(0.f, 1.f), ub(-0.1f, 0.1f);
  std::normal_distribution<float> n(0.f, std::sqrt(1.f / kC));
  b.skip.resize(b.d.x.size());
  for (auto& v : b.skip) v = u(g);
  b.w1.resize(kC * kC);
  for (auto& v : b.w1) v = n(g);
  b.b1.resize(kC);
  for (auto& v : b.b1) v = ub(g);
  return b;
}

__global__ void tail_ref_kernel(const float* c3, const float* skip, const float* w1, const float* b1,
                                float* out, size_t npix) {
  size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= npix) return;
  float c[kC];
  for (int k = 0; k < kC; ++k) c[k] = c3[p * kC + k];
  for (int j = 0; j < kC; ++j) {
    float acc = b1[j];
    for (int k = 0; k < kC; ++k) acc = fmaf(w1[j * kC + k], c[k], acc);
    out[p * kC + j] = (c[j] + skip[p * kC + j]) * tanhf(acc);
  }
}

// Référence fp32 GPU du bloc (conv 3x3 sans Relu, puis fin de bloc)
void block_ref(const std::vector<float>& x, const std::vector<float>& w3, const std::vector<float>& b3,
               const std::vector<float>& skip, const std::vector<float>& w1, const std::vector<float>& b1,
               std::vector<float>& out, int H, int W) {
  std::vector<float> c3;
  conv_gpu_ref<false>(x, w3, b3, c3, H, W);
  size_t n = c3.size(), npix = size_t(H) * W;
  float *dc, *ds, *dw, *db, *dy;
  CK(cudaMalloc(&dc, n * 4));
  CK(cudaMalloc(&ds, n * 4));
  CK(cudaMalloc(&dw, w1.size() * 4));
  CK(cudaMalloc(&db, b1.size() * 4));
  CK(cudaMalloc(&dy, n * 4));
  CK(cudaMemcpy(dc, c3.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(ds, skip.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw, w1.data(), w1.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, b1.data(), b1.size() * 4, cudaMemcpyHostToDevice));
  tail_ref_kernel<<<(npix + 127) / 128, 128>>>(dc, ds, dw, db, dy, npix);
  CK(cudaGetLastError());
  out.resize(n);
  CK(cudaMemcpy(out.data(), dy, n * 4, cudaMemcpyDeviceToHost));
  cudaFree(dc);
  cudaFree(ds);
  cudaFree(dw);
  cudaFree(db);
  cudaFree(dy);
}

template <class C, int Mode>
Result run(const Block& blk, bool timing) {
  using E = typename C::E;
  using Base = typename C::Base;
  Result res;
  const int H = blk.H, W = blk.W;
  const size_t n = blk.d.x.size();
  auto amax = [](const std::vector<float>& v) {
    float m = 0;
    for (float f : v) m = std::max(m, std::fabs(f));
    return m;
  };
  float sx = amax(blk.d.x) / 448.f, sw = amax(blk.d.w) / 448.f, sk = amax(blk.skip) / 448.f;
  std::vector<E> hx(n), hw(blk.d.w.size());
  std::vector<uint8_t> hs(n);
  std::vector<float> xq(n), wq(blk.d.w.size()), skq(n), w1q(blk.w1.size());
  for (size_t i = 0; i < n; ++i) {
    hx[i] = E(blk.d.x[i] / sx);
    xq[i] = float(hx[i]) * sx;
    E s(blk.skip[i] / sk);
    hs[i] = s.storage;
    skq[i] = float(s) * sk;
  }
  for (size_t i = 0; i < hw.size(); ++i) {
    hw[i] = E(blk.d.w[i] / sw);
    wq[i] = float(hw[i]) * sw;
  }
  std::vector<__half> hw1(blk.w1.size());
  for (size_t i = 0; i < hw1.size(); ++i) {
    hw1[i] = __float2half(blk.w1[i]);
    w1q[i] = __half2float(hw1[i]);
  }

  std::vector<float> ref_true, ref_q;
  if (Mode == 1) {
    block_ref(blk.d.x, blk.d.w, blk.d.b, blk.skip, blk.w1, blk.b1, ref_true, H, W);
    block_ref(xq, wq, blk.d.b, skq, w1q, blk.b1, ref_q, H, W);
  } else {
    conv_gpu_ref(blk.d.x, blk.d.w, blk.d.b, ref_true, H, W);
    conv_gpu_ref(xq, wq, blk.d.b, ref_q, H, W);
  }
  float sy = amax(ref_true) / 448.f;
  for (auto& v : ref_q) v = float(E(v / sy)) * sy;  // la sortie du noyau est arrondie en E4M3

  E *dx, *dw;
  uint8_t *dskip, *dout;
  __half* dw1;
  float *db3, *db1;
  CK(cudaMalloc(&dx, n));
  CK(cudaMalloc(&dw, hw.size()));
  CK(cudaMalloc(&dskip, n));
  CK(cudaMalloc(&dout, n));
  CK(cudaMalloc(&dw1, hw1.size() * 2));
  CK(cudaMalloc(&db3, kC * 4));
  CK(cudaMalloc(&db1, kC * 4));
  CK(cudaMemcpy(dx, hx.data(), n, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dskip, hs.data(), n, cudaMemcpyHostToDevice));
  CK(cudaMemset(dout, 0, n));
  CK(cudaMemcpy(dw1, hw1.data(), hw1.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db3, blk.d.b.data(), kC * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db1, blk.b1.data(), kC * 4, cudaMemcpyHostToDevice));

  Launcher<C, Mode> L;
  L.init(H, W, dx, dw, (E*)dout, TailParams{dskip, sk, sx * sw, db3, dw1, db1, dout, 1.f / sy});
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  L.launch(s);
  CK(cudaStreamSynchronize(s));
  CK(cudaGetLastError());

  std::vector<uint8_t> ho(n);
  CK(cudaMemcpy(ho.data(), dout, n, cudaMemcpyDeviceToHost));
  std::vector<float> got(n);
  for (size_t i = 0; i < n; ++i) {
    E e;
    e.storage = ho[i];
    got[i] = float(e) * sy;
  }
  res.e_kernel = compare(ref_q, got);
  res.e_true = compare(ref_true, got);
  res.ok = true;

  if (timing) {
    cudaGraph_t gr;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    L.launch(s);
    CK(cudaStreamEndCapture(s, &gr));
    CK(cudaGraphInstantiate(&ge, gr, 0));
    for (int i = 0; i < 200; ++i) CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    std::vector<cudaEvent_t> ev(400);
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
    double flop = 2.0 * H * W * kC * kC * 9 + (Mode == 1 ? 2.0 * H * W * kC * kC : 0);
    res.tflops = flop / (res.ms * 1e-3) / 1e12;
  }
  cudaStreamDestroy(s);
  cudaFree(dx);
  cudaFree(dw);
  cudaFree(dskip);
  cudaFree(dout);
  cudaFree(dw1);
  cudaFree(db3);
  cudaFree(db1);
  return res;
}

struct Entry {
  std::string name;
  std::function<Result(const Block&, bool)> fn;
};

#define ENTRY(NAME, MODE, ...)                                                         \
  Entry {                                                                              \
    NAME, [](const Block& b, bool t) { return run<TailCfg<__VA_ARGS__>, MODE>(b, t); } \
  }

}  // namespace

int main(int argc, char** argv) {
  std::string only = argc > 1 ? argv[1] : "";
  std::vector<Entry> es = {
      ENTRY("conv seule 128x64x64 w32x64 s3", 0, 128, 64, 32, 3),
      ENTRY("bloc    128x64x64 w32x64 s3", 1, 128, 64, 32, 3),
      ENTRY("conv seule 128x64x64 w32x64 s4", 0, 128, 64, 32, 4),
      ENTRY("bloc    128x64x64 w32x64 s4", 1, 128, 64, 32, 4),
      ENTRY("conv seule 64x64x64 w16x64 s4", 0, 64, 64, 16, 4),
      ENTRY("bloc    64x64x64 w16x64 s4", 1, 64, 64, 16, 4),
  };
  std::printf("== Petit tenseur 1x64x64x64 : correction ==\n");
  Block small = make_block(64, 64, 1);
  for (auto& e : es) {
    if (!only.empty() && e.name.find(only) == std::string::npos) continue;
    Result r = e.fn(small, false);
    if (r.ok)
      std::printf("  %-34s noyau: max %.3g PSNR %.1f dB | vs fp32 d'origine: max %.3g PSNR %.1f dB\n",
                  e.name.c_str(), r.e_kernel.max_abs, r.e_kernel.psnr, r.e_true.max_abs, r.e_true.psnr);
  }
  std::printf("== Grand tenseur 1x64x544x960 ==\n");
  Block big = make_block(544, 960, 2);
  std::printf("%-34s %9s %9s %8s | %-22s | %-22s\n", "config", "ms méd.", "ms min", "TFLOPS",
              "noyau max / PSNR", "vs fp32 max / PSNR");
  for (auto& e : es) {
    if (!only.empty() && e.name.find(only) == std::string::npos) continue;
    Result r = e.fn(big, true);
    if (r.ok)
      std::printf("%-34s %9.4f %9.4f %8.1f | %9.3g / %6.1f dB | %9.3g / %6.1f dB\n", e.name.c_str(), r.ms,
                  r.ms_min, r.tflops, r.e_kernel.max_abs, r.e_kernel.psnr, r.e_true.max_abs, r.e_true.psnr);
  }
  return 0;
}
