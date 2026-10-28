// Utilitaires communs aux prototypes CUTLASS : données aléatoires, références conv 3x3 (CPU et GPU),
// métriques d'erreur.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(2);                                                                        \
    }                                                                                      \
  } while (0)

static constexpr int kC = 64;  // canaux d'entrée et de sortie

// ---------------------------------------------------------------- références ----

// Convolution 3x3 naïve fp32, NHWC, filtre KRSC, remplissage 1, biais + Relu.
static void conv_cpu(const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>& b,
                     std::vector<float>& y, int H, int W) {
  y.assign(size_t(H) * W * kC, 0.f);
  for (int h = 0; h < H; ++h)
    for (int ww = 0; ww < W; ++ww)
      for (int k = 0; k < kC; ++k) {
        double acc = b[k];
        for (int r = 0; r < 3; ++r)
          for (int s = 0; s < 3; ++s) {
            int ih = h + r - 1, iw = ww + s - 1;
            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
            const float* xp = &x[(size_t(ih) * W + iw) * kC];
            const float* wp = &w[((size_t(k) * 3 + r) * 3 + s) * kC];
            for (int c = 0; c < kC; ++c) acc += double(xp[c]) * wp[c];
          }
        y[(size_t(h) * W + ww) * kC + k] = float(std::max(acc, 0.0));
      }
}

template <bool Relu>
__global__ void conv_ref_kernel(const float* x, const float* w, const float* b, float* y, int H, int W) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(H) * W * kC) return;
  int k = idx % kC;
  size_t p = idx / kC;
  int ww = p % W, h = p / W;
  float acc = b[k];
  for (int r = 0; r < 3; ++r)
    for (int s = 0; s < 3; ++s) {
      int ih = h + r - 1, iw = ww + s - 1;
      if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
      const float* xp = x + (size_t(ih) * W + iw) * kC;
      const float* wp = w + ((size_t(k) * 3 + r) * 3 + s) * kC;
      for (int c = 0; c < kC; ++c) acc = fmaf(xp[c], wp[c], acc);
    }
  y[idx] = Relu ? fmaxf(acc, 0.f) : acc;
}

template <bool Relu = true>
static void conv_gpu_ref(const std::vector<float>& x, const std::vector<float>& w,
                         const std::vector<float>& b, std::vector<float>& y, int H, int W) {
  float *dx, *dw, *db, *dy;
  size_t n = size_t(H) * W * kC;
  CK(cudaMalloc(&dx, n * 4));
  CK(cudaMalloc(&dw, w.size() * 4));
  CK(cudaMalloc(&db, b.size() * 4));
  CK(cudaMalloc(&dy, n * 4));
  CK(cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
  conv_ref_kernel<Relu><<<(n + 255) / 256, 256>>>(dx, dw, db, dy, H, W);
  CK(cudaGetLastError());
  y.resize(n);
  CK(cudaMemcpy(y.data(), dy, n * 4, cudaMemcpyDeviceToHost));
  cudaFree(dx);
  cudaFree(dw);
  cudaFree(db);
  cudaFree(dy);
}

struct Err {
  double max_abs = 0, psnr = 0, rel_rms = 0;
};
static Err compare(const std::vector<float>& ref, const std::vector<float>& got) {
  double mse = 0, mx = 0, peak = 0, e2 = 0, r2 = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double d = double(got[i]) - ref[i];
    mse += d * d;
    mx = std::max(mx, std::fabs(d));
    peak = std::max(peak, double(std::fabs(ref[i])));
    e2 += d * d;
    r2 += double(ref[i]) * ref[i];
  }
  mse /= ref.size();
  Err e;
  e.max_abs = mx;
  e.psnr = mse > 0 ? 10 * std::log10(peak * peak / mse) : 200.0;
  e.rel_rms = std::sqrt(e2 / std::max(r2, 1e-30));
  return e;
}

// ---------------------------------------------------------------- données ----

struct Data {
  int H, W;
  std::vector<float> x, w, b;  // valeurs d'origine fp32
};

static Data make_data(int H, int W, unsigned seed) {
  Data d;
  d.H = H;
  d.W = W;
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> u(0.f, 1.f), ub(-0.1f, 0.1f);
  std::normal_distribution<float> n(0.f, std::sqrt(2.f / (9 * kC)));
  d.x.resize(size_t(H) * W * kC);
  d.w.resize(size_t(kC) * 9 * kC);
  d.b.resize(kC);
  for (auto& v : d.x) v = u(g);
  for (auto& v : d.w) v = n(g);
  for (auto& v : d.b) v = ub(g);
  return d;
}
