// Bloc SPAN complet en FP8 E4M3 avec CUTLASS : c1 Conv3x3+Relu, c2 Conv3x3+Relu, puis c3 + saut + attention
// fusionnés. Trois lancements : deux noyaux « conv + biais + Relu -> E4M3 » et le noyau de fin de bloc
// (span_kernels.h). Lit un dossier de fichiers float32 écrit par real_block.py (x, poids KRSC, biais,
// échelles), écrit la sortie déquantifiée et chronomètre la chaîne (graphe CUDA, médiane de 200 lancements
// après 200 de chauffe).
//
//   span_block DOSSIER
#include <cstdint>
#include <cuda_fp16.h>
#include <fstream>
#include <string>

#include "common.h"
#include "span_kernels.h"

using namespace span;

namespace {

std::vector<float> read_f32(const std::string& path, size_t n) {
  std::vector<float> v(n);
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "fichier illisible : %s\n", path.c_str());
    std::exit(1);
  }
  f.read(reinterpret_cast<char*>(v.data()), n * 4);
  if (size_t(f.gcount()) != n * 4) {
    std::fprintf(stderr, "fichier trop court : %s\n", path.c_str());
    std::exit(1);
  }
  return v;
}

void write_f32(const std::string& path, const std::vector<float>& v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), v.size() * 4);
}

using Cfg =
    TailCfg<128, 64, 32, 3>;  // bloc 128x64x64, warp 32x64, 3 étages : meilleure config mesurée (span_tail)
using E = Cfg::E;

float amax(const std::vector<float>& v) {
  float m = 0;
  for (float f : v) m = std::max(m, std::fabs(f));
  return m;
}

// Quantifie en E4M3 (échelle amax/448) ; renvoie l'échelle
float quantize(const std::vector<float>& v, std::vector<E>& q) {
  float s = amax(v) / 448.f;
  q.resize(v.size());
  for (size_t i = 0; i < v.size(); ++i) q[i] = E(v[i] / s);
  return s;
}

template <class T>
T* upload(const std::vector<T>& v) {
  T* d;
  CK(cudaMalloc(&d, v.size() * sizeof(T)));
  CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage : span_block DOSSIER\n");
    return 1;
  }
  const std::string dir = std::string(argv[1]) + "/";
  std::vector<float> dims = read_f32(dir + "dims.f32", 2);
  const int H = int(dims[0]), W = int(dims[1]);
  const size_t n = size_t(H) * W * kC;
  std::vector<float> sc = read_f32(dir + "scales.f32", 4);  // x, relu1, relu2, sortie
  const float sx = sc[0], s1 = sc[1], s2 = sc[2], so = sc[3];

  std::vector<float> x = read_f32(dir + "x.f32", n);
  std::vector<E> xq, w1q, w2q, w3q;
  for (size_t i = 0; i < n; ++i) {  // x est quantifié avec l'échelle imposée (la même sert de saut)
    xq.push_back(E(x[i] / sx));
  }
  auto w1 = read_f32(dir + "w1.f32", kC * 9 * kC), w2 = read_f32(dir + "w2.f32", kC * 9 * kC),
       w3 = read_f32(dir + "w3.f32", kC * 9 * kC);
  float sw1 = quantize(w1, w1q), sw2 = quantize(w2, w2q), sw3 = quantize(w3, w3q);
  auto b1 = read_f32(dir + "b1.f32", kC), b2 = read_f32(dir + "b2.f32", kC),
       b3 = read_f32(dir + "b3.f32", kC);
  auto wa = read_f32(dir + "wa.f32", kC * kC), ba = read_f32(dir + "ba.f32", kC);
  std::vector<__half> wah(wa.size());
  for (size_t i = 0; i < wa.size(); ++i) wah[i] = __float2half(wa[i]);

  E *dx = upload(xq), *dw1 = upload(w1q), *dw2 = upload(w2q), *dw3 = upload(w3q);
  float *db1 = upload(b1), *db2 = upload(b2), *db3 = upload(b3), *dba = upload(ba);
  __half* dwa = upload(wah);
  uint8_t *dr1, *dr2, *dout;
  CK(cudaMalloc(&dr1, n));
  CK(cudaMalloc(&dr2, n));
  CK(cudaMalloc(&dout, n));

  Launcher<Cfg, 0> k1, k2;
  Launcher<Cfg, 1> k3;
  k1.init(H, W, dx, dw1, (E*)dr1, TailParams{nullptr, 0.f, sx * sw1, db1, nullptr, nullptr, dr1, 1.f / s1});
  k2.init(H, W, (E*)dr1, dw2, (E*)dr2,
          TailParams{nullptr, 0.f, s1 * sw2, db2, nullptr, nullptr, dr2, 1.f / s2});
  k3.init(H, W, (E*)dr2, dw3, (E*)dout,
          TailParams{reinterpret_cast<const uint8_t*>(dx), sx, s2 * sw3, db3, dwa, dba, dout, 1.f / so});

  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  k1.launch(s);
  k2.launch(s);
  k3.launch(s);
  CK(cudaStreamSynchronize(s));
  CK(cudaGetLastError());

  auto fetch = [&](uint8_t* d, float scale, const std::string& name) {
    std::vector<uint8_t> h(n);
    CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost));
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) {
      E e;
      e.storage = h[i];
      f[i] = float(e) * scale;
    }
    write_f32(dir + name, f);
  };
  fetch(dr1, s1, "r1_out.f32");
  fetch(dout, so, "out.f32");

  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  k1.launch(s);
  k2.launch(s);
  k3.launch(s);
  CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
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
  std::sort(t.begin(), t.end());
  double flop = 3.0 * 2.0 * H * W * kC * kC * 9 + 2.0 * H * W * kC * kC;
  std::printf("bloc SPAN complet (3 noyaux CUTLASS) %dx%d : médiane %.4f ms, min %.4f ms, %.1f TFLOPS\n", H,
              W, t[100], t[0], flop / (t[100] * 1e-3) / 1e12);
  return 0;
}
