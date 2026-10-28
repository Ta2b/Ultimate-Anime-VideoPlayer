// SPAN entier (AnimeJaNai SPANF3 b8f64 « unshuffle ») hors TensorRT, en un graphe CUDA :
//   dégroupage 3->12 canaux (demi-résolution) + remplissage à 16, conv_1 3x3 12->64 (FP16),
//   8 blocs SPAB en FP8 E4M3 (conv3x3+Relu, conv3x3+Relu, fin de bloc fusionnée),
//   concaténation [conv_1, bloc 8, bloc 1, relu1 du bloc 8] écrite directement par les noyaux dans un tampon
//   FP16 de 256 canaux, conv_cat 1x1 256->64 (FP16), to_pixel 3x3 64->48 (FP16), DepthToSpace(4, mode CRD).
// Les parties hors blocs sont en FP16 (accumulateur f32, ou f16 avec -DACC16).
//
//   span_net DOSSIER [--sans-ecriture]
// DOSSIER contient les fichiers float32 écrits par span_net.py (entrée NCHW, poids KRSC, échelles).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "span_net_kernels.h"

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(2);                                                                        \
    }                                                                                      \
  } while (0)

using namespace net;
using E4 = cutlass::float_e4m3_t;
using F16 = cutlass::half_t;

#ifdef ACC16
using FAcc = cutlass::half_t;
#else
using FAcc = float;
#endif

namespace {

constexpr int H = 1088, W = 1920;    // image d'entrée
constexpr int h = H / 2, w = W / 2;  // résolution des caractéristiques
constexpr int NPIX = h * w;

std::string dir;

std::vector<float> read_f32(const std::string& name, size_t n) {
  std::vector<float> v(n);
  std::ifstream f(dir + name, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "fichier illisible : %s\n", (dir + name).c_str());
    std::exit(1);
  }
  f.read(reinterpret_cast<char*>(v.data()), n * 4);
  if (size_t(f.gcount()) != n * 4) {
    std::fprintf(stderr, "fichier trop court : %s\n", name.c_str());
    std::exit(1);
  }
  return v;
}

template <class T>
T* upload(const std::vector<T>& v) {
  T* d;
  CK(cudaMalloc(&d, v.size() * sizeof(T)));
  CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

template <class T>
T* alloc(size_t n) {
  T* d;
  CK(cudaMalloc(&d, n * sizeof(T)));
  CK(cudaMemset(d, 0, n * sizeof(T)));
  return d;
}

float amax(const std::vector<float>& v) {
  float m = 0;
  for (float f : v) m = std::max(m, std::fabs(f));
  return m;
}

// Poids FP8 : échelle par tenseur amax/448
float quant8(const std::vector<float>& v, std::vector<E4>& q) {
  float s = amax(v) / 448.f;
  q.resize(v.size());
  for (size_t i = 0; i < v.size(); ++i) q[i] = E4(v[i] / s);
  return s;
}

std::vector<F16> to_f16(const std::vector<float>& v) {
  std::vector<F16> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = F16(v[i]);
  return o;
}

// Dégroupage : NCHW FP16 3xHxW -> NHWC 16 canaux (canal = c*4 + a*2 + b ; 12..15 = 0)
__global__ void unshuffle_kernel(const __half* in, __half* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= NPIX) return;
  int y = idx / w, x = idx % w;
  __half v[16];
#pragma unroll
  for (int c = 0; c < 3; ++c)
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int b = 0; b < 2; ++b) v[c * 4 + a * 2 + b] = in[(size_t(c) * H + 2 * y + a) * W + 2 * x + b];
#pragma unroll
  for (int i = 12; i < 16; ++i) v[i] = __float2half(0.f);
  uint4* o = reinterpret_cast<uint4*>(out + size_t(idx) * 16);
  o[0] = *reinterpret_cast<uint4*>(&v[0]);
  o[1] = *reinterpret_cast<uint4*>(&v[8]);
}

// DepthToSpace(4, mode CRD du modèle) : NHWC 48 canaux -> NCHW 3 x 4h x 4w ; canal d'entrée = c*16 + bh*4 +
// bw
__global__ void d2s_kernel(const __half* in, __half* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // (pixel, bh, c)
  if (idx >= NPIX * 12) return;
  int c = idx % 3, bh = (idx / 3) % 4, pix = idx / 12;
  int y = pix / w, x = pix % w;
  const __half* p = in + size_t(pix) * 48 + c * 16 + bh * 4;
  __half o[4];
#pragma unroll
  for (int bw = 0; bw < 4; ++bw) o[bw] = p[bw];
  *reinterpret_cast<uint2*>(out + (size_t(c) * 4 * h + 4 * y + bh) * (4 * w) + 4 * x) =
      *reinterpret_cast<uint2*>(o);
}

using C8 = Cfg<E4, 128, 64, 32, 3>;
using L8 = Launcher<C8, 0, true>;
using LT = Launcher<C8, 1, false>;
using C16A = Cfg<F16, 128, 32, 32, 4, FAcc, cutlass::conv::IteratorAlgorithm::kAnalytic>;
using C16 = Cfg<F16, 128, 32, 32, 4, FAcc>;
using LC1 = Launcher<C16A, 0, false>;
using LCAT = Launcher<C16, 0, false>;

struct Stage {
  std::string name;
  std::function<void(cudaStream_t)> run;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage : span_net DOSSIER\n");
    return 1;
  }
  dir = std::string(argv[1]) + "/";
  auto sc = read_f32("scales.f32", 25);  // conv_1, puis (relu1, relu2, sortie) de chaque bloc

  // --- poids
  auto wc1 = read_f32("w_c1.f32", 64 * 9 * 16), bc1 = read_f32("b_c1.f32", 64);
  auto wcat = read_f32("w_cat.f32", 64 * 256), bcat = read_f32("b_cat.f32", 64);
  auto wpx = read_f32("w_pix.f32", 48 * 9 * 64), bpx = read_f32("b_pix.f32", 48);
  F16 *d_wc1 = upload(to_f16(wc1)), *d_wcat = upload(to_f16(wcat)), *d_wpx = upload(to_f16(wpx));
  float *d_bc1 = upload(bc1), *d_bcat = upload(bcat), *d_bpx = upload(bpx);

  struct Blk {
    E4* w[3];
    float* b[3];
    __half* wa;
    float* ba;
    float sw[3];
  } blk[8];
  for (int k = 0; k < 8; ++k) {
    for (int i = 0; i < 3; ++i) {
      std::vector<E4> q;
      auto wv = read_f32("blk" + std::to_string(k + 1) + "_w" + std::to_string(i + 1) + ".f32", 64 * 9 * 64);
      blk[k].sw[i] = quant8(wv, q);
      blk[k].w[i] = upload(q);
      blk[k].b[i] =
          upload(read_f32("blk" + std::to_string(k + 1) + "_b" + std::to_string(i + 1) + ".f32", 64));
    }
    auto wa = read_f32("blk" + std::to_string(k + 1) + "_wa.f32", 64 * 64);
    std::vector<__half> wah(wa.size());
    for (size_t i = 0; i < wa.size(); ++i) wah[i] = __float2half(wa[i]);
    blk[k].wa = upload(wah);
    blk[k].ba = upload(read_f32("blk" + std::to_string(k + 1) + "_ba.f32", 64));
  }

  // --- entrée
  auto xin = read_f32("input.f32", size_t(3) * H * W);
  std::vector<__half> xh(xin.size());
  for (size_t i = 0; i < xin.size(); ++i) xh[i] = __float2half(xin[i]);
  __half* d_in = upload(xh);
  __half* d_u = alloc<__half>(size_t(NPIX) * 16);
  uint8_t* X[2] = {alloc<uint8_t>(size_t(NPIX) * 64), alloc<uint8_t>(size_t(NPIX) * 64)};
  uint8_t* R1 = alloc<uint8_t>(size_t(NPIX) * 64);
  uint8_t* R2 = alloc<uint8_t>(size_t(NPIX) * 64);
  __half* cat_in = alloc<__half>(size_t(NPIX) * 256);
  __half* cat_out = alloc<__half>(size_t(NPIX) * 64);
  __half* pix_out = alloc<__half>(size_t(NPIX) * 48);
  __half* d_out = alloc<__half>(size_t(3) * 4 * h * 4 * w);

  std::vector<Stage> st;
  st.push_back({"dégroupage",
                [&](cudaStream_t s) { unshuffle_kernel<<<(NPIX + 255) / 256, 256, 0, s>>>(d_in, d_u); }});

  // conv_1 : sorties FP8 (entrée du bloc 1, échelle sc[0]) et FP16 (tranche 0 du concat)
  static LC1 c1;
  {
    KP k;
    k.bias = d_bc1;
    k.out8 = X[0];
    k.inv_s8 = 1.f / sc[0];
    k.out16 = cat_in;
    k.ld16 = 256;
    c1.init(h, w, 3, 1, 16, 64, reinterpret_cast<const F16*>(d_u), d_wc1, k);
  }
  st.push_back({"conv_1", [&](cudaStream_t s) { c1.launch(s); }});

  static L8 la[8], lb[8];
  static LT lc[8];
  for (int b = 0; b < 8; ++b) {
    float s_in = b == 0 ? sc[0] : sc[1 + 3 * (b - 1) + 2];
    float s_r1 = sc[1 + 3 * b], s_r2 = sc[2 + 3 * b], s_out = sc[3 + 3 * b];
    uint8_t* xin8 = X[b % 2];
    uint8_t* xout8 = X[(b + 1) % 2];
    {  // c1 + Relu -> FP8 ; pour le bloc 8, aussi FP16 dans la tranche 3 du concat
      KP k;
      k.alpha = s_in * blk[b].sw[0];
      k.bias = blk[b].b[0];
      k.out8 = R1;
      k.inv_s8 = 1.f / s_r1;
      if (b == 7) {
        k.out16 = cat_in + 192;
        k.ld16 = 256;
      }
      la[b].init(h, w, 3, 1, 64, 64, reinterpret_cast<const E4*>(xin8), blk[b].w[0], k);
    }
    {
      KP k;
      k.alpha = s_r1 * blk[b].sw[1];
      k.bias = blk[b].b[1];
      k.out8 = R2;
      k.inv_s8 = 1.f / s_r2;
      lb[b].init(h, w, 3, 1, 64, 64, reinterpret_cast<const E4*>(R1), blk[b].w[1], k);
    }
    {
      KP k;
      k.skip = xin8;
      k.skip_scale = s_in;
      k.alpha = s_r2 * blk[b].sw[2];
      k.bias = blk[b].b[2];
      k.w1 = blk[b].wa;
      k.b1 = blk[b].ba;
      if (b < 7) {
        k.out8 = xout8;
        k.inv_s8 = 1.f / s_out;
      }
      if (b == 0) {  // sortie du bloc 1 : tranche 2 du concat
        k.out16 = cat_in + 128;
        k.ld16 = 256;
      }
      if (b == 7) {  // sortie du bloc 8 : tranche 1
        k.out16 = cat_in + 64;
        k.ld16 = 256;
      }
      lc[b].init(h, w, 3, 1, 64, 64, reinterpret_cast<const E4*>(R2), blk[b].w[2], k);
    }
    st.push_back({"bloc c1", [b](cudaStream_t s) { la[b].launch(s); }});
    st.push_back({"bloc c2", [b](cudaStream_t s) { lb[b].launch(s); }});
    st.push_back({"bloc fin", [b](cudaStream_t s) { lc[b].launch(s); }});
  }

  static LCAT lcat, lpix;
  {
    KP k;
    k.bias = d_bcat;
    k.out16 = cat_out;
    k.ld16 = 64;
    lcat.init(h, w, 1, 0, 256, 64, reinterpret_cast<const F16*>(cat_in), d_wcat, k);
    KP p;
    p.bias = d_bpx;
    p.out16 = pix_out;
    p.ld16 = 48;
    lpix.init(h, w, 3, 1, 64, 48, reinterpret_cast<const F16*>(cat_out), d_wpx, p);
  }
  st.push_back({"conv_cat 1x1", [&](cudaStream_t s) { lcat.launch(s); }});
  st.push_back({"to_pixel 3x3", [&](cudaStream_t s) { lpix.launch(s); }});
  st.push_back({"DepthToSpace",
                [&](cudaStream_t s) { d2s_kernel<<<(NPIX * 12 + 255) / 256, 256, 0, s>>>(pix_out, d_out); }});

  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  for (auto& x : st) x.run(s);
  CK(cudaStreamSynchronize(s));
  CK(cudaGetLastError());

  // sortie
  {
    std::vector<__half> ho(size_t(3) * 4 * h * 4 * w);
    CK(cudaMemcpy(ho.data(), d_out, ho.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> fo(ho.size());
    for (size_t i = 0; i < ho.size(); ++i) fo[i] = __half2float(ho[i]);
    std::ofstream f(dir + "out.f32", std::ios::binary);
    f.write(reinterpret_cast<const char*>(fo.data()), fo.size() * 4);
  }

  // débogage : SPAN_DUMP=1 écrit le tampon de concaténation (FP16, NHWC 256 canaux), la sortie de conv_cat et
  // de to_pixel
  if (std::getenv("SPAN_DUMP")) {
    auto dump = [&](const char* name, const __half* d, size_t n) {
      std::vector<__half> hh(n);
      CK(cudaMemcpy(hh.data(), d, n * 2, cudaMemcpyDeviceToHost));
      std::ofstream f(dir + name, std::ios::binary);
      f.write(reinterpret_cast<const char*>(hh.data()), n * 2);
    };
    dump("dump_cat_in.f16", cat_in, size_t(NPIX) * 256);
    dump("dump_cat_out.f16", cat_out, size_t(NPIX) * 64);
    dump("dump_pix.f16", pix_out, size_t(NPIX) * 48);
  }

  // temps total : graphe CUDA, 200 chauffes + 200 mesures, médiane
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (auto& x : st) x.run(s);
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
  std::printf("TOTAL graphe CUDA (SPAN entier, 1x3x%dx%d) : médiane %.3f ms, min %.3f ms\n", H, W, t[100],
              t[0]);

  // répartition par noyau : 200 passes avec événements autour de chaque noyau, médiane par noyau
  size_t ns = st.size();
  std::vector<std::vector<float>> times(ns, std::vector<float>(200));
  std::vector<cudaEvent_t> e2(ns + 1);
  for (auto& e : e2) CK(cudaEventCreate(&e));
  for (int it = 0; it < 200; ++it) {
    CK(cudaEventRecord(e2[0], s));
    for (size_t i = 0; i < ns; ++i) {
      st[i].run(s);
      CK(cudaEventRecord(e2[i + 1], s));
    }
    CK(cudaStreamSynchronize(s));
    for (size_t i = 0; i < ns; ++i) CK(cudaEventElapsedTime(&times[i][it], e2[i], e2[i + 1]));
  }
  // agrégation par type de noyau : somme des médianes, nombre de lancements, médiane moyenne
  std::vector<std::string> names;
  std::vector<double> sum;
  std::vector<int> cnt;
  for (size_t i = 0; i < ns; ++i) {
    std::sort(times[i].begin(), times[i].end());
    double m = times[i][100];
    size_t j = 0;
    while (j < names.size() && names[j] != st[i].name) ++j;
    if (j == names.size()) {
      names.push_back(st[i].name);
      sum.push_back(0);
      cnt.push_back(0);
    }
    sum[j] += m;
    cnt[j]++;
  }
  double tot = 0;
  for (double x : sum) tot += x;
  std::printf("%-16s %6s %10s %10s %7s\n", "noyau", "nb", "ms/lanc.", "ms total", "%");
  for (size_t j = 0; j < names.size(); ++j)
    std::printf("%-16s %6d %10.4f %10.4f %6.1f%%\n", names[j].c_str(), cnt[j], sum[j] / cnt[j], sum[j],
                100 * sum[j] / tot);
  std::printf("somme des noyaux (hors graphe) : %.3f ms\n", tot);
  return 0;
}
