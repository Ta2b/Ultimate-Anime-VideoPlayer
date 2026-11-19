// La tête de confiance de TemporalFix natif fondue (tfix_head.h). Couche par couche, ses deux activations du
// milieu (32 canaux à la moitié du réseau, six appuis) s'écrivaient puis se relisaient : 200 des 390 Mo de la
// fenêtre en demi-taille (/mnt/project-files/chantier-natif/tfix-plafonds.md). Ici, un bloc calcule une tuile
// de 16 × 8 points de sortie : l'entrée, halo de trois points compris, est copiée une fois en mémoire
// partagée ; la première convolution en tire 20 × 12 points, la deuxième 18 × 10 (dans le tampon de
// l'entrée, lue jusqu'au bout avant), la dernière les 16 × 8. Hors de l'image, les activations du milieu
// valent zéro, comme le bord de la convolution suivante.
//
// Les deux premières passent par les tensor cores (mma m16n8k16, sommes f16, mixtes ou f32 comme
// rife_native_kernels.cuh) : un fragment m16 est seize points consécutifs de la sortie, lignes de la tuile
// mises bout à bout ; ldmatrix prend une adresse par ligne, la prise décalant chacune. Les poids sont rangés
// dans l'ordre des fragments B : un uint2 par lane, par prise, demi-K et tuile de huit sorties. La dernière
// (un canal) se fait sur les cœurs CUDA, un fil par point, sommes f32, comme single_kernel.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/rife_native_kernels.cuh"
#include "gpu/cuda/backend/tfix_head.h"

namespace kirari::gpu::cuda {
namespace {

namespace rn = rife_native;

constexpr int kC = 32;           // canaux de l'entrée (25 complétés) et du milieu
constexpr int kStride = kC + 8;  // pas d'un point en mémoire partagée : les lignes de ldmatrix sur des
                                 // banques distinctes
constexpr int kTW = 16;
constexpr int kTH = 8;
// Huit warps : deux blocs de 45 Ko par SM sur Ada, seize warps pour couvrir l'attente de ldmatrix et des
// poids (quatre n'en laissaient que huit, et la tête faisait 36 % du temps GPU de la lecture).
constexpr int kWarps = 8;
constexpr int kThreads = 32 * kWarps;
constexpr int kLastLanes = kThreads / (kTW * kTH);  // fils par point de la dernière convolution
static_assert(kLastLanes == 2, "deux fils par point de sortie pour la dernière convolution");
constexpr int kInW = kTW + 6;
constexpr int kInH = kTH + 6;
constexpr int kM1W = kTW + 4;
constexpr int kM1H = kTH + 4;
constexpr int kM2W = kTW + 2;
constexpr int kM2H = kTH + 2;
constexpr int kNt = kC / 8;                   // tuiles n8 d'une couche
constexpr int kFragWords = 9 * 2 * kNt * 32;  // uint2 rangés d'une couche

struct HeadParams {
  const __half* in = nullptr;
  __half* out = nullptr;
  const uint2* w1 = nullptr;  // [prise][demi-K][tuile n8][lane]
  const uint2* w2 = nullptr;
  const float* bias1 = nullptr;
  const float* bias2 = nullptr;
  const float* slope1 = nullptr;
  const float* slope2 = nullptr;
  const float* w3 = nullptr;  // [prise][canal]
  float bias3 = 0.F;
  int h = 0, w = 0;
};

__device__ __forceinline__ __half2 activated(float2 v, float2 bias, float2 slope) {
  // Comme rife_native_kernels.cuh : la somme et le biais arrondis en f16, puis l'activation.
  const float2 r = __half22float2(__floats2half2_rn(v.x + bias.x, v.y + bias.y));
  return __floats2half2_rn(r.x < 0.F ? r.x * slope.x : r.x, r.y < 0.F ? r.y * slope.y : r.y);
}

// Une convolution 3 × 3 de 32 vers 32 canaux, activation comprise, de src (IW points de large) vers dst
// (OW × OH) ; (gx0, gy0) : le point de l'image du premier point de dst. Chaque warp prend deux fragments m16
// à la fois.
template <rn::Acc A, int IW, int OW, int OH>
__device__ __forceinline__ void conv(const __half* src, __half* dst, const uint2* w, const float* bias,
                                     const float* slope, int gx0, int gy0, int h, int width) {
  constexpr int kM = OW * OH;
  constexpr int kF = (kM + 15) / 16;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int g = lane >> 2;
  const int q = lane & 3;
  const std::uint32_t base = rn::smem_u32(src);
  for (int f0 = 2 * warp; f0 < kF; f0 += 2 * kWarps) {
    std::uint32_t row[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const int pix = min((f0 + m) * 16 + (lane & 15), kM - 1);
      row[m] =
          base + static_cast<std::uint32_t>((((pix / OW) * IW + pix % OW) * kStride + (lane >> 4) * 8) * 2);
    }
    rn::Frag<A> acc[2][kNt];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int nt = 0; nt < kNt; ++nt) acc[m][nt].clear();
#pragma unroll 1
    for (int tap = 0; tap < 9; ++tap) {
      const auto shift = static_cast<std::uint32_t>(((tap / 3) * IW + tap % 3) * kStride * 2);
#pragma unroll
      for (int kc = 0; kc < 2; ++kc) {
        uint2 b[kNt];
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) b[nt] = __ldg(w + ((tap * 2 + kc) * kNt + nt) * 32 + lane);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          std::uint32_t a[4];
          rn::ldmatrix4(a, row[m] + shift + static_cast<std::uint32_t>(kc * 32));
#pragma unroll
          for (int nt = 0; nt < kNt; ++nt) acc[m][nt].mma(a, b[nt].x, b[nt].y);
        }
      }
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) acc[m][nt].promote();
    }
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int pix = (f0 + m) * 16 + g + 8 * hh;
        if (pix >= kM) continue;
        const int gx = gx0 + pix % OW;
        const int gy = gy0 + pix / OW;
        const bool inside = gx >= 0 && gx < width && gy >= 0 && gy < h;
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
          const int n = nt * 8 + 2 * q;
          const __half2 v =
              inside ? activated(acc[m][nt].pair(hh), __ldg(reinterpret_cast<const float2*>(bias + n)),
                                 __ldg(reinterpret_cast<const float2*>(slope + n)))
                     : __float2half2_rn(0.F);
          *reinterpret_cast<__half2*>(dst + pix * kStride + n) = v;
        }
      }
  }
}

template <rn::Acc A>
__global__ void __launch_bounds__(kThreads, 2) head_kernel(HeadParams p) {
  __shared__ __align__(16) __half big[kInW * kInH * kStride];  // l'entrée, puis la deuxième activation
  __shared__ __align__(16) __half mid[kM1W * kM1H * kStride];
  __shared__ float last[9 * kC];
  const int tid = static_cast<int>(threadIdx.x);
  const int tx0 = static_cast<int>(blockIdx.x) * kTW;
  const int ty0 = static_cast<int>(blockIdx.y) * kTH;
  const long plane = static_cast<long>(p.h) * p.w;
  const __half* in = p.in + static_cast<long>(blockIdx.z) * plane * kC;
  for (int i = tid; i < 9 * kC; i += kThreads) last[i] = p.w3[i];
  for (int i = tid; i < kInW * kInH * 4; i += kThreads) {
    const int point = i >> 2;
    const int part = i & 3;
    const int gx = tx0 - 3 + point % kInW;
    const int gy = ty0 - 3 + point / kInW;
    const bool ok = gx >= 0 && gx < p.w && gy >= 0 && gy < p.h;
    const __half* from = ok ? in + (static_cast<long>(gy) * p.w + gx) * kC + part * 8 : in;
    rn::cp_async16(rn::smem_u32(big + point * kStride + part * 8), from, ok);
  }
  rn::cp_async_wait_all();
  __syncthreads();
  conv<A, kInW, kM1W, kM1H>(big, mid, p.w1, p.bias1, p.slope1, tx0 - 2, ty0 - 2, p.h, p.w);
  __syncthreads();
  conv<A, kM1W, kM2W, kM2H>(mid, big, p.w2, p.bias2, p.slope2, tx0 - 1, ty0 - 1, p.h, p.w);
  __syncthreads();

  // Deux fils par point, seize canaux chacun ; les deux sommes se rejoignent par échange de chaîne.
  const int point = tid / kLastLanes;
  const int side = tid % kLastLanes;
  const int ox = point % kTW;
  const int oy = point / kTW;
  const int gx = tx0 + ox;
  const int gy = ty0 + oy;
  float sum = 0.F;
#pragma unroll 1
  for (int tap = 0; tap < 9; ++tap) {
    const __half* at = big + ((oy + tap / 3) * kM2W + ox + tap % 3) * kStride + side * 16;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
      const uint4 v = *reinterpret_cast<const uint4*>(at + part * 8);
      const auto* pairs = reinterpret_cast<const __half2*>(&v);
      const float* t = last + tap * kC + side * 16 + part * 8;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(pairs[j]);
        sum += f.x * t[2 * j] + f.y * t[2 * j + 1];
      }
    }
  }
  sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 1);
  if (side != 0 || gx >= p.w || gy >= p.h) return;
  const __half2 zero = __float2half2_rn(0.F);
  uint4 out;
  auto* pairs = reinterpret_cast<__half2*>(&out);
  pairs[0] = __halves2half2(__float2half_rn(sum + p.bias3), __float2half_rn(0.F));
  pairs[1] = zero;
  pairs[2] = zero;
  pairs[3] = zero;
  *reinterpret_cast<uint4*>(
      p.out + (static_cast<long>(blockIdx.z) * plane + static_cast<long>(gy) * p.w + gx) * 8) = out;
}

// Mémoire d'appareil possédée.
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() {
    if (ptr_ != nullptr) static_cast<void>(cudaFree(ptr_));
  }
  template <class T>
  bool upload(const std::vector<T>& host, std::string& error) {
    const std::size_t bytes = host.size() * sizeof(T);
    cudaError_t e = cudaMalloc(&ptr_, bytes);
    if (e != cudaSuccess) ptr_ = nullptr;
    if (e == cudaSuccess) e = cudaMemcpy(ptr_, host.data(), bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) error = std::string("tête fondue : envoi des poids : ") + cudaGetErrorString(e);
    return e == cudaSuccess;
  }
  template <class T>
  const T* as() const {
    return static_cast<const T*>(ptr_);
  }

 private:
  void* ptr_ = nullptr;
};

// Le poids (sortie o, prise, canal c) d'une convolution, zéro dans le rembourrage des canaux.
float weight(const RifeNativeConv& conv, int o, int tap, int c) {
  if (o >= conv.cout || c >= conv.cin) return 0.F;
  return conv.weights[(static_cast<std::size_t>(o) * 9 + tap) * static_cast<std::size_t>(conv.cin) + c];
}

std::uint32_t half_pair(float lo, float hi) {
  const __half2 v = __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));  // GCC 16 refuse le reinterpret_cast (strict-aliasing)
  return bits;
}

// Les fragments B d'une couche de 32 sorties : b0 aux canaux 2q et 2q + 1 du demi-K, b1 huit plus loin, la
// sortie lane / 4 de la tuile.
std::vector<uint2> packed(const RifeNativeConv& conv) {
  std::vector<uint2> out(kFragWords);
  for (int tap = 0; tap < 9; ++tap)
    for (int kc = 0; kc < 2; ++kc)
      for (int nt = 0; nt < kNt; ++nt)
        for (int lane = 0; lane < 32; ++lane) {
          const int n = nt * 8 + lane / 4;
          const int k = kc * 16 + (lane % 4) * 2;
          out[static_cast<std::size_t>(((tap * 2 + kc) * kNt + nt) * 32 + lane)] = {
              half_pair(weight(conv, n, tap, k), weight(conv, n, tap, k + 1)),
              half_pair(weight(conv, n, tap, k + 8), weight(conv, n, tap, k + 9))};
        }
  return out;
}

// Une convolution 3 × 3 de pas 1, sans résiduel, de cin canaux (au plus 32) vers cout, activée ou non.
bool shaped(const RifeNativeConv& conv, int cin_min, int cout, bool activation) {
  return !conv.transposed && conv.stride == 1 && conv.scale.empty() && conv.cin >= cin_min &&
         conv.cin <= kC && conv.cout == cout && conv.slope.empty() != activation &&
         (!activation || conv.slope.size() == static_cast<std::size_t>(cout)) &&
         conv.weights.size() == static_cast<std::size_t>(cout) * 9 * static_cast<std::size_t>(conv.cin) &&
         conv.bias.size() == static_cast<std::size_t>(cout);
}

class Head final : public TfixHead {
 public:
  explicit Head(RifeNativeSums sums) : sums_(sums) {}

  bool build(const RifeNativePiece& head, std::string& error) {
    const RifeNativeConv& last_conv = head[2];
    std::vector<float> taps(static_cast<std::size_t>(9) * kC);
    for (int tap = 0; tap < 9; ++tap)
      for (int c = 0; c < kC; ++c)
        taps[static_cast<std::size_t>(tap) * kC + c] = weight(last_conv, 0, tap, c);
    bias3_ = last_conv.bias[0];
    return w1_.upload(packed(head[0]), error) && w2_.upload(packed(head[1]), error) &&
           bias1_.upload(head[0].bias, error) && bias2_.upload(head[1].bias, error) &&
           slope1_.upload(head[0].slope, error) && slope2_.upload(head[1].slope, error) &&
           w3_.upload(taps, error);
  }

  cudaError_t run(cudaStream_t stream, const void* in, void* out, int h, int w, int batch) const override {
    HeadParams p;
    p.in = static_cast<const __half*>(in);
    p.out = static_cast<__half*>(out);
    p.w1 = w1_.as<uint2>();
    p.w2 = w2_.as<uint2>();
    p.bias1 = bias1_.as<float>();
    p.bias2 = bias2_.as<float>();
    p.slope1 = slope1_.as<float>();
    p.slope2 = slope2_.as<float>();
    p.w3 = w3_.as<float>();
    p.bias3 = bias3_;
    p.h = h;
    p.w = w;
    const dim3 grid(static_cast<unsigned>((w + kTW - 1) / kTW), static_cast<unsigned>((h + kTH - 1) / kTH),
                    static_cast<unsigned>(batch));
    switch (sums_) {
      case RifeNativeSums::kF16:
        head_kernel<rn::Acc::kF16><<<grid, kThreads, 0, stream>>>(p);
        break;
      case RifeNativeSums::kMixed:
        head_kernel<rn::Acc::kMixed><<<grid, kThreads, 0, stream>>>(p);
        break;
      case RifeNativeSums::kF32:
        head_kernel<rn::Acc::kF32><<<grid, kThreads, 0, stream>>>(p);
        break;
    }
    return cudaGetLastError();
  }

 private:
  RifeNativeSums sums_;
  Buffer w1_, w2_, bias1_, bias2_, slope1_, slope2_, w3_;
  float bias3_ = 0.F;
};

}  // namespace

std::unique_ptr<TfixHead> TfixHead::create(const RifeNativePiece& head, RifeNativeSums sums,
                                           std::string& error) {
  if (head.size() != 3 || !shaped(head[0], 17, kC, true) || !shaped(head[1], kC, kC, true) ||
      !shaped(head[2], kC, 1, false)) {
    error = "tête : ni 25 → 32, 32 → 32 activées, ni 32 → 1";
    return nullptr;
  }
  auto out = std::make_unique<Head>(sums);
  if (!out->build(head, error)) return nullptr;
  return out;
}

}  // namespace kirari::gpu::cuda
