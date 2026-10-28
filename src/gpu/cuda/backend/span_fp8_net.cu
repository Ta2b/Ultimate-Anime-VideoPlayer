// Le SPAN entier en FP8 : voir span_fp8_net.h et docs/span-fp8-cutlass.md. Le graphe des noyaux est celui du
// banc tools/bench/fp8/cutlass/span_net.cu, repris tel que mesuré (9,3 ms contre 14,3 ms pour TensorRT FP16
// en 1088×1920 sur RTX 4070 Ti) : seuls changent la source des poids (le paquet), la taille (à l'exécution)
// et la zone de travail (fournie par l'appelant, avec recouvrement des tampons qui ne vivent pas en même
// temps).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/span_fp8_kernels.cuh"
#include "gpu/cuda/backend/span_fp8_net.h"

namespace ajp::gpu::cuda {
namespace {

namespace sk = span_fp8_kernels;
using E4 = cutlass::float_e4m3_t;
using F16 = cutlass::half_t;

// Blocs : entrée FP8, 64 canaux. Hors blocs : entrée FP16, accumulateur f32.
using C8 = sk::Cfg<E4, 128, 64, 32, 3>;
using L8 = sk::Launcher<C8, 0, true>;
using LT = sk::Launcher<C8, 1, false>;
using C16A = sk::Cfg<F16, 128, 32, 32, 4, float, cutlass::conv::IteratorAlgorithm::kAnalytic>;
using C16 = sk::Cfg<F16, 128, 32, 32, 4, float>;
using LC1 = sk::Launcher<C16A, 0, false>;
using LCAT = sk::Launcher<C16, 0, false>;

constexpr int kBlocks = SpanFp8Weights::kBlocks;
constexpr int kChannels = 64;
constexpr int kPadded = 16;  // conv_1 lit 16 canaux (12 + 4 nuls)
constexpr int kConcat = 4 * kChannels;
constexpr int kPixel = 48;
constexpr int kScaleCount = SpanFp8Weights::kScales;

// Mémoire d'appareil possédée.
class Device {
 public:
  Device() = default;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
  Device& operator=(Device&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = std::exchange(other.ptr_, nullptr);
    }
    return *this;
  }
  ~Device() { reset(); }
  void reset() {
    if (ptr_ != nullptr) cudaFree(ptr_);
    ptr_ = nullptr;
  }
  cudaError_t upload(const void* host, std::size_t bytes) {
    reset();
    cudaError_t error = cudaMalloc(&ptr_, bytes);
    if (error != cudaSuccess) {
      ptr_ = nullptr;
      return error;
    }
    return cudaMemcpy(ptr_, host, bytes, cudaMemcpyHostToDevice);
  }
  template <class T>
  const T* as() const {
    return static_cast<const T*>(ptr_);
  }

 private:
  void* ptr_ = nullptr;
};

// Poids FP8 d'un bloc : échelle par tenseur « maximum / 448 », arrondi au plus proche, saturé.
float quantize_fp8(const std::uint16_t* half_bits, std::size_t count, std::vector<std::uint8_t>& out) {
  std::vector<float> values(count);
  float peak = 0.F;
  for (std::size_t i = 0; i < count; ++i) {
    __half h;
    std::memcpy(&h, &half_bits[i], sizeof(h));
    values[i] = __half2float(h);
    peak = std::max(peak, std::abs(values[i]));
  }
  const float scale = peak > 0.F ? peak / 448.F : 1.F;
  out.resize(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    out[i] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(values[i] / scale, __NV_SATFINITE, __NV_E4M3));
  return scale;
}

// Dégroupage : NCHW FP16 3×H×W -> NHWC 16 canaux à demi-résolution (canal = c·4 + a·2 + b ; 12 à 15 nuls).
__global__ void unshuffle_kernel(const __half* in, __half* out, int height, int width) {
  const int w = width / 2;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= (height / 2) * w) return;
  const int y = idx / w, x = idx % w;
  __half v[16];
#pragma unroll
  for (int c = 0; c < 3; ++c)
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int b = 0; b < 2; ++b)
        v[c * 4 + a * 2 + b] =
            in[(static_cast<size_t>(c) * static_cast<size_t>(height) + static_cast<size_t>(2 * y + a)) *
                   static_cast<size_t>(width) +
               static_cast<size_t>(2 * x + b)];
#pragma unroll
  for (int i = 12; i < 16; ++i) v[i] = __float2half(0.F);
  uint4* o = reinterpret_cast<uint4*>(out + static_cast<size_t>(idx) * 16);
  o[0] = *reinterpret_cast<uint4*>(&v[0]);
  o[1] = *reinterpret_cast<uint4*>(&v[8]);
}

// DepthToSpace(4, mode CRD du modèle) : NHWC 48 canaux -> NCHW 3×4h×4w ; canal d'entrée = c·16 + bh·4 + bw.
__global__ void depth_to_space_kernel(const __half* in, __half* out, int h, int w) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);  // (pixel, bh, c)
  if (idx >= h * w * 12) return;
  const int c = idx % 3, bh = (idx / 3) % 4, pix = idx / 12;
  const int y = pix / w, x = pix % w;
  const __half* p = in + static_cast<size_t>(pix) * 48 + static_cast<size_t>(c * 16 + bh * 4);
  __half o[4];
#pragma unroll
  for (int bw = 0; bw < 4; ++bw) o[bw] = p[bw];
  *reinterpret_cast<uint2*>(
      out +
      (static_cast<size_t>(c) * static_cast<size_t>(4 * h) + static_cast<size_t>(4 * y + bh)) *
          static_cast<size_t>(4 * w) +
      static_cast<size_t>(4 * x)) = *reinterpret_cast<uint2*>(o);
}

constexpr std::int64_t kAlign = 256;
constexpr std::int64_t align_up(std::int64_t bytes) {
  return (bytes + kAlign - 1) / kAlign * kAlign;
}

}  // namespace

struct SpanFp8Net::Impl {
  int height = 0, width = 0;  // image d'entrée
  int h = 0, w = 0;           // caractéristiques (demi-résolution)
  std::int64_t npix = 0;

  // Poids sur l'appareil.
  Device c1_w, c1_b, cat_w, cat_b, pix_w, pix_b;
  struct BlockWeights {
    std::array<Device, 3> w, b;
    Device att_w, att_b;
    std::array<float, 3> w_scale{};
  };
  std::array<BlockWeights, kBlocks> blocks;
  std::array<float, kScaleCount> scales{};

  // Lancements : leurs paramètres portent les adresses de la zone de travail (bind).
  LC1 c1;
  std::array<L8, kBlocks> la, lb;
  std::array<LT, kBlocks> lc;
  LCAT cat, pix;
  void* workspace = nullptr;
  cudaError_t attribute = cudaSuccess;

  std::int64_t workspace_bytes() const {
    // concaténation FP16 de 256 canaux + quatre tampons FP8 de 64 canaux ; dégroupage, sortie de conv_cat et
    // de to_pixel recouvrent des tampons FP8 alors inactifs.
    return 2 * kAlign + align_up(npix * 512) + align_up(npix * 256);
  }

  void bind(void* base) {
    workspace = base;
    auto* bytes = static_cast<std::uint8_t*>(base);
    const std::int64_t cat_bytes = align_up(npix * 512);
    __half* cat_in = reinterpret_cast<__half*>(bytes);
    std::uint8_t* region = bytes + cat_bytes;
    std::uint8_t* x[2] = {region, region + npix * 64};
    std::uint8_t* r1 = region + npix * 128;
    std::uint8_t* r2 = region + npix * 192;
    __half* unshuffled = reinterpret_cast<__half*>(r1);
    __half* cat_out = reinterpret_cast<__half*>(r1);
    __half* pix_out = reinterpret_cast<__half*>(x[0]);
    unshuffled_ptr = unshuffled;
    pix_out_ptr = pix_out;

    {  // conv_1 : sorties FP8 (entrée du bloc 1) et FP16 (tranche 0 de la concaténation)
      sk::KP k;
      k.bias = c1_b.as<float>();
      k.out8 = x[0];
      k.inv_s8 = 1.F / scales[0];
      k.out16 = cat_in;
      k.ld16 = kConcat;
      c1.init(h, w, 3, 1, kPadded, kChannels, reinterpret_cast<const F16*>(unshuffled), c1_w.as<F16>(), k);
    }
    for (int b = 0; b < kBlocks; ++b) {
      const auto bi = static_cast<std::size_t>(b);
      const float s_in = b == 0 ? scales[0] : scales[static_cast<std::size_t>(1 + 3 * (b - 1) + 2)];
      const float s_r1 = scales[static_cast<std::size_t>(1 + 3 * b)];
      const float s_r2 = scales[static_cast<std::size_t>(2 + 3 * b)];
      const float s_out = scales[static_cast<std::size_t>(3 + 3 * b)];
      const std::uint8_t* xin = x[b % 2];
      std::uint8_t* xout = x[(b + 1) % 2];
      BlockWeights& bw = blocks[bi];
      {  // c1 + Relu -> FP8 ; pour le bloc 8, aussi FP16 dans la tranche 3 (relu1 du dernier bloc)
        sk::KP k;
        k.alpha = s_in * bw.w_scale[0];
        k.bias = bw.b[0].as<float>();
        k.out8 = r1;
        k.inv_s8 = 1.F / s_r1;
        if (b == kBlocks - 1) {
          k.out16 = cat_in + 3 * kChannels;
          k.ld16 = kConcat;
        }
        la[bi].init(h, w, 3, 1, kChannels, kChannels, reinterpret_cast<const E4*>(xin), bw.w[0].as<E4>(), k);
      }
      {
        sk::KP k;
        k.alpha = s_r1 * bw.w_scale[1];
        k.bias = bw.b[1].as<float>();
        k.out8 = r2;
        k.inv_s8 = 1.F / s_r2;
        lb[bi].init(h, w, 3, 1, kChannels, kChannels, reinterpret_cast<const E4*>(r1), bw.w[1].as<E4>(), k);
      }
      {  // fin de bloc : c3, attention 1x1, tanh, saut, produit
        sk::KP k;
        k.skip = xin;
        k.skip_scale = s_in;
        k.alpha = s_r2 * bw.w_scale[2];
        k.bias = bw.b[2].as<float>();
        k.w1 = bw.att_w.as<__half>();
        k.b1 = bw.att_b.as<float>();
        if (b < kBlocks - 1) {
          k.out8 = xout;
          k.inv_s8 = 1.F / s_out;
        }
        if (b == 0) {  // sortie du bloc 1 : tranche 2
          k.out16 = cat_in + 2 * kChannels;
          k.ld16 = kConcat;
        }
        if (b == kBlocks - 1) {  // sortie du bloc 8 : tranche 1
          k.out16 = cat_in + kChannels;
          k.ld16 = kConcat;
        }
        lc[bi].init(h, w, 3, 1, kChannels, kChannels, reinterpret_cast<const E4*>(r2), bw.w[2].as<E4>(), k);
      }
    }
    {
      sk::KP k;
      k.bias = cat_b.as<float>();
      k.out16 = cat_out;
      k.ld16 = kChannels;
      cat.init(h, w, 1, 0, kConcat, kChannels, reinterpret_cast<const F16*>(cat_in), cat_w.as<F16>(), k);
      sk::KP p;
      p.bias = pix_b.as<float>();
      p.out16 = pix_out;
      p.ld16 = kPixel;
      pix.init(h, w, 3, 1, kChannels, kPixel, reinterpret_cast<const F16*>(cat_out), pix_w.as<F16>(), p);
    }
    attribute = c1.attribute;
    for (std::size_t b = 0; b < static_cast<std::size_t>(kBlocks); ++b)
      for (const cudaError_t e : {la[b].attribute, lb[b].attribute, lc[b].attribute})
        if (attribute == cudaSuccess) attribute = e;
    for (const cudaError_t e : {cat.attribute, pix.attribute})
      if (attribute == cudaSuccess) attribute = e;
  }

  __half* unshuffled_ptr = nullptr;  // recouvre r1 (posés par bind)
  __half* pix_out_ptr = nullptr;     // recouvre x[0]
};

SpanFp8Net::SpanFp8Net(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SpanFp8Net::~SpanFp8Net() = default;

bool SpanFp8Net::supported_device(int major, int minor) {
  return major == 8 && minor == 9;
}

std::int64_t SpanFp8Net::workspace_bytes() const {
  return impl_->workspace_bytes();
}

void SpanFp8Net::bind(void* workspace) {
  impl_->bind(workspace);
}

std::unique_ptr<SpanFp8Net> SpanFp8Net::create(const SpanFp8Weights& weights, int height, int width,
                                               SpanFp8Status& status, std::string& error) {
  auto impl = std::make_unique<Impl>();
  impl->height = height;
  impl->width = width;
  impl->h = height / 2;
  impl->w = width / 2;
  impl->npix = static_cast<std::int64_t>(impl->h) * impl->w;
  for (std::size_t i = 0; i < static_cast<std::size_t>(kScaleCount); ++i) impl->scales[i] = weights.scales[i];

  // false : l'envoi a échoué, status et error le disent.
  const auto sent = [&](cudaError_t result, const char* what) {
    if (result == cudaSuccess) return true;
    static_cast<void>(cudaGetLastError());
    status = result == cudaErrorMemoryAllocation ? SpanFp8Status::OutOfMemory : SpanFp8Status::Failed;
    error = std::string(what) + " : " + cudaGetErrorString(result);
    return false;
  };
  const auto put_half = [&](Device& to, const SpanFp8ConvView& conv, std::size_t count) {
    return sent(to.upload(conv.weights, count * sizeof(std::uint16_t)), "poids");
  };
  const auto put_bias = [&](Device& to, const SpanFp8ConvView& conv, std::size_t count) {
    return sent(to.upload(conv.bias, count * sizeof(float)), "biais");
  };
  constexpr std::size_t kBlockWeights = std::size_t{kChannels} * 9 * kChannels;
  constexpr std::size_t kAttentionWeights = std::size_t{kChannels} * kChannels;
  if (!put_half(impl->c1_w, weights.first, std::size_t{kChannels} * 9 * kPadded) ||
      !put_bias(impl->c1_b, weights.first, kChannels) ||
      !put_half(impl->cat_w, weights.concat, std::size_t{kChannels} * kConcat) ||
      !put_bias(impl->cat_b, weights.concat, kChannels) ||
      !put_half(impl->pix_w, weights.pixel, std::size_t{kPixel} * 9 * kChannels) ||
      !put_bias(impl->pix_b, weights.pixel, kPixel))
    return nullptr;
  for (std::size_t b = 0; b < static_cast<std::size_t>(kBlocks); ++b) {
    Impl::BlockWeights& bw = impl->blocks[b];
    for (std::size_t i = 0; i < 3; ++i) {
      std::vector<std::uint8_t> q;
      bw.w_scale[i] = quantize_fp8(weights.conv[b][i].weights, kBlockWeights, q);
      if (!sent(bw.w[i].upload(q.data(), q.size()), "poids FP8") ||
          !put_bias(bw.b[i], weights.conv[b][i], kChannels))
        return nullptr;
    }
    if (!put_half(bw.att_w, weights.attention[b], kAttentionWeights) ||
        !put_bias(bw.att_b, weights.attention[b], kChannels))
      return nullptr;
  }
  status = SpanFp8Status::Ok;
  return std::unique_ptr<SpanFp8Net>(new SpanFp8Net(std::move(impl)));
}

bool SpanFp8Net::launch(cudaStream_t stream, const void* input, void* output, std::string& error) {
  Impl& n = *impl_;
  if (n.workspace == nullptr) {
    error = "zone de travail absente";
    return false;
  }
  if (n.attribute != cudaSuccess) {
    error = std::string("mémoire partagée des noyaux refusée : ") + cudaGetErrorString(n.attribute);
    return false;
  }
  const int blocks = static_cast<int>((n.npix + 255) / 256);
  unshuffle_kernel<<<blocks, 256, 0, stream>>>(static_cast<const __half*>(input), n.unshuffled_ptr, n.height,
                                               n.width);
  n.c1.launch(stream);
  for (std::size_t b = 0; b < static_cast<std::size_t>(kBlocks); ++b) {
    n.la[b].launch(stream);
    n.lb[b].launch(stream);
    n.lc[b].launch(stream);
  }
  n.cat.launch(stream);
  n.pix.launch(stream);
  depth_to_space_kernel<<<static_cast<int>((n.npix * 12 + 255) / 256), 256, 0, stream>>>(
      n.pix_out_ptr, static_cast<__half*>(output), n.h, n.w);
  const cudaError_t result = cudaGetLastError();
  if (result != cudaSuccess) {
    error = std::string("lancement des noyaux : ") + cudaGetErrorString(result);
    return false;
  }
  return true;
}

}  // namespace ajp::gpu::cuda
