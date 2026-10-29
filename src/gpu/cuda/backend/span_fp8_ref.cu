// Le SPAN entier en FP16 : voir span_fp8_ref.h. Le graphe est celui de span_fp8_net.cu, sans quantification :
// les convolutions sont les mêmes lancements CUTLASS (entrée FP16, accumulation FP32), la fin de bloc est un
// noyau simple (convolution 1×1 d'attention, tanh, saut, produit), et chaque tenseur à calibrer a son noyau
// de maximum.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#include "gpu/cuda/backend/span_fp8_kernels.cuh"
#include "gpu/cuda/backend/span_fp8_layout.cuh"
#include "gpu/cuda/backend/span_fp8_ref.h"

namespace ajp::gpu::cuda {
namespace {

namespace sk = span_fp8_kernels;
using F16 = cutlass::half_t;

using C16A = sk::Cfg<F16, 128, 32, 32, 4, float, cutlass::conv::IteratorAlgorithm::kAnalytic>;
using C16 = sk::Cfg<F16, 128, 32, 32, 4, float>;
using LC1 = sk::Launcher<C16A, 0, false>;
using LRelu = sk::Launcher<C16, 0, true>;
using LPlain = sk::Launcher<C16, 0, false>;

constexpr int kBlocks = SpanFp8Weights::kBlocks;
constexpr int kChannels = 64;
constexpr int kPadded = 16;
constexpr int kConcat = 4 * kChannels;
constexpr int kPixel = 48;

// Maximum de |valeur| d'un tenseur NHWC FP16 : les flottants positifs se comparent comme des entiers, et un
// NaN ou un infini donne un entier plus grand que tout fini (refusé ensuite par l'appelant).
__global__ void absmax_kernel(const __half* data, std::int64_t count, int* peak) {
  float best = 0.F;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const float v = fabsf(__half2float(data[i]));
    best = isnan(best) || isnan(v) ? __int_as_float(0x7FC00000) : fmaxf(best, v);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_xor_sync(0xFFFFFFFFU, best, offset);
    best = isnan(other) || isnan(best) ? __int_as_float(0x7FC00000) : fmaxf(best, other);
  }
  if ((threadIdx.x & 31) == 0) atomicMax(peak, __float_as_int(best));
}

// Les 64 canaux d'un tenseur NHWC copiés dans une tranche d'un tampon de 256 canaux.
__global__ void copy_slice_kernel(const __half* src, __half* dst, std::int64_t npix, int offset) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= npix * 8) return;
  const std::int64_t pixel = idx / 8;
  const int part = static_cast<int>(idx % 8);
  *reinterpret_cast<uint4*>(dst + pixel * kConcat + offset + part * 8) =
      *reinterpret_cast<const uint4*>(src + pixel * kChannels + part * 8);
}

// Fin d'un bloc SPAB, en place dans t (la sortie de c3) : t <- (t + x) · tanh(W1 · t + b1).
__global__ void __launch_bounds__(128)
    block_end_kernel(__half* t, const __half* x, const __half* w1, const float* b1, std::int64_t npix) {
  __shared__ float ws[kChannels * kChannels];
  for (int i = static_cast<int>(threadIdx.x); i < kChannels * kChannels; i += 128)
    ws[i] = __half2float(w1[i]);
  __syncthreads();
  const std::int64_t pixel = static_cast<std::int64_t>(blockIdx.x) * 128 + threadIdx.x;
  if (pixel >= npix) return;
  __half* tp = t + pixel * kChannels;
  const __half* xp = x + pixel * kChannels;
  float tv[kChannels];
#pragma unroll
  for (int i = 0; i < kChannels; ++i) tv[i] = __half2float(tp[i]);
  for (int o = 0; o < kChannels; ++o) {
    float acc = b1[o];
#pragma unroll
    for (int i = 0; i < kChannels; ++i) acc = fmaf(ws[o * kChannels + i], tv[i], acc);
    const float v = (__half2float(tp[o]) + __half2float(xp[o])) * tanhf(acc);
    tp[o] = __float2half(v);
  }
}

constexpr std::int64_t kAlign = 256;
constexpr std::int64_t align_up(std::int64_t bytes) {
  return (bytes + kAlign - 1) / kAlign * kAlign;
}

}  // namespace

struct SpanFp8Reference::Impl {
  int height = 0, width = 0;
  int h = 0, w = 0;
  std::int64_t npix = 0;

  Device c1_w, c1_b, cat_w, cat_b, pix_w, pix_b;
  struct BlockWeights {
    std::array<Device, 3> w, b;
    Device att_w, att_b;
  };
  std::array<BlockWeights, kBlocks> blocks;
  Device workspace;
  Device peak_device;

  LC1 c1;
  std::array<LRelu, kBlocks> la, lb;
  std::array<LPlain, kBlocks> lc;
  LPlain cat, pix;
  cudaError_t attribute = cudaSuccess;

  // Adresses de la zone de travail.
  __half* cat_in = nullptr;
  std::array<__half*, 2> ab{};
  __half* r1 = nullptr;
  __half* r2 = nullptr;
  __half* unshuffled = nullptr;
  __half* pix_out = nullptr;

  void bind() {
    auto* bytes = static_cast<std::uint8_t*>(workspace.get());
    const std::int64_t plane = align_up(npix * kChannels * 2);
    cat_in = reinterpret_cast<__half*>(bytes);
    std::uint8_t* region = bytes + align_up(npix * kConcat * 2);
    ab[0] = reinterpret_cast<__half*>(region);
    ab[1] = reinterpret_cast<__half*>(region + plane);
    r1 = reinterpret_cast<__half*>(region + 2 * plane);
    r2 = reinterpret_cast<__half*>(region + 3 * plane);
    unshuffled = r1;
    __half* cat_out = r1;
    pix_out = ab[0];

    {
      sk::KP k;
      k.bias = c1_b.as<float>();
      k.out16 = ab[0];
      k.ld16 = kChannels;
      c1.init(h, w, 3, 1, kPadded, kChannels, reinterpret_cast<const F16*>(unshuffled), c1_w.as<F16>(), k);
    }
    for (std::size_t b = 0; b < static_cast<std::size_t>(kBlocks); ++b) {
      BlockWeights& bw = blocks[b];
      __half* x = ab[b % 2];
      __half* t = ab[(b + 1) % 2];
      sk::KP k1;
      k1.bias = bw.b[0].as<float>();
      k1.out16 = r1;
      k1.ld16 = kChannels;
      la[b].init(h, w, 3, 1, kChannels, kChannels, reinterpret_cast<const F16*>(x), bw.w[0].as<F16>(), k1);
      sk::KP k2;
      k2.bias = bw.b[1].as<float>();
      k2.out16 = r2;
      k2.ld16 = kChannels;
      lb[b].init(h, w, 3, 1, kChannels, kChannels, reinterpret_cast<const F16*>(r1), bw.w[1].as<F16>(), k2);
      sk::KP k3;
      k3.bias = bw.b[2].as<float>();
      k3.out16 = t;
      k3.ld16 = kChannels;
      lc[b].init(h, w, 3, 1, kChannels, kChannels, reinterpret_cast<const F16*>(r2), bw.w[2].as<F16>(), k3);
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
};

SpanFp8Reference::SpanFp8Reference(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SpanFp8Reference::~SpanFp8Reference() = default;

std::unique_ptr<SpanFp8Reference> SpanFp8Reference::create(const SpanFp8Weights& weights, int height,
                                                           int width, SpanFp8Status& status,
                                                           std::string& error) {
  auto impl = std::make_unique<Impl>();
  impl->height = height;
  impl->width = width;
  impl->h = height / 2;
  impl->w = width / 2;
  impl->npix = static_cast<std::int64_t>(impl->h) * impl->w;

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
    for (std::size_t i = 0; i < 3; ++i)
      if (!put_half(bw.w[i], weights.conv[b][i], kBlockWeights) ||
          !put_bias(bw.b[i], weights.conv[b][i], kChannels))
        return nullptr;
    if (!put_half(bw.att_w, weights.attention[b], kAttentionWeights) ||
        !put_bias(bw.att_b, weights.attention[b], kChannels))
      return nullptr;
  }
  // La concaténation de 256 canaux FP16, quatre tenseurs de 64 canaux ; dégroupage, sortie de conv_cat et de
  // to_pixel recouvrent des tampons alors inactifs.
  const std::int64_t bytes = align_up(impl->npix * kConcat * 2) + 4 * align_up(impl->npix * kChannels * 2);
  if (!sent(impl->workspace.reserve(static_cast<std::size_t>(bytes)), "zone de travail") ||
      !sent(impl->peak_device.reserve(sizeof(int) * kPeaks), "maxima") ||
      !sent(cudaMemset(impl->peak_device.get(), 0, sizeof(int) * kPeaks), "maxima"))
    return nullptr;
  impl->bind();
  status = SpanFp8Status::Ok;
  return std::unique_ptr<SpanFp8Reference>(new SpanFp8Reference(std::move(impl)));
}

bool SpanFp8Reference::run(cudaStream_t stream, const void* input, void* output, bool track,
                           std::string& error) {
  Impl& n = *impl_;
  if (n.attribute != cudaSuccess) {
    error = std::string("mémoire partagée des noyaux refusée : ") + cudaGetErrorString(n.attribute);
    return false;
  }
  // Une erreur non collante laissée par un appel d'avant (une sonde de TensorRT, un attribut refusé par ce
  // pilote) ne doit pas passer pour un échec de ces lancements : on la vide avant le premier.
  static_cast<void>(cudaGetLastError());
  int* peaks = static_cast<int*>(n.peak_device.get());
  const std::int64_t count = n.npix * kChannels;
  const auto observe = [&](int index, const __half* tensor) {
    if (!track) return;
    absmax_kernel<<<static_cast<unsigned>(std::min<std::int64_t>((count + 255) / 256, 2048)), 256, 0,
                    stream>>>(tensor, count, peaks + index);
  };
  const unsigned slice_blocks = static_cast<unsigned>((n.npix * 8 + 255) / 256);
  const auto to_slice = [&](const __half* tensor, int slice) {
    copy_slice_kernel<<<slice_blocks, 256, 0, stream>>>(tensor, n.cat_in, n.npix, slice * kChannels);
  };
  unshuffle_kernel<<<static_cast<unsigned>((n.npix + 255) / 256), 256, 0, stream>>>(
      static_cast<const __half*>(input), n.unshuffled, n.height, n.width);
  n.c1.launch(stream);
  observe(0, n.ab[0]);
  to_slice(n.ab[0], 0);
  for (std::size_t b = 0; b < static_cast<std::size_t>(kBlocks); ++b) {
    const int base = 1 + 3 * static_cast<int>(b);
    __half* t = n.ab[(b + 1) % 2];
    const __half* x = n.ab[b % 2];
    n.la[b].launch(stream);
    observe(base, n.r1);
    if (b == static_cast<std::size_t>(kBlocks - 1)) to_slice(n.r1, 3);
    n.lb[b].launch(stream);
    observe(base + 1, n.r2);
    n.lc[b].launch(stream);
    block_end_kernel<<<static_cast<unsigned>((n.npix + 127) / 128), 128, 0, stream>>>(
        t, x, n.blocks[b].att_w.as<__half>(), n.blocks[b].att_b.as<float>(), n.npix);
    observe(base + 2, t);
    if (b == 0) to_slice(t, 2);
    if (b == static_cast<std::size_t>(kBlocks - 1)) to_slice(t, 1);
  }
  n.cat.launch(stream);
  n.pix.launch(stream);
  depth_to_space_kernel<<<static_cast<unsigned>((n.npix * 12 + 255) / 256), 256, 0, stream>>>(
      n.pix_out, static_cast<__half*>(output), n.h, n.w);
  cudaError_t result = cudaGetLastError();
  if (result == cudaSuccess) result = cudaStreamSynchronize(stream);
  if (result != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    error = std::string("référence FP16 : ") + cudaGetErrorString(result);
    return false;
  }
  return true;
}

bool SpanFp8Reference::peaks(float (&out)[kPeaks], std::string& error) {
  int raw[kPeaks];
  const cudaError_t result = cudaMemcpy(raw, impl_->peak_device.get(), sizeof(raw), cudaMemcpyDeviceToHost);
  if (result != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    error = std::string("maxima : ") + cudaGetErrorString(result);
    return false;
  }
  for (int i = 0; i < kPeaks; ++i) std::memcpy(&out[i], &raw[i], sizeof(float));
  return true;
}

}  // namespace ajp::gpu::cuda
