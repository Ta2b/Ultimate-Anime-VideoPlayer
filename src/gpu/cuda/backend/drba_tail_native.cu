#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/drba_tail_native.h"
#include "gpu/cuda/backend/drba_tail_wino.cuh"
#include "gpu/cuda/backend/rife_native_layer.h"

namespace kirari::gpu::cuda {
namespace {

namespace rn = rife_native;
constexpr int kWidth = 120;
constexpr int kHeight = 68;
constexpr int kChannels = 96;
constexpr int kPackedInput = 112;
constexpr int kMidChannels = 48;
constexpr int kFlowAndMask = 20;
constexpr int kFixedChannels = 48;
constexpr std::size_t kMiddleBytes = 4U * kWidth * kHeight * kMidChannels * sizeof(__half);
constexpr std::size_t kStateBytes = kWidth * kHeight * kChannels * sizeof(__half);
constexpr std::size_t kCacheOffset = kMiddleBytes + 2U * kStateBytes;

// La première couche somme les 48 canaux fixes en premier. Sauver ses fragments
// FP32 avant les prises mobiles permet de reprendre la même suite de MMA, au
// même arrondi. Le cache n'a besoin d'être relu que si le moteur a déjà gardé
// les images de la paire.
template <bool Reuse>
__global__ void __launch_bounds__(rn::kThreads, 2) conv_with_cache(rn::Params p, float4* cache) {
  constexpr int Kc = 16, CT = 48, Rows = 1, S = 2, KH = 3;
  using Geometry = rn::Geometry<KH, S, Rows>;
  extern __shared__ __align__(16) unsigned char raw[];
  auto* const tile = reinterpret_cast<__half*>(raw);
  const int thread = static_cast<int>(threadIdx.x), warp = thread / 32, lane = thread % 32;
  const int ty = static_cast<int>(blockIdx.y) * Geometry::kTH, tx = static_cast<int>(blockIdx.x) * rn::kTW;
  const int z = static_cast<int>(blockIdx.z);
  const int kcs = p.cin / 16;
  const auto* weights = p.w + static_cast<long>(z) * KH * KH * kcs * (CT / 16) * 32;
  const auto block = (static_cast<std::size_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  auto* const partial = cache + (12 * block * rn::kThreads) + thread;
  rn::Frag<rn::Acc::kF32> acc[2][6];
#pragma unroll
  for (int f = 0; f < 2; ++f) {
#pragma unroll
    for (int n = 0; n < 6; ++n) {
      if constexpr (!Reuse) {
        acc[f][n].clear();
      } else {
        const auto a = __ldg(partial + (6 * f + n) * rn::kThreads);
        acc[f][n].r[0] = a.x;
        acc[f][n].r[1] = a.y;
        acc[f][n].r[2] = a.z;
        acc[f][n].r[3] = a.w;
      }
    }
  }
  constexpr int kFixedChunks = kFixedChannels / Kc;
  const int begin = Reuse ? kFixedChunks : 0;
#pragma unroll 1
  for (int q = begin; q < kcs; ++q) {
    if (q != begin) __syncthreads();
    rn::load_nhwc<Kc, S, Geometry::kIR, Geometry::kIC, false>(tile, p, q * Kc, ty * S - 1, tx * S - 1);
    rn::cp_async_wait_all();
    __syncthreads();
    rn::accumulate<Kc, CT, KH, S, Rows, rn::Acc::kF32, false>(acc, tile, weights, q, kcs, 0, 9, warp, lane);
    if constexpr (!Reuse) {
      if (q + 1 == kFixedChunks) {
#pragma unroll
        for (int f = 0; f < 2; ++f)
#pragma unroll
          for (int n = 0; n < 6; ++n) {
            const auto& a = acc[f][n].r;
            partial[(6 * f + n) * rn::kThreads] = make_float4(a[0], a[1], a[2], a[3]);
          }
      }
    }
  }
  __syncthreads();
  rn::stage<CT, Rows, rn::Acc::kF32>(acc, tile, p.bias + z * CT, warp, lane);
  __syncthreads();
  rn::store<CT, Rows, rn::Out::kNhwc>(tile, p, ty, tx, 0, z * CT);
}

template <class T>
cudaError_t upload_sync(rn::Buffer& buffer, const std::vector<T>& values, cudaStream_t stream) {
  const auto status = buffer.upload(values.data(), values.size() * sizeof(T), stream);
  return status == cudaSuccess ? cudaStreamSynchronize(stream) : status;
}

struct Boundary {
  rn::Layer layer;
  cudaError_t prepare(const RifeNativeConv& description, cudaStream_t stream, int height, int width,
                      int in_pitch, int out_pitch) {
    auto conv = description;
    if (conv.transposed) {
      // DepthToSpace ne fournit que les quatre flots et le masque au warp
      // final.
      conv.cout = kFlowAndMask;
      conv.weights.resize(static_cast<std::size_t>(conv.cout) * 16U * static_cast<std::size_t>(conv.cin));
      conv.bias.resize(static_cast<std::size_t>(conv.cout));
    }
    layer.kind = conv.transposed ? rn::Kind::kPhase : rn::Kind::kHalf;
    layer.ih = height;
    layer.iw = width;
    layer.in_ld = in_pitch;
    layer.out_ch = conv.cout;
    layer.out_ld = out_pitch;
    layer.cin = (conv.cin + 15) / 16 * 16;
    layer.oh = conv.transposed ? height : height / 2;
    layer.ow = conv.transposed ? width : width / 2;
    const auto dense = rn::dense_weights(conv, layer.kind, layer.cin);
    const int ct = conv.transposed ? 32 : 48;
    const int rows = conv.transposed ? 2 : 1;
    layer.sums = RifeNativeSums::kF32;
    layer.phases = dense.phases;
    layer.cout = dense.cout;
    layer.tiles = dense.cout / ct;
    layer.grid = dim3(static_cast<unsigned>((layer.ow + rn::kTW - 1) / rn::kTW),
                      static_cast<unsigned>((layer.oh + rn::kWarps * rows - 1) / (rn::kWarps * rows)),
                      static_cast<unsigned>(dense.phases * layer.tiles));
    layer.smem = rn::smem_bytes(16, ct, rn::kernel_size(layer.kind), rn::stride_of(layer.kind), rows);
    if (conv.transposed) layer.launch = rn::pick_for(layer.sums, layer.kind, 16, ct, rows);
    const auto packed = rn::packed_weights(dense, ct);
    const auto slopes = conv.slope;
    if (auto result = upload_sync(layer.w, packed, stream); result != cudaSuccess) return result;
    if (auto result = upload_sync(layer.bias, dense.bias, stream); result != cudaSuccess) return result;
    if (!slopes.empty()) {
      if (auto result = upload_sync(layer.slope, slopes, stream); result != cudaSuccess) return result;
    }
    return cudaSuccess;
  }

  rn::Params params(const __half* input, __half* output) const {
    return {.in = input,
            .in_ld = layer.in_ld,
            .ih = layer.ih,
            .iw = layer.iw,
            .out = output,
            .out_ld = layer.out_ld,
            .out_ch = layer.out_ch,
            .oh = layer.oh,
            .ow = layer.ow,
            .w = layer.w.as<const uint4>(),
            .bias = layer.bias.as<const float>(),
            .slope = layer.slope.as<const float>(),
            .cin = layer.cin,
            .tiles = layer.tiles};
  }
};

struct Residual {
  rn::Buffer weights, bias, scale;
};

bool compatible_recipe(std::span<const RifeNativeConv> layers) {
  return layers.size() == 11 && layers.front().cin == 109 && layers.front().cout == kMidChannels &&
         layers[1].cin == kMidChannels && layers[1].cout == kChannels && layers.back().cin == kChannels &&
         layers.back().cout == 52 && layers.back().transposed && !layers.front().transposed &&
         !layers[1].transposed && layers.front().stride == 2 && layers[1].stride == 2;
}

class Native final : public DrbaTailNative {
 public:
  cudaError_t setup(std::span<const RifeNativeConv> layers, cudaStream_t stream) {
    if (!compatible_recipe(layers)) return cudaErrorInvalidValue;
    const std::vector<float> slopes(static_cast<std::size_t>(kChannels), 0.2F);
    if (auto status = upload_sync(slopes_, slopes, stream); status != cudaSuccess) return status;
    if (auto status =
            boundary_[0].prepare(layers[0], stream, 4 * kHeight, 4 * kWidth, kPackedInput, kMidChannels);
        status != cudaSuccess)
      return status;
    if (auto status =
            boundary_[1].prepare(layers[1], stream, 2 * kHeight, 2 * kWidth, kMidChannels, kChannels);
        status != cudaSuccess)
      return status;
    if (auto status = boundary_[2].prepare(layers.back(), stream, kHeight, kWidth, kChannels, kOutputPitch);
        status != cudaSuccess)
      return status;
    for (std::size_t i = 0; i < residuals_.size(); ++i) {
      const auto& layer = layers[i + 2];
      if (layer.cin != kChannels || layer.cout != kChannels || layer.stride != 1 || layer.transposed ||
          layer.scale.size() != kChannels)
        return cudaErrorInvalidValue;
      auto& gpu = residuals_[i];
      auto unscaled = layer;
      unscaled.scale.clear();  // garder l'arrondi FP16 avant la multiplication résiduelle
      const auto dense = rn::dense_weights(unscaled, rn::Kind::kSame, kChannels);
      const auto packed = drba_tail_wino::pack(dense);
      if (auto status = upload_sync(gpu.weights, packed, stream); status != cudaSuccess) return status;
      if (auto status = upload_sync(gpu.bias, dense.bias, stream); status != cudaSuccess) return status;
      if (auto status = upload_sync(gpu.scale, layer.scale, stream); status != cudaSuccess) return status;
    }
    return workspace_.allocate(workspace_bytes());
  }

  std::size_t workspace_bytes() const {
    const dim3 grid = boundary_[0].layer.grid;
    return kCacheOffset +
           static_cast<std::size_t>(grid.x) * grid.y * grid.z * rn::kThreads * 12U * sizeof(float4);
  }

  cudaError_t enqueue(cudaStream_t stream, const void* input, void* output, bool reuse) const override {
    auto* const base = workspace_.as<std::byte>();
    auto* const mid = reinterpret_cast<__half*>(base);
    auto* state = reinterpret_cast<__half*>(base + kMiddleBytes);
    auto* next = reinterpret_cast<__half*>(base + kMiddleBytes + kStateBytes);
    auto* const cache = reinterpret_cast<float4*>(base + kCacheOffset);
    const auto& first = boundary_[0];
    const auto& second = boundary_[1];
    const auto entry = first.params(static_cast<const __half*>(input), mid);
    if (reuse)
      conv_with_cache<true><<<first.layer.grid, rn::kThreads, first.layer.smem, stream>>>(entry, cache);
    else
      conv_with_cache<false><<<first.layer.grid, rn::kThreads, first.layer.smem, stream>>>(entry, cache);
    rn::conv_kernel<16, 48, 3, 2, 1, rn::In::kNhwc, rn::Out::kNhwc, rn::Acc::kF32>
        <<<second.layer.grid, rn::kThreads, second.layer.smem, stream>>>(second.params(mid, state));
    using Tile = drba_tail_wino::Tile<kChannels, 16, kChannels>;
    const dim3 grid((kWidth + 15) / 16, (kHeight + Tile::kHeight - 1) / Tile::kHeight);
    for (const auto& layer : residuals_) {
      const rn::Params params{.in = state,
                              .in_ld = kChannels,
                              .ih = kHeight,
                              .iw = kWidth,
                              .out = next,
                              .out_ld = kChannels,
                              .out_ch = kChannels,
                              .oh = kHeight,
                              .ow = kWidth,
                              .w = layer.weights.as<const uint4>(),
                              .bias = layer.bias.as<const float>(),
                              .slope = slopes_.as<const float>(),
                              .skip = state,
                              .skip_ld = kChannels,
                              .cin = kChannels};
      drba_tail_wino::launch<kChannels, 16, kChannels>(grid, Tile::kBytes, stream, params,
                                                       layer.scale.as<const float>());
      std::swap(state, next);
    }
    const auto& projection = boundary_[2];
    projection.layer.launch(projection.layer.grid, projection.layer.smem, stream,
                            projection.params(state, static_cast<__half*>(output)));
    return cudaGetLastError();
  }

 private:
  std::array<Boundary, 3> boundary_;
  std::array<Residual, 8> residuals_;
  rn::Buffer slopes_;
  rn::Buffer workspace_;
};

}  // namespace

std::unique_ptr<DrbaTailNative> DrbaTailNative::create(std::span<const RifeNativeConv> layers,
                                                       cudaStream_t stream, cudaError_t& error) {
  auto native = std::make_unique<Native>();
  error = native->setup(layers, stream);
  if (error != cudaSuccess) return {};
  return native;
}

}  // namespace kirari::gpu::cuda
