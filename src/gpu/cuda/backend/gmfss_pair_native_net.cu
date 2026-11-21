// Le FeatureNet et le MetricNet natifs de GMFSS (gmfss_pair_native_net.h) : chaque réseau planifié et réglé
// une fois par gmfss_conv_plan.cuh (un banc pour les deux), dans une zone commune. Le FeatureNet lit l'image
// en plans (In::kImage, pas 2) et ses trois sorties, rangées par pixel, reviennent en plans ; l'entrée du
// MetricNet passe d'abord par pixel, et sa sortie (2 canaux sur 8) devient la métrique des six mouvements :
// tanh × 10, puis les réductions du Resize linéaire de la paire (half_pixel : la moitié est la moyenne 2 × 2,
// le quart celle des pixels 4x + 1 et 4x + 2 sur chaque axe).
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "gpu/cuda/backend/gmfss_conv_plan.cuh"
#include "gpu/cuda/backend/gmfss_metric_motion.cuh"
#include "gpu/cuda/backend/gmfss_pair_native_net.h"

namespace kirari::gpu::cuda {
namespace {

using gmfss_plan::blocks_for;
using gmfss_plan::ConvNet;
using gmfss_plan::Io;
using gmfss_plan::rn::Buffer;
using prepare::kGmfssFeatureLevels;
using prepare::kGmfssMetricOutput;
using prepare::kGmfssMetricPlanes;

__device__ __forceinline__ std::uint32_t pair_bits(__half lo, __half hi) {
  return static_cast<std::uint32_t>(__half_as_ushort(lo)) |
         (static_cast<std::uint32_t>(__half_as_ushort(hi)) << 16U);
}

// Les 14 plans de l'entrée du MetricNet en 16 canaux par pixel (les deux derniers nuls).
__global__ void pack_metric(const __half* planes, long plane, __half* out) {
  const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= plane) return;
  std::uint32_t w[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int c = 2 * j;
    const __half zero = __float2half(0.F);
    const __half lo = c < kGmfssMetricPlanes ? planes[c * plane + i] : zero;
    const __half hi = c + 1 < kGmfssMetricPlanes ? planes[(c + 1) * plane + i] : zero;
    w[j] = pair_bits(lo, hi);
  }
  auto* to = reinterpret_cast<uint4*>(out + i * 16);
  to[0] = make_uint4(w[0], w[1], w[2], w[3]);
  to[1] = make_uint4(w[4], w[5], w[6], w[7]);
}

// Les count premiers canaux (multiple de 8) d'un tenseur rangé par pixel (ld canaux) en count plans ; huit
// canaux par fil, les fils voisins sur des pixels voisins (écritures des plans groupées).
__global__ void unpack_planes(const __half* in, int ld, long plane, int count, __half* out) {
  const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int groups = count / 8;
  if (i >= plane * groups) return;
  const auto g = static_cast<int>(i / plane);
  const long px = i - g * plane;
  const uint4 v = *reinterpret_cast<const uint4*>(in + px * ld + g * 8);
  const std::uint32_t w[4] = {v.x, v.y, v.z, v.w};
  __half* to = out + static_cast<long>(g) * 8 * plane + px;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    to[(2 * j) * plane] = __ushort_as_half(static_cast<unsigned short>(w[j] & 0xFFFFU));
    to[(2 * j + 1) * plane] = __ushort_as_half(static_cast<unsigned short>(w[j] >> 16U));
  }
}

// Transposer une tuile rend les lectures NHWC et les écritures planaires contiguës.
constexpr int kUnpackSide = 32, kUnpackThreads = 256;

__global__ void unpack_tiles(const __half* in, int ld, int plane, int count, __half* out) {
  __shared__ __half transposed[kUnpackSide][kUnpackSide + 2];
  const int left = static_cast<int>(blockIdx.x) * kUnpackSide;
  const int first = static_cast<int>(blockIdx.y) * kUnpackSide;
  const int thread = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int i = thread; i < kUnpackSide * kUnpackSide; i += kUnpackThreads) {
    const int x = i / kUnpackSide, c = i % kUnpackSide;
    if (left + x < plane && first + c < count)
      transposed[x][c] = in[static_cast<std::size_t>(left + x) * ld + first + c];
  }
  __syncthreads();
#pragma unroll
  for (int i = thread; i < kUnpackSide * kUnpackSide; i += kUnpackThreads) {
    const int x = i % kUnpackSide, c = i / kUnpackSide;
    if (left + x < plane && first + c < count)
      out[static_cast<std::size_t>(first + c) * plane + left + x] = transposed[x][c];
  }
}

bool tiled_unpack_for(int height, int width) {
  if (height != 576 || width != 960) return false;
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return false;
  }
  return properties.major == 8 && properties.minor == 9;
}

class Net final : public GmfssPairNativeNet {
 public:
  Net(int height, int width)
      : height_(height),
        width_(width),
        tiled_unpack_(tiled_unpack_for(height, width)),
        features_(height, width),
        metric_(height, width) {}

  bool plan(const prepare::GmfssPairNets& nets, const gmfss_plan::Tuner& tuner, std::string& error) {
    const Io feature_io{.external = {prepare::kGmfssFeatureImage},
                        .image = prepare::kGmfssFeatureImage,
                        .written = {},
                        .kept = {1, 2, 3}};
    if (!features_.plan(nets.features, feature_io, {}, tuner, error)) {
      error = "FeatureNet, " + error;
      return false;
    }
    const Io metric_io{
        .external = {}, .image = -1, .written = {prepare::kGmfssMetricInput}, .kept = {kGmfssMetricOutput}};
    if (!metric_.plan(nets.metric, metric_io, {}, tuner, error)) {
      error = "MetricNet, " + error;
      return false;
    }
    if (!check(error)) return false;
    use(std::vector<RifeNativePrecision>(2, {RifeNativeSums::kF32, RifeNativeSums::kF32}));
    zone_bytes_ = std::max<std::int64_t>({features_.zone_bytes(), metric_.zone_bytes(), 256});
    partial_bytes_ = std::max<std::int64_t>({features_.partial_bytes(), metric_.partial_bytes(), 256});
    const auto sent = [&](cudaError_t result) {
      if (result == cudaSuccess) return true;
      static_cast<void>(cudaGetLastError());
      error = std::string("zone de la paire native : ") + cudaGetErrorString(result);
      return false;
    };
    return sent(zone_.allocate(static_cast<std::size_t>(zone_bytes_))) &&
           sent(partial_.allocate(static_cast<std::size_t>(partial_bytes_)));
  }

  int pieces() const override { return 2; }
  std::string piece_name(int piece) const override {
    return piece == kFeatureNet ? "FeatureNet" : piece == kMetricNet ? "MetricNet" : "?";
  }
  bool has_head(int piece) const override {
    return piece == kFeatureNet ? features_.has_head(0) : piece == kMetricNet && metric_.has_head(0);
  }
  void use(std::span<const RifeNativePrecision> precisions) override {
    if (precisions.size() < 2) return;
    features_.use(precisions.subspan(kFeatureNet, 1));
    metric_.use(precisions.subspan(kMetricNet, 1));
  }
  void trim() override {
    features_.trim();
    metric_.trim();
  }
  GmfssTuning& tuning() { return tuning_; }
  const GmfssTuning& tuning() const override { return tuning_; }
  std::int64_t workspace_bytes() const override { return zone_bytes_ + partial_bytes_; }

  cudaError_t features(cudaStream_t stream, const void* image, const GmfssPairFeatures& out) const override {
    const std::array<const void*, 1> bound{image};
    features_.run(stream, zone_.as<void>(), partial_.as<float>(), bound);
    for (int level = 0; level < kGmfssFeatureLevels; ++level) {
      const gmfss_plan::Placed& t = features_.tensor(level + 1);
      const long plane = static_cast<long>(t.height) * t.width;
      const auto* input = ConvNet::at(zone_.as<void>(), t);
      auto* output = static_cast<__half*>(out.levels[static_cast<std::size_t>(level)]);
      if (tiled_unpack_ && ((level == 0 && t.channels == 64) || (level == 1 && t.channels == 128))) {
        const dim3 grid(static_cast<unsigned>((plane + kUnpackSide - 1) / kUnpackSide),
                        static_cast<unsigned>(t.channels / kUnpackSide));
        unpack_tiles<<<grid, kUnpackThreads, 0, stream>>>(input, t.channels, static_cast<int>(plane),
                                                          t.channels, output);
      } else {
        unpack_planes<<<blocks_for(plane * (t.channels / 8)), 256, 0, stream>>>(input, t.channels, plane,
                                                                                t.channels, output);
      }
    }
    return cudaGetLastError();
  }

  cudaError_t metric(cudaStream_t stream, const void* in, const GmfssPairMotion& motion) const override {
    const long plane = static_cast<long>(height_) * width_;
    pack_metric<<<blocks_for(plane), 256, 0, stream>>>(
        static_cast<const __half*>(in), plane,
        ConvNet::at(zone_.as<void>(), metric_.tensor(prepare::kGmfssMetricInput)));
    metric_.run(stream, zone_.as<void>(), partial_.as<float>(), {});
    gmfss_metric::MotionPlanes planes{};
    for (int side = 0; side < 2; ++side)
      for (int level = 0; level < 3; ++level) {
        const long level_plane = static_cast<long>(height_ >> level) * (width_ >> level);
        planes.metric[side][level] =
            static_cast<__half*>(motion.motion[static_cast<std::size_t>(side * 3 + level)]) + 2 * level_plane;
      }
    const auto* logits = ConvNet::at(zone_.as<void>(), metric_.tensor(kGmfssMetricOutput));
    return gmfss_metric::launch_metric_motion(stream, logits, height_, width_, planes);
  }

 private:
  // Les sorties du FeatureNet aux trois niveaux ; l'entrée et la sortie du MetricNet.
  bool check(std::string& error) const {
    for (int level = 0; level < kGmfssFeatureLevels; ++level) {
      const gmfss_plan::Placed& t = features_.tensor(level + 1);
      if (t.height != height_ >> level || t.width != width_ >> level || t.channels % 8 != 0) {
        error = "FeatureNet : sortie " + std::to_string(level + 1);
        return false;
      }
    }
    const gmfss_plan::Placed& in = metric_.tensor(prepare::kGmfssMetricInput);
    const gmfss_plan::Placed& out = metric_.tensor(kGmfssMetricOutput);
    if (in.channels != 16 || out.channels != 8 || out.height != height_ || out.width != width_) {
      error = "MetricNet : entrée ou sortie";
      return false;
    }
    return true;
  }

  int height_;
  int width_;
  bool tiled_unpack_;
  ConvNet features_;
  ConvNet metric_;
  GmfssTuning tuning_;
  std::int64_t zone_bytes_ = 0;
  std::int64_t partial_bytes_ = 0;
  Buffer zone_;
  Buffer partial_;
};

}  // namespace

std::unique_ptr<GmfssPairNativeNet> GmfssPairNativeNet::create(const prepare::GmfssPairNets& nets, int height,
                                                               int width, cudaStream_t stream,
                                                               std::string& error) {
  if (height < 32 || width < 32 || height % 32 != 0 || width % 32 != 0) {
    error = "taille de la fusion : des multiples de 32 attendus";
    return nullptr;
  }
  gmfss_plan::rn::Bench bench(stream);
  if (!bench.ready()) {
    error = "événements CUDA du réglage";
    return nullptr;
  }
  auto net = std::make_unique<Net>(height, width);
  if (!net->plan(nets, gmfss_plan::Tuner{.bench = bench, .stream = stream, .tuning = net->tuning()}, error))
    return nullptr;
  // Les envois sur stream finis avant qu'une autre voie (la lecture) lance les réseaux.
  if (const cudaError_t result = cudaStreamSynchronize(stream); result != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    error = std::string("chargement : ") + cudaGetErrorString(result);
    return nullptr;
  }
  return net;
}

}  // namespace kirari::gpu::cuda
