// Le FeatureNet et le MetricNet natifs de GMFSS (gmfss_pair_native_net.h) : chaque réseau planifié une fois
// par gmfss_conv_plan.cuh, dans une zone commune. Le FeatureNet lit l'image en plans (In::kImage, pas 2) et
// ses trois sorties, rangées par pixel, reviennent en plans ; l'entrée du MetricNet passe d'abord par pixel,
// et sa sortie (2 canaux sur 8) devient la métrique des six mouvements : tanh × 10, puis les réductions du
// Resize linéaire de la paire (half_pixel : la moitié est la moyenne 2 × 2, le quart celle des pixels 4x + 1
// et 4x + 2 sur chaque axe).
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "gpu/cuda/backend/gmfss_conv_plan.cuh"
#include "gpu/cuda/backend/gmfss_pair_native_net.h"

namespace kirari::gpu::cuda {
namespace {

using gmfss_plan::blocks_for;
using gmfss_plan::Buffer;
using gmfss_plan::ConvNet;
using gmfss_plan::Io;
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

struct MotionPlanes {
  __half* metric[2][3];  // le troisième plan de motion{k}_{l}
};

// La métrique du côté side au pixel (y, x) : tanh arrondi en FP16, puis × 10 arrondi, comme Tanh et Mul de la
// paire en FP16.
__device__ __forceinline__ float metric_at(const __half* m, int w, int y, int x, int side) {
  const __half t = __float2half(tanhf(__half2float(m[(static_cast<long>(y) * w + x) * 8 + side])));
  return __half2float(__float2half(__half2float(t) * prepare::kGmfssMetricScale));
}

// Un fil par pixel de la taille de la fusion : il écrit ses deux métriques, et celles de la moitié et du
// quart dont il est le coin.
__global__ void metric_motion(const __half* m, int h, int w, MotionPlanes out) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y);
  if (x >= w || y >= h) return;
#pragma unroll
  for (int side = 0; side < 2; ++side) {
    out.metric[side][0][static_cast<long>(y) * w + x] = __float2half(metric_at(m, w, y, x, side));
    if ((x & 1) == 0 && (y & 1) == 0) {
      const float sum = metric_at(m, w, y, x, side) + metric_at(m, w, y, x + 1, side) +
                        metric_at(m, w, y + 1, x, side) + metric_at(m, w, y + 1, x + 1, side);
      out.metric[side][1][static_cast<long>(y / 2) * (w / 2) + x / 2] = __float2half(0.25F * sum);
    }
    if ((x & 3) == 0 && (y & 3) == 0) {
      const float sum = metric_at(m, w, y + 1, x + 1, side) + metric_at(m, w, y + 1, x + 2, side) +
                        metric_at(m, w, y + 2, x + 1, side) + metric_at(m, w, y + 2, x + 2, side);
      out.metric[side][2][static_cast<long>(y / 4) * (w / 4) + x / 4] = __float2half(0.25F * sum);
    }
  }
}

class Net final : public GmfssPairNativeNet {
 public:
  Net(int height, int width, RifeNativeSums sums)
      : height_(height),
        width_(width),
        sums_(sums),
        features_(height, width, sums),
        metric_(height, width, sums) {}

  bool plan(const prepare::GmfssPairNets& nets, std::string& error) {
    const Io feature_io{.external = {prepare::kGmfssFeatureImage},
                        .image = prepare::kGmfssFeatureImage,
                        .written = {},
                        .kept = {1, 2, 3}};
    if (!features_.plan(nets.features, feature_io, error)) {
      error = "FeatureNet, " + error;
      return false;
    }
    const Io metric_io{
        .external = {}, .image = -1, .written = {prepare::kGmfssMetricInput}, .kept = {kGmfssMetricOutput}};
    if (!metric_.plan(nets.metric, metric_io, error)) {
      error = "MetricNet, " + error;
      return false;
    }
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
    zone_bytes_ = std::max(features_.zone_bytes(), metric_.zone_bytes());
    partial_bytes_ = std::max(features_.partial_bytes(), metric_.partial_bytes());
    const auto sent = [&](cudaError_t result) {
      if (result == cudaSuccess) return true;
      static_cast<void>(cudaGetLastError());
      error = std::string("zone de la paire native : ") + cudaGetErrorString(result);
      return false;
    };
    return sent(zone_.allocate(static_cast<std::size_t>(zone_bytes_))) &&
           sent(partial_.allocate(static_cast<std::size_t>(partial_bytes_)));
  }

  std::int64_t workspace_bytes() const override { return zone_bytes_ + partial_bytes_; }
  RifeNativeSums sums() const override { return sums_; }

  cudaError_t features(cudaStream_t stream, const void* image, const GmfssPairFeatures& out) const override {
    const std::array<const void*, 1> bound{image};
    features_.run(stream, zone_.as<void>(), partial_.as<float>(), bound);
    for (int level = 0; level < kGmfssFeatureLevels; ++level) {
      const gmfss_plan::Placed& t = features_.tensor(level + 1);
      const long plane = static_cast<long>(t.height) * t.width;
      unpack_planes<<<blocks_for(plane * (t.channels / 8)), 256, 0, stream>>>(
          ConvNet::at(zone_.as<void>(), t), t.channels, plane, t.channels,
          static_cast<__half*>(out.levels[static_cast<std::size_t>(level)]));
    }
    return cudaGetLastError();
  }

  cudaError_t metric(cudaStream_t stream, const void* in, const GmfssPairMotion& motion) const override {
    const long plane = static_cast<long>(height_) * width_;
    pack_metric<<<blocks_for(plane), 256, 0, stream>>>(
        static_cast<const __half*>(in), plane,
        ConvNet::at(zone_.as<void>(), metric_.tensor(prepare::kGmfssMetricInput)));
    metric_.run(stream, zone_.as<void>(), partial_.as<float>(), {});
    MotionPlanes planes{};
    for (int side = 0; side < 2; ++side)
      for (int level = 0; level < 3; ++level) {
        const long level_plane = static_cast<long>(height_ >> level) * (width_ >> level);
        planes.metric[side][level] =
            static_cast<__half*>(motion.motion[static_cast<std::size_t>(side * 3 + level)]) + 2 * level_plane;
      }
    const dim3 grid(static_cast<unsigned>((width_ + 127) / 128), static_cast<unsigned>(height_));
    metric_motion<<<grid, 128, 0, stream>>>(ConvNet::at(zone_.as<void>(), metric_.tensor(kGmfssMetricOutput)),
                                            height_, width_, planes);
    return cudaGetLastError();
  }

 private:
  int height_;
  int width_;
  RifeNativeSums sums_;
  ConvNet features_;
  ConvNet metric_;
  std::int64_t zone_bytes_ = 0;
  std::int64_t partial_bytes_ = 0;
  Buffer zone_;
  Buffer partial_;
};

}  // namespace

std::unique_ptr<GmfssPairNativeNet> GmfssPairNativeNet::create(const prepare::GmfssPairNets& nets, int height,
                                                               int width, RifeNativeSums sums,
                                                               std::string& error) {
  if (height < 32 || width < 32 || height % 32 != 0 || width % 32 != 0) {
    error = "taille de la fusion : des multiples de 32 attendus";
    return nullptr;
  }
  auto net = std::make_unique<Net>(height, width, sums);
  if (!net->plan(nets, error)) return nullptr;
  return net;
}

}  // namespace kirari::gpu::cuda
