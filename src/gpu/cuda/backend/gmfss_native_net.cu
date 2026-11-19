// Le GridNet natif de GMFSS (gmfss_native_net.h) : les convolutions planifiées une fois par
// gmfss_conv_plan.cuh (tranches, tuiles, grilles, K découpé, poids rangés pour le mma, tenseurs placés dans
// la zone par durée de vie : le résiduel n'est jamais la sortie). Ici, les entrées en plans passent d'abord
// par pixel dans x0 à x3, et la sortie, 8 canaux par pixel, revient en trois plans bornés.
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "gpu/cuda/backend/gmfss_conv_plan.cuh"
#include "gpu/cuda/backend/gmfss_native_net.h"

namespace kirari::gpu::cuda {
namespace {

using gmfss_plan::blocks_for;
using prepare::GmfssGrid;

__device__ __forceinline__ std::uint32_t pair_bits(__half lo, __half hi) {
  return static_cast<std::uint32_t>(__half_as_ushort(lo)) |
         (static_cast<std::uint32_t>(__half_as_ushort(hi)) << 16U);
}

// x0 : par pixel, les trois plans de w0, ceux de rife, ceux de w1, puis sept zéros.
__global__ void pack_head(const __half* w0, const __half* rife, const __half* w1, long plane, __half* out) {
  const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= plane) return;
  const __half zero = __float2half(0.F);
  const __half v[10] = {
      w0[i], w0[plane + i], w0[2 * plane + i], rife[i], rife[plane + i], rife[2 * plane + i],
      w1[i], w1[plane + i], w1[2 * plane + i], zero};
  auto* to = reinterpret_cast<uint4*>(out + i * 16);
  to[0] =
      make_uint4(pair_bits(v[0], v[1]), pair_bits(v[2], v[3]), pair_bits(v[4], v[5]), pair_bits(v[6], v[7]));
  to[1] = make_uint4(pair_bits(v[8], v[9]), 0U, 0U, 0U);
}

// count plans de src (pas plane) vers les canaux [c0, c0 + count) de out, ld canaux par pixel ; huit canaux
// par fil, les fils voisins sur des pixels voisins (lectures des plans groupées).
__global__ void pack_planes(const __half* src, long plane, int count, __half* out, int ld, int c0) {
  const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int groups = count / 8;
  if (i >= plane * groups) return;
  const auto g = static_cast<int>(i / plane);
  const long px = i - g * plane;
  const __half* from = src + static_cast<long>(g) * 8 * plane + px;
  std::uint32_t w[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) w[j] = pair_bits(from[(2 * j) * plane], from[(2 * j + 1) * plane]);
  *reinterpret_cast<uint4*>(out + px * ld + c0 + g * 8) = make_uint4(w[0], w[1], w[2], w[3]);
}

// La sortie : les trois premiers des 8 canaux par pixel, bornés à [0, 1], en trois plans.
// Les entrées telles que pack_head et pack_planes les remplissent : niveau et canaux de x0 à x3.
struct InputForm {
  int level = 0;
  int channels = 0;
};
constexpr std::array<InputForm, prepare::kGmfssGridInputs> kInputForms{{
    {.level = 0, .channels = 16},
    {.level = 0, .channels = 2 * prepare::kGmfssFeatureChannels},
    {.level = 1, .channels = 2 * 128},
    {.level = 2, .channels = 2 * 192},
}};

__global__ void unpack_image(const __half* in, long plane, __half* out) {
  const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= plane) return;
#pragma unroll
  for (int c = 0; c < 3; ++c)
    out[c * plane + i] = __float2half(fminf(fmaxf(__half2float(in[i * 8 + c]), 0.F), 1.F));
}

class Net final : public GmfssNativeNet {
 public:
  Net(int height, int width, bool acc32)
      : height_(height), width_(width), acc32_(acc32), net_(height, width, acc32) {}

  bool plan(const GmfssGrid& grid, std::string& error) {
    if (grid.tensors.size() <= static_cast<std::size_t>(prepare::kGmfssGridInputs) || grid.steps.empty()) {
      error = "réseau vide";
      return false;
    }
    out_ = static_cast<int>(grid.tensors.size()) - 1;
    gmfss_plan::Io io{.external = {}, .image = -1, .written = {0, 1, 2, 3}, .kept = {out_}};
    if (!net_.plan(grid, io, error)) return false;
    const gmfss_plan::Placed& out = net_.tensor(out_);
    if (out.channels != 8 || out.height != 2 * height_ || out.width != 2 * width_) {
      error = "la sortie n'est pas l'image au double de la taille";
      return false;
    }
    for (std::size_t i = 0; i < kInputForms.size(); ++i) {
      const gmfss_plan::Placed& t = net_.tensor(static_cast<int>(i));
      if (t.channels != kInputForms[i].channels || t.height != height_ >> kInputForms[i].level ||
          t.width != width_ >> kInputForms[i].level) {
        error = "entrée x" + std::to_string(i) + " de forme inattendue";
        return false;
      }
      inputs_[i] = t;
    }
    const auto sent = [&](cudaError_t result) {
      if (result == cudaSuccess) return true;
      static_cast<void>(cudaGetLastError());
      error = std::string("zone du GMFSS natif : ") + cudaGetErrorString(result);
      return false;
    };
    return sent(zone_.allocate(static_cast<std::size_t>(net_.zone_bytes()))) &&
           sent(partial_.allocate(static_cast<std::size_t>(net_.partial_bytes())));
  }

  std::int64_t workspace_bytes() const override { return net_.zone_bytes() + net_.partial_bytes(); }
  bool acc32() const override { return acc32_; }

  cudaError_t run(cudaStream_t stream, const GmfssNativeInputs& in, void* out) const override {
    const auto plane = [&](int level) { return static_cast<long>(height_ >> level) * (width_ >> level); };
    void* zone = zone_.as<void>();
    const auto at = [&](int tensor) {
      return gmfss_plan::ConvNet::at(zone, inputs_[static_cast<std::size_t>(tensor)]);
    };
    const auto* w01 = static_cast<const __half*>(in.warped0[0]);
    const auto* w11 = static_cast<const __half*>(in.warped1[0]);
    const long p0 = plane(0);
    pack_head<<<blocks_for(p0), 256, 0, stream>>>(w01, static_cast<const __half*>(in.rife), w11, p0, at(0));
    const int features = prepare::kGmfssFeatureChannels;
    const auto planes = [&](const void* src, long n, int count, int tensor, int c0) {
      pack_planes<<<blocks_for(n * (count / 8)), 256, 0, stream>>>(
          static_cast<const __half*>(src), n, count, at(tensor),
          inputs_[static_cast<std::size_t>(tensor)].channels, c0);
    };
    planes(w01 + 3 * p0, p0, features, 1, 0);
    planes(w11 + 3 * p0, p0, features, 1, features);
    for (int level = 1; level <= 2; ++level) {
      const int c = inputs_[static_cast<std::size_t>(level + 1)].channels / 2;
      planes(in.warped0[static_cast<std::size_t>(level)], plane(level), c, level + 1, 0);
      planes(in.warped1[static_cast<std::size_t>(level)], plane(level), c, level + 1, c);
    }
    net_.run(stream, zone, partial_.as<float>(), {});
    const long full = 4 * p0;
    unpack_image<<<blocks_for(full), 256, 0, stream>>>(gmfss_plan::ConvNet::at(zone, net_.tensor(out_)), full,
                                                       static_cast<__half*>(out));
    return cudaGetLastError();
  }

 private:
  int height_;
  int width_;
  bool acc32_;
  gmfss_plan::ConvNet net_;
  std::array<gmfss_plan::Placed, prepare::kGmfssGridInputs> inputs_{};
  int out_ = 0;
  gmfss_plan::Buffer zone_;
  gmfss_plan::Buffer partial_;
};

}  // namespace

std::unique_ptr<GmfssNativeNet> GmfssNativeNet::create(const prepare::GmfssGrid& grid, int height, int width,
                                                       bool acc32, std::string& error) {
  if (height < 32 || width < 32 || height % 32 != 0 || width % 32 != 0) {
    error = "taille de la fusion : des multiples de 32 attendus";
    return nullptr;
  }
  auto net = std::make_unique<Net>(height, width, acc32);
  if (!net->plan(grid, error)) return nullptr;
  return net;
}

}  // namespace kirari::gpu::cuda
