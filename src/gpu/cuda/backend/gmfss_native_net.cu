// Le GridNet natif de GMFSS (gmfss_native_net.h) : les convolutions planifiées et réglées une fois par
// gmfss_conv_plan.cuh (formes réglées au banc, poids rangés pour le mma, tenseurs placés dans la zone par
// durée de vie : le résiduel n'est jamais la sortie). Ici, les entrées en plans passent d'abord par pixel
// dans x0 à x3, et la sortie, 8 canaux par pixel, revient en trois plans bornés. Les morceaux : les trois
// lignes du GridNet et la queue (prepare::gmfss_grid_pieces).
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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

// x0 : par pixel, les trois plans de w0, ceux de l'image de l'IFNet, ceux de w1, puis sept zéros.
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

// Les deux sources contiennent déjà des demi-flottants. La transposition les déplace bit pour bit.
constexpr int kPackPixels = 32;
constexpr int kPackChannels = 32;
constexpr int kPackThreads = kPackPixels * (kPackChannels / 8);

// Une tuile lit 32 pixels consécutifs dans chaque plan et écrit 32 canaux contigus par pixel.
// Une tuile lit une seule source : la frontière entre les deux images est un multiple de 32 canaux.
template <bool Blocked8>
__global__ void pack_features(const __half* a, const __half* b, long plane, int channels, __half* out) {
  __shared__ __half tile[kPackChannels][kPackPixels + 1];
  const long x = static_cast<long>(blockIdx.x) * kPackPixels;
  const int c = static_cast<int>(blockIdx.y) * kPackChannels;
  const __half* from =
      c < channels ? a + static_cast<long>(c) * plane : b + static_cast<long>(c - channels) * plane;
  const int lane = static_cast<int>(threadIdx.x);
  const long px = x + lane % kPackPixels;
  if (px < plane) {
    for (int ch = lane / kPackPixels; ch < kPackChannels; ch += kPackThreads / kPackPixels)
      tile[ch][lane % kPackPixels] = from[ch * plane + px];
  }
  __syncthreads();
  constexpr int kGroups = kPackChannels / 8;
  const int local_pixel = lane / kGroups;
  const int group = (lane % kGroups) * 8;
  const long pixel = x + local_pixel;
  if (pixel >= plane) return;
  std::uint32_t w[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    w[j] = pair_bits(tile[group + 2 * j][local_pixel], tile[group + 2 * j + 1][local_pixel]);
  const long at = Blocked8 ? (c + group) * plane + pixel * 8 : pixel * (2 * channels) + c + group;
  *reinterpret_cast<uint4*>(out + at) = make_uint4(w[0], w[1], w[2], w[3]);
}

// Les entrées telles que pack_head et pack_features les remplissent : niveau et canaux de x0 à x3.
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

// La sortie : les trois premiers des 8 canaux par pixel, bornés à [0, 1], en trois plans.
__global__ void unpack_image(const __half* in, long plane, __half* out) {
  const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= plane) return;
#pragma unroll
  for (int c = 0; c < 3; ++c)
    out[c * plane + i] = __float2half(fminf(fmaxf(__half2float(in[i * 8 + c]), 0.F), 1.F));
}

// Les noms des morceaux, dans l'ordre de prepare::gmfss_grid_pieces.
constexpr std::array<const char*, prepare::kGmfssGridPieces> kPieceNames{"ligne 0", "ligne 1", "ligne 2",
                                                                         "queue"};

class Net final : public GmfssNativeNet {
 public:
  Net(int height, int width, bool features_c8)
      : height_(height), width_(width), features_c8_(features_c8), net_(height, width) {}

  bool plan(const GmfssGrid& grid, const gmfss_plan::Tuner& tuner, std::string& error) {
    if (grid.tensors.size() <= static_cast<std::size_t>(prepare::kGmfssGridInputs) || grid.steps.empty()) {
      error = "réseau vide";
      return false;
    }
    const std::vector<int> pieces = prepare::gmfss_grid_pieces(grid);
    if (pieces.empty()) {
      error = "pas de queue au PixelShuffle, ou une convolution hors des trois lignes";
      return false;
    }
    out_ = static_cast<int>(grid.tensors.size()) - 1;
    const gmfss_plan::Io io{.external = {},
                            .image = -1,
                            .written = {0, 1, 2, 3},
                            .kept = {out_},
                            .rgb = out_,
                            .features_c8 = features_c8_};
    if (!net_.plan(grid, io, pieces, tuner, error) || !check(error)) return false;
    use(std::vector<RifeNativePrecision>(kPieceNames.size(), {RifeNativeSums::kF32, RifeNativeSums::kF32}));
    const auto sent = [&](cudaError_t result) {
      if (result == cudaSuccess) return true;
      static_cast<void>(cudaGetLastError());
      error = std::string("zone du GMFSS natif : ") + cudaGetErrorString(result);
      return false;
    };
    return sent(zone_.allocate(static_cast<std::size_t>(std::max<std::int64_t>(net_.zone_bytes(), 256)))) &&
           sent(partial_.allocate(
               static_cast<std::size_t>(std::max<std::int64_t>(net_.partial_bytes(), 256))));
  }

  int pieces() const override { return static_cast<int>(kPieceNames.size()); }
  std::string piece_name(int piece) const override {
    return piece >= 0 && piece < pieces() ? kPieceNames[static_cast<std::size_t>(piece)] : "?";
  }
  bool has_head(int piece) const override { return net_.has_head(piece); }
  bool join_ready() const override { return net_.join_ready(); }
  void use_join(bool enabled) override { net_.use_join(enabled); }
  bool tail_ready() const override { return net_.tail_ready(); }
  void use_tail(bool enabled) override { net_.use_tail(enabled); }
  bool planar_head_ready() const override { return net_.planar_head_ready(); }
  void use_planar_head(bool enabled) override { planar_head_ = enabled; }
  void use(std::span<const RifeNativePrecision> precisions) override { net_.use(precisions); }
  void trim() override { net_.trim(); }
  GmfssTuning& tuning() { return tuning_; }
  const GmfssTuning& tuning() const override { return tuning_; }
  std::int64_t workspace_bytes() const override { return net_.zone_bytes() + net_.partial_bytes(); }
  std::array<void*, 3> feature_inputs() const override {
    void* zone = zone_.as<void>();
    return {gmfss_plan::ConvNet::at(zone, net_.tensor(1)), gmfss_plan::ConvNet::at(zone, net_.tensor(2)),
            gmfss_plan::ConvNet::at(zone, net_.tensor(3))};
  }

  cudaError_t run(cudaStream_t stream, const GmfssNativeInputs& in, void* out) const override {
    const auto plane = [&](int level) { return static_cast<long>(height_ >> level) * (width_ >> level); };
    void* zone = zone_.as<void>();
    const auto at = [&](int tensor) { return gmfss_plan::ConvNet::at(zone, net_.tensor(tensor)); };
    const auto* w01 = static_cast<const __half*>(in.warped0[0]);
    const auto* w11 = static_cast<const __half*>(in.warped1[0]);
    const long p0 = plane(0);
    const std::array<const __half*, 3> images{w01, static_cast<const __half*>(in.rife), w11};
    if (!planar_head_)
      pack_head<<<blocks_for(p0), 256, 0, stream>>>(images[0], images[1], images[2], p0, at(0));
    if (!in.packed_features) {
      const int features = prepare::kGmfssFeatureChannels;
      const auto planes = [&](const void* a, const void* b, long n, int count, int tensor) {
        const dim3 grid(static_cast<unsigned>((n + kPackPixels - 1) / kPackPixels),
                        static_cast<unsigned>(2 * count / kPackChannels));
        if (features_c8_) {
          pack_features<true><<<grid, kPackThreads, 0, stream>>>(
              static_cast<const __half*>(a), static_cast<const __half*>(b), n, count, at(tensor));
        } else {
          pack_features<false><<<grid, kPackThreads, 0, stream>>>(
              static_cast<const __half*>(a), static_cast<const __half*>(b), n, count, at(tensor));
        }
      };
      planes(w01 + 3 * p0, w11 + 3 * p0, p0, features, 1);
      for (int level = 1; level <= 2; ++level) {
        const int c = net_.tensor(level + 1).channels / 2;
        planes(in.warped0[static_cast<std::size_t>(level)], in.warped1[static_cast<std::size_t>(level)],
               plane(level), c, level + 1);
      }
    }
    auto* rgb = static_cast<__half*>(out);
    if (!net_.run(stream, zone, partial_.as<float>(), {}, -1, rgb, planar_head_ ? &images : nullptr)) {
      const long full = 4 * p0;
      unpack_image<<<blocks_for(full), 256, 0, stream>>>(gmfss_plan::ConvNet::at(zone, net_.tensor(out_)),
                                                         full, rgb);
    }
    return cudaGetLastError();
  }

  cudaError_t run_piece(cudaStream_t stream, int piece, void* out) const override {
    if (piece < 0 || piece >= pieces()) return cudaErrorInvalidValue;
    void* zone = zone_.as<void>();
    auto* rgb = out == nullptr ? gmfss_plan::ConvNet::at(zone, net_.tensor(out_)) : static_cast<__half*>(out);
    net_.run(stream, zone, partial_.as<float>(), {}, piece, rgb);
    return cudaGetLastError();
  }

 private:
  // La sortie, l'image au double de la taille ; les entrées, aux formes que remplissent pack_head et
  // pack_features.
  bool check(std::string& error) {
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
    }
    return true;
  }

  int height_;
  int width_;
  bool features_c8_;
  bool planar_head_ = false;
  gmfss_plan::ConvNet net_;
  GmfssTuning tuning_;
  int out_ = 0;
  gmfss_plan::rn::Buffer zone_;
  gmfss_plan::rn::Buffer partial_;
};

}  // namespace

std::unique_ptr<GmfssNativeNet> GmfssNativeNet::create(const prepare::GmfssGrid& grid, int height, int width,
                                                       cudaStream_t stream, std::string& error,
                                                       bool features_c8) {
  if (height < 32 || width < 32 || height % 32 != 0 || width % 32 != 0) {
    error = "taille de la fusion : des multiples de 32 attendus";
    return nullptr;
  }
  gmfss_plan::rn::Bench bench(stream);
  if (!bench.ready()) {
    error = "événements CUDA du réglage";
    return nullptr;
  }
  auto net = std::make_unique<Net>(height, width, features_c8);
  if (!net->plan(grid, gmfss_plan::Tuner{.bench = bench, .stream = stream, .tuning = net->tuning()}, error))
    return nullptr;
  // Les envois sur stream finis avant qu'une autre voie (la lecture) lance le réseau.
  if (const cudaError_t result = cudaStreamSynchronize(stream); result != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    error = std::string("chargement : ") + cudaGetErrorString(result);
    return nullptr;
  }
  return net;
}

}  // namespace kirari::gpu::cuda
