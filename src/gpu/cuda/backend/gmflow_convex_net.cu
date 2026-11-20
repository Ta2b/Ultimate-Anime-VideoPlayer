#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "gpu/cuda/backend/gmflow_backbone_kernels.cuh"
#include "gpu/cuda/backend/gmflow_convex_kernels.h"
#include "gpu/cuda/backend/gmflow_convex_math.h"
#include "gpu/cuda/backend/gmflow_convex_net.h"
#include "gpu/cuda/backend/rife_native_layer.h"

namespace kirari::gpu::cuda {
namespace {

namespace gb = gmflow_backbone;
namespace rn = rife_native;
using namespace gmflow_convex;
constexpr int kHidden = 256;
constexpr int kDirections = 2;
constexpr int kRows = 2;
constexpr int kHeight = gb::kWarps * kRows;
constexpr int kSmemLimit = 48 * 1024;
static_assert(kInput == kChannels);

bool weights_match(const prepare::GmflowConv& c, int kernel, int cin, int cout) {
  return c.kernel == kernel && c.cin == cin && c.cout == cout &&
         c.weights.size() == static_cast<std::size_t>(kernel * kernel * cin * cout) &&
         c.bias.size() == static_cast<std::size_t>(cout);
}

struct Conv {
  rn::Buffer w, bias;
  gb::Params p;
  dim3 grid;

  bool build(const prepare::GmflowConv& c, int padded_in, int ct, int height, int width, std::string& error) {
    rn::Dense dense{.cout = c.cout, .taps = c.kernel * c.kernel, .cin = padded_in, .w = {}, .bias = {}};
    const auto vectors = static_cast<std::size_t>(c.cout * dense.taps);
    const auto in = static_cast<std::size_t>(c.cin);
    dense.w.resize(vectors * static_cast<std::size_t>(padded_in), __float2half_rn(0.F));
    for (std::size_t i = 0; i < vectors; ++i) {
      const float* from = c.weights.data() + i * in;
      __half* to = dense.w.data() + i * static_cast<std::size_t>(padded_in);
      std::transform(from, from + in, to, [](float v) { return __float2half_rn(v); });
    }
    const auto packed = rn::packed_weights(dense, ct);
    cudaError_t result = w.upload(packed.data(), packed.size() * sizeof(std::uint32_t), cudaStreamPerThread);
    if (result == cudaSuccess)
      result = bias.upload(c.bias.data(), c.bias.size() * sizeof(float), cudaStreamPerThread);
    const cudaError_t synced = cudaStreamSynchronize(cudaStreamPerThread);
    if (result == cudaSuccess) result = synced;
    if (result != cudaSuccess) {
      error = cudaGetErrorString(result);
      return false;
    }
    p = {.in_ld = padded_in,
         .ih = height,
         .iw = width,
         .out_ld = c.cout,
         .oh = height,
         .ow = width,
         .w = w.as<const uint4>(),
         .bias = bias.as<const float>(),
         .cin = padded_in,
         .tiles = c.cout / ct};
    grid = {static_cast<unsigned>((width + gb::kTW - 1) / gb::kTW),
            static_cast<unsigned>((height + kHeight - 1) / kHeight),
            static_cast<unsigned>(kDirections * p.tiles)};
    return true;
  }

  template <int Kc, int CT, int Kernel, bool Relu = false>
  cudaError_t run(cudaStream_t stream, const __half* in, __half* out) const {
    using G = rn::Geometry<Kernel, 1, kRows>;
    constexpr int input_bytes = G::kIR * G::kIC * (Kc + 8) * sizeof(__half);
    constexpr int output_bytes =
        kHeight * gb::kTW * (CT + 8) * sizeof(__half) + (Relu ? 0 : gb::kReduceBytes);
    constexpr int smem = std::max(input_bytes, output_bytes);
    static_assert(smem <= kSmemLimit);
    gb::Params shot = p;
    shot.in = in;
    shot.out = out;
    gb::conv_kernel<Kc, CT, Kernel, 1, kRows, gb::In::kPlain, gb::Acc::kF32, Relu>
        <<<grid, gb::kThreads, smem, stream>>>(shot);
    return cudaGetLastError();
  }
};

class Net final : public GmflowConvexNet {
 public:
  Net(int height, int width) : height_(height), width_(width) {}

  bool build(const prepare::GmflowUpsampler& weights, std::string& error) {
    return first_.build(weights.first, kInput, 32, height_, width_, error) &&
           mask_.build(weights.mask, kHidden, 16, height_, width_, error);
  }

  std::int64_t workspace_bytes() const override { return input_bytes() + tensor_bytes(kHidden); }

  cudaError_t run(cudaStream_t stream, const void* features, const float* flow, void* workspace,
                  float* output) const override {
    if (workspace == nullptr || reinterpret_cast<std::uintptr_t>(workspace) % 16 != 0 ||
        features == workspace || flow == workspace || output == workspace || features == output ||
        flow == output || output == nullptr || reinterpret_cast<std::uintptr_t>(output) % alignof(float) != 0)
      return cudaErrorInvalidValue;
    auto* packed = static_cast<__half*>(workspace);
    auto* hidden = reinterpret_cast<__half*>(static_cast<std::byte*>(workspace) + input_bytes());
    cudaError_t result = launch_gmflow_convex_input(stream, features, flow, height_, width_, packed);
    if (result == cudaSuccess) result = first_.run<16, 32, 3, true>(stream, packed, hidden);
    // Après la 3 × 3, sa zone d'entrée reçoit les logits de même taille.
    if (result == cudaSuccess) result = mask_.run<64, 16, 1>(stream, hidden, packed);
    if (result == cudaSuccess) result = launch_gmflow_convex(stream, packed, flow, height_, width_, output);
    return result;
  }

 private:
  std::int64_t tensor_bytes(int channels) const {
    return std::int64_t{kDirections} * height_ * width_ * channels * sizeof(__half);
  }
  std::int64_t input_bytes() const { return rn::align256(tensor_bytes(kInput)); }
  int height_;
  int width_;
  Conv first_, mask_;
};

}  // namespace

std::unique_ptr<GmflowConvexNet> GmflowConvexNet::create(const prepare::GmflowUpsampler& weights, int height,
                                                         int width, std::string& error) {
  if (const auto pixels = std::int64_t{height} * width;
      height < 1 || width < 1 || height > 65535 * kHeight ||
      pixels > (std::numeric_limits<int>::max() - 128) / (kDirections * (kInput / 2))) {
    error = "agrandissement convexe : taille au quart";
    return nullptr;
  }
  if (!weights_match(weights.first, 3, kFeatures + 2, kHidden) ||
      !weights_match(weights.mask, 1, kHidden, kChannels)) {
    error = "agrandissement convexe : formes des convolutions";
    return nullptr;
  }
  auto net = std::make_unique<Net>(height, width);
  if (!net->build(weights, error)) return nullptr;
  return net;
}

}  // namespace kirari::gpu::cuda
