// Les couches d'attention de GMFlow natif (gmflow_attention_net.h) : les poids de chaque couche rangés une
// fois pour le mma (QKV 384 × 128 en trois parties, fusion 128 × 128), la géométrie des fenêtres de
// l'échelle, et pour chaque couche deux lancements, la GEMM QKV puis l'attention fondue.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/gmflow_attention_kernels.cuh"
#include "gpu/cuda/backend/gmflow_attention_net.h"
#include "gpu/cuda/backend/gmflow_propagation_net.h"

namespace kirari::gpu::cuda {
namespace {

namespace ga = gmflow_attention;
using prepare::GmflowAttention;
using prepare::GmflowAttentionLayer;

static_assert(ga::kD == prepare::kGmflowModel && ga::kQkv == prepare::kGmflowQkv);

// Les noyaux lisent et écrivent les jetons par 16 octets (cp.async, uint4) : un tampon décalé dans un espace
// de travail partagé doit le rester.
bool aligned16(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

// Mémoire d'appareil possédée.
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
  Buffer& operator=(Buffer&&) = delete;
  ~Buffer() {
    if (ptr_ != nullptr) static_cast<void>(cudaFree(ptr_));
  }
  cudaError_t allocate(std::size_t bytes) {
    const cudaError_t error = cudaMalloc(&ptr_, bytes < 256 ? 256 : bytes);
    if (error != cudaSuccess) ptr_ = nullptr;
    return error;
  }
  cudaError_t upload(const void* host, std::size_t bytes) {
    if (const cudaError_t error = allocate(bytes); error != cudaSuccess) return error;
    return cudaMemcpy(ptr_, host, bytes, cudaMemcpyHostToDevice);
  }
  template <class T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }

 private:
  void* ptr_ = nullptr;
};

// Les poids [n][128] (lignes rows0.. de w, [sortie][entrée]) en FP16 dans l'ordre des fragments B du mma
// m16n8k16, celui que lisent qkv_kernel et la fusion : pas k (16 entrées), paire de tuiles n8, lane ; b0 =
// entrées 2t et 2t + 1 de la sortie 8 · tuile + g, b1 = entrées 2t + 8 et 2t + 9 (le rangement de rife_native
// et du squelette).
void pack(const std::vector<float>& w, int row0, int n, std::vector<std::uint32_t>& out) {
  const auto bits = [&](int row, int k) -> std::uint32_t {
    return __half_as_ushort(__float2half_rn(w[static_cast<std::size_t>(row0 + row) * ga::kD + k]));
  };
  for (int ks = 0; ks < ga::kSteps; ++ks)
    for (int np = 0; np < n / 16; ++np)
      for (int lane = 0; lane < 32; ++lane) {
        const int g = lane >> 2, t = lane & 3, k0 = ks * 16 + 2 * t;
        for (int j = 0; j < 2; ++j) {
          const int row = (2 * np + j) * 8 + g;
          out.push_back(bits(row, k0) | (bits(row, k0 + 1) << 16U));
          out.push_back(bits(row, k0 + 8) | (bits(row, k0 + 9) << 16U));
        }
      }
}

struct Layer {
  Buffer qkv, merge, gamma, beta;
  bool shifted = false;
  bool cross = false;
};

class Net final : public GmflowAttentionNet {
 public:
  Net(const GmflowAttention& a, int scale, int batch, int height, int width)
      : batch_(batch), tokens_(static_cast<long>(height) * width) {
    const int s = a.splits[static_cast<std::size_t>(scale)];
    grid_ = {.height = height, .width = width, .splits = s, .wh = height / s, .ww = width / s};
    grid_.length = grid_.wh * grid_.ww;
    scale_log2_ = a.scale * ga::kLog2e;
    mask_log2_ = a.mask * ga::kLog2e;
    eps_ = a.epsilon;
  }

  bool plan(const GmflowAttention& a, std::string& error) {
    layers_ = std::vector<Layer>(a.layers.size());
    for (std::size_t i = 0; i < a.layers.size(); ++i) {
      const GmflowAttentionLayer& from = a.layers[i];
      Layer& to = layers_[i];
      to.shifted = from.shifted;
      to.cross = from.cross;
      std::vector<std::uint32_t> qkv, merge;
      for (int part = 0; part < 3; ++part) pack(from.qkv, part * ga::kD, ga::kD, qkv);
      pack(from.merge, 0, ga::kD, merge);
      if (!sent(to.qkv.upload(qkv.data(), qkv.size() * 4), error) ||
          !sent(to.merge.upload(merge.data(), merge.size() * 4), error) ||
          !sent(to.gamma.upload(from.gamma.data(), from.gamma.size() * 4), error) ||
          !sent(to.beta.upload(from.beta.data(), from.beta.size() * 4), error))
        return false;
    }
    return sent(workspace_.allocate(static_cast<std::size_t>(workspace_bytes())), error);
  }

  std::int64_t workspace_bytes() const override { return std::int64_t{batch_} * tokens_ * ga::kQkv * 2; }

  cudaError_t run_layer(cudaStream_t stream, int layer, const void* query, const void* keys,
                        void* out) const override {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()) || query == nullptr || keys == nullptr ||
        out == nullptr || out == query || out == keys || !aligned16(query) || !aligned16(keys) ||
        !aligned16(out))
      return cudaErrorInvalidValue;
    const Layer& l = layers_[static_cast<std::size_t>(layer)];
    ga::QkvParams q;
    q.query = static_cast<const __half*>(query);
    q.keys = static_cast<const __half*>(keys);
    q.out = workspace_.as<__half>();
    q.w = l.qkv.as<const uint4>();
    q.tokens = tokens_;
    q.batch = batch_;
    q.key_shift = l.cross ? batch_ / 2 : 0;
    const dim3 qkv_grid(static_cast<unsigned>((tokens_ + ga::kRows - 1) / ga::kRows), 3,
                        static_cast<unsigned>(batch_));
    ga::qkv_kernel<ga::Acc::kF32><<<qkv_grid, ga::kThreads, 0, stream>>>(q);
    ga::AttnParams p;
    p.qkv = workspace_.as<const __half>();
    p.source = l.cross ? nullptr : static_cast<const __half*>(query);
    p.out = static_cast<__half*>(out);
    p.merge = l.merge.as<const uint4>();
    p.gamma = l.gamma.as<const float>();
    p.beta = l.beta.as<const float>();
    p.grid = grid_;
    p.grid.shifted = l.shifted;
    p.grid.sh = l.shifted ? grid_.wh / 2 : 0;
    p.grid.sw = l.shifted ? grid_.ww / 2 : 0;
    p.tokens = tokens_;
    p.scale_log2 = scale_log2_;
    p.mask_log2 = mask_log2_;
    p.eps = eps_;
    const dim3 grid(static_cast<unsigned>((grid_.length + ga::kRows - 1) / ga::kRows),
                    static_cast<unsigned>(grid_.splits * grid_.splits), static_cast<unsigned>(batch_));
    ga::attention_kernel<ga::Acc::kF32><<<grid, ga::kThreads, ga::kSmemBytes, stream>>>(p);
    return cudaGetLastError();
  }

 private:
  static bool sent(cudaError_t result, std::string& error) {
    if (result == cudaSuccess) return true;
    static_cast<void>(cudaGetLastError());
    error = std::string("mémoire de l'attention de GMFlow : ") + cudaGetErrorString(result);
    return false;
  }

  int batch_;
  long tokens_;
  ga::Grid grid_;
  float scale_log2_ = 0.F;
  float mask_log2_ = 0.F;
  float eps_ = 1e-5F;
  std::vector<Layer> layers_;
  Buffer workspace_;
};

// Les poids d'une projection de propagation : même rangement que la GEMM QKV.
struct Projection {
  Buffer w, bias;

  bool upload(const prepare::GmflowProjection& p, std::string& error) {
    if (p.weights.size() != static_cast<std::size_t>(ga::kD) * ga::kD || p.bias.size() != ga::kD) {
      error = "GMFlow : poids ou biais de propagation";
      return false;
    }
    std::vector<std::uint32_t> packed;
    pack(p.weights, 0, ga::kD, packed);
    cudaError_t e = w.upload(packed.data(), packed.size() * sizeof(std::uint32_t));
    if (e == cudaSuccess) e = bias.upload(p.bias.data(), p.bias.size() * sizeof(float));
    if (e == cudaSuccess) return true;
    error = std::string("GMFlow : poids de propagation : ") + cudaGetErrorString(e);
    return false;
  }

  cudaError_t project(cudaStream_t stream, const void* input, void* out, long pixels) const {
    const ga::QkvParams p{.query = static_cast<const __half*>(input),
                          .out = static_cast<__half*>(out),
                          .w = w.as<const uint4>(),
                          .tokens = pixels,
                          .batch = 2,
                          .bias = bias.as<const float>()};
    const dim3 grid(static_cast<unsigned>((pixels + ga::kRows - 1) / ga::kRows), 1, 2);
    ga::qkv_kernel<ga::Acc::kF32, true><<<grid, ga::kThreads, 0, stream>>>(p);
    return cudaGetLastError();
  }
};

class PropagationNet final : public GmflowPropagationNet {
 public:
  bool build(const prepare::GmflowPropagation& weights, std::string& error) {
    return q_.upload(weights.query, error) && k_.upload(weights.keys, error);
  }

  cudaError_t project(cudaStream_t stream, const void* features, long pixels, bool global, void* query,
                      void* keys) const override {
    if (pixels < 1 || pixels > std::numeric_limits<long>::max() / (2 * ga::kD) || features == nullptr ||
        query == nullptr || keys == nullptr || features == query || features == keys || keys == query ||
        !aligned16(features) || !aligned16(query) || !aligned16(keys))
      return cudaErrorInvalidValue;
    if (const auto e = q_.project(stream, features, query, pixels); e != cudaSuccess) return e;
    return k_.project(stream, global ? query : features, keys, pixels);
  }

 private:
  Projection q_, k_;
};

}  // namespace

std::unique_ptr<GmflowPropagationNet> GmflowPropagationNet::create(const prepare::GmflowPropagation& weights,
                                                                   std::string& error) {
  auto net = std::make_unique<PropagationNet>();
  if (!net->build(weights, error)) return nullptr;
  return net;
}

std::unique_ptr<GmflowAttentionNet> GmflowAttentionNet::create(const GmflowAttention& attention, int scale,
                                                               int batch, int height, int width,
                                                               std::string& error) {
  if (scale < 0 || scale > 1 || attention.layers.size() != static_cast<std::size_t>(prepare::kGmflowLayers)) {
    error = "attention de GMFlow : échelle ou couches";
    return nullptr;
  }
  const int s = attention.splits[static_cast<std::size_t>(scale)];
  if (batch < 2 || batch % 2 != 0 || s < 1 || height < s || width < s || height % s != 0 || width % s != 0) {
    error = "attention de GMFlow : lot pair et grille coupée en fenêtres égales attendus";
    return nullptr;
  }
  for (const GmflowAttentionLayer& l : attention.layers) {
    if (l.qkv.size() != static_cast<std::size_t>(ga::kQkv) * ga::kD ||
        l.merge.size() != static_cast<std::size_t>(ga::kD) * ga::kD || l.gamma.size() != ga::kD ||
        l.beta.size() != ga::kD) {
      error = "attention de GMFlow : poids";
      return nullptr;
    }
  }
  auto net = std::make_unique<Net>(attention, scale, batch, height, width);
  if (!net->plan(attention, error)) return nullptr;
  return net;
}

}  // namespace kirari::gpu::cuda
