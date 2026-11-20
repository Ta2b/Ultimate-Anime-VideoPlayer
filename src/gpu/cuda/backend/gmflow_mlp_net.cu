// Le MLP fondu de GMFlow natif (gmflow_mlp_net.h) : les poids de chaque couche rangés une fois en un flux
// d'étages pour l'anneau du noyau (gmflow_mlp_kernels.cuh), et chaque lancement découpé selon sa grille : la
// couche cachée se partage entre plusieurs blocs (splits) quand les blocs de jetons ne remplissent pas la
// carte.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/gmflow_mlp_kernels.cuh"
#include "gpu/cuda/backend/gmflow_mlp_net.h"

namespace kirari::gpu::cuda {
namespace {

namespace gm = gmflow_mlp;
using prepare::GmflowMlp;
using prepare::GmflowMlpLayer;

static_assert(gm::kChannels == prepare::kGmflowMlpChannels &&
              gm::kHiddenTile == prepare::kGmflowMlpHiddenTile);

constexpr int kMaxSplits = 8;

// Le noyau lit et écrit les jetons par 16 octets (cp.async, uint4) : un tampon décalé dans un espace de
// travail partagé doit le rester.
bool aligned16(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}
constexpr int kDefaultSmem = 48 * 1024;

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
    const cudaError_t error = cudaMalloc(&ptr_, std::max<std::size_t>(bytes, 256));
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

// Le fragment B d'une lane pour la paire de tuiles n8 [n0, n0 + 16) au pas k [k0, k0 + 16), w FP16 [n][ld]
// (sortie n, entrée k) : b0 = (k 2t, 2t + 1) et b1 = (k 2t + 8, 2t + 9) de la sortie n0 + g, puis de la
// sortie n0 + 8 + g.
uint4 fragment(const std::vector<__half>& w, int ld, int n0, int k0, int lane) {
  const int g = lane >> 2, t = lane & 3;
  const auto bits = [&](int n, int k) -> std::uint32_t {
    return __half_as_ushort(
        w[static_cast<std::size_t>(n) * static_cast<std::size_t>(ld) + static_cast<std::size_t>(k)]);
  };
  std::uint32_t v[4];
  for (int j = 0; j < 2; ++j) {
    const int n = n0 + 8 * j + g;
    v[2 * j] = bits(n, k0 + 2 * t) | (bits(n, k0 + 2 * t + 1) << 16U);
    v[2 * j + 1] = bits(n, k0 + 2 * t + 8) | (bits(n, k0 + 2 * t + 9) << 16U);
  }
  return make_uint4(v[0], v[1], v[2], v[3]);
}

std::vector<__half> to_half(const std::vector<float>& from) {
  std::vector<__half> out(from.size());
  std::ranges::transform(from, out.begin(), [](float v) { return __float2half_rn(v); });
  return out;
}

// Le flux d'étages d'une couche, dans l'ordre où le noyau les prend : pour chaque tranche c de 64 canaux
// cachés, 8 étages de W1 (pas k 2i et 2i + 1, 4 paires de tuiles, 32 lanes) puis 4 étages de W2 (pas k j de
// la tranche, 8 paires, 32 lanes).
std::vector<uint4> stages(const GmflowMlpLayer& layer, int hidden, bool paired) {
  const std::vector<__half> w1 = to_half(layer.w1);  // [hidden][256]
  const std::vector<__half> w2 = to_half(layer.w2);  // [128][hidden]
  std::vector<uint4> out;
  out.reserve(static_cast<std::size_t>(hidden / gm::kHiddenTile) * gm::kChunkStages * gm::kStageVec);
  const int group = paired ? 2 : 1;
  for (int c = 0; c < hidden / gm::kHiddenTile; c += group) {
    // Dans un groupe de deux, les étages W1 au même K sont adjacents ; W2 garde son ordre.
    for (int ks = 0; ks < gm::kK1; ks += 2)
      for (int part = 0; part < group; ++part)
        for (int kl = 0; kl < 2; ++kl)
          for (int np = 0; np < gm::kNP1; ++np)
            for (int lane = 0; lane < 32; ++lane)
              out.push_back(
                  fragment(w1, gm::kIn, (c + part) * gm::kHiddenTile + 16 * np, 16 * (ks + kl), lane));
    for (int part = 0; part < group; ++part)
      for (int j = 0; j < gm::kK2; ++j)
        for (int np = 0; np < gm::kNP2; ++np)
          for (int lane = 0; lane < 32; ++lane)
            out.push_back(fragment(w2, hidden, 16 * np, (c + part) * gm::kHiddenTile + 16 * j, lane));
  }
  return out;
}

template <int Warps, bool Paired = false, gm::Acc A = gm::Acc::kF32>
void launch(dim3 grid, cudaStream_t stream, const gm::Params& p) {
  constexpr auto smem = Paired ? gm::Block<Warps>::kWideSmemBytes : gm::Block<Warps>::kSmemBytes;
  gm::mlp_kernel<Warps, A, Paired><<<grid, gm::Block<Warps>::kThreads, smem, stream>>>(p);
}

// Le noyau du bloc demandé, réglé pour sa mémoire partagée (au-delà de 48 Ko, et la part de la mémoire
// partagée au plus haut pour deux blocs par SM).
template <int Warps, bool Paired = false, gm::Acc A = gm::Acc::kF32>
cudaError_t prepare_kernel() {
  const auto kernel = gm::mlp_kernel<Warps, A, Paired>;
  constexpr auto smem = Paired ? gm::Block<Warps>::kWideSmemBytes : gm::Block<Warps>::kSmemBytes;
  if (const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      e != cudaSuccess)
    return e;
  return cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                              cudaSharedmemCarveoutMaxShared);
}

struct Layer {
  Buffer w, gamma, beta;
  float eps = 1e-5F;
};

class Net final : public GmflowMlpNet {
 public:
  Net(GmflowMlpBlock block, long max_tokens) : warps_(static_cast<int>(block)), max_tokens_(max_tokens) {}

  bool build(const GmflowMlp& mlp, std::string& error, bool fast) {
    if (!fits(mlp)) {
      error = "MLP de GMFlow : 128 canaux, couche cachée par 64 et poids à leur taille attendus";
      return false;
    }
    hidden_ = mlp.hidden;
    int device = 0, optin = 0, major = 0, minor = 0;
    if (!sent(cudaGetDevice(&device), error) ||
        !sent(cudaDeviceGetAttribute(&sms_, cudaDevAttrMultiProcessorCount, device), error) ||
        !sent(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device), error) ||
        !sent(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), error) ||
        !sent(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), error))
      return false;
    // Seule la forme mesurée sur Ada et dont chaque split peut grouper les tranches par deux.
    const char* ab = std::getenv("KIRARI_AB_GMFLOW_MLP_PAIRS");  // env: banc
    paired_ = major == 8 && minor == 9 && warps_ == 4 && hidden_ % (2 * kMaxSplits * gm::kHiddenTile) == 0 &&
              (ab == nullptr || std::strcmp(ab, "0") != 0);
    const int smem = paired_ ? gm::Block<4>::kWideSmemBytes
                             : (warps_ == 8 ? gm::Block<8>::kSmemBytes : gm::Block<4>::kSmemBytes);
    if (smem > std::max(optin, kDefaultSmem)) {
      error = "MLP de GMFlow : la carte n'a pas la mémoire partagée de ce bloc";
      return false;
    }
    fast_ = fast && paired_;
    if (fast && !fast_) {
      error = "MLP de GMFlow rapide : forme ou GPU incompatibles";
      return false;
    }
    const cudaError_t prepared =
        paired_ ? prepare_kernel<4, true>() : (warps_ == 8 ? prepare_kernel<8>() : prepare_kernel<4>());
    if (!sent(prepared, error)) return false;
    if (fast_ && !sent(prepare_kernel<4, true, gm::Acc::kF16>(), error)) return false;
    if (!upload_layers(mlp, error)) return false;
    partial_floats_ = partial_floats();
    return sent(partial_.allocate(static_cast<std::size_t>(partial_floats_) * sizeof(float)), error);
  }

  std::int64_t workspace_bytes() const override { return partial_floats_ * 4; }

  cudaError_t run(cudaStream_t stream, int layer, const void* x, const void* message, void* out,
                  long tokens) const override {
    if (layer < 0 || layer >= static_cast<int>(layers_.size()) || x == nullptr || message == nullptr ||
        out == nullptr || tokens < 1 || tokens > max_tokens_ || !aligned16(x) || !aligned16(message) ||
        !aligned16(out))
      return cudaErrorInvalidValue;
    const Layer& l = layers_[static_cast<std::size_t>(layer)];
    const long blocks = (tokens + tokens_per_block() - 1) / tokens_per_block();
    gm::Params p;
    p.x = static_cast<const __half*>(x);
    p.message = static_cast<const __half*>(message);
    p.out = static_cast<__half*>(out);
    p.tokens = tokens;
    p.hidden = hidden_;
    const bool fast = fast_ && layer < 5;
    p.splits = fast ? 1 : splits_for(blocks);
    p.partial = p.splits > 1 ? partial_.as<float>() : nullptr;
    p.w = l.w.as<const uint4>();
    p.gamma = l.gamma.as<const float>();
    p.beta = l.beta.as<const float>();
    p.eps = l.eps;
    const dim3 grid(static_cast<unsigned>(blocks), static_cast<unsigned>(p.splits));
    if (fast) {
      launch<4, true, gm::Acc::kF16>(grid, stream, p);
    } else if (paired_) {
      launch<4, true>(grid, stream, p);
    } else if (warps_ == 8) {
      launch<8>(grid, stream, p);
    } else {
      launch<4>(grid, stream, p);
    }
    if (p.splits > 1) gm::finish_kernel<<<static_cast<unsigned>((tokens + 7) / 8), 256, 0, stream>>>(p);
    return cudaGetLastError();
  }

 private:
  bool upload_layers(const GmflowMlp& mlp, std::string& error) {
    layers_ = std::vector<Layer>(mlp.layers.size());
    for (std::size_t i = 0; i < mlp.layers.size(); ++i) {
      const GmflowMlpLayer& from = mlp.layers[i];
      const std::vector<uint4> packed = stages(from, hidden_, paired_);
      Layer& l = layers_[i];
      l.eps = from.epsilon;
      if (!sent(l.w.upload(packed.data(), packed.size() * sizeof(uint4)), error) ||
          !sent(l.gamma.upload(from.gamma.data(), from.gamma.size() * sizeof(float)), error) ||
          !sent(l.beta.upload(from.beta.data(), from.beta.size() * sizeof(float)), error))
        return false;
    }
    return true;
  }

  static bool fits(const GmflowMlp& mlp) {
    const auto c = static_cast<std::size_t>(mlp.channels), h = static_cast<std::size_t>(mlp.hidden);
    if (mlp.channels != gm::kChannels || mlp.hidden < gm::kHiddenTile || mlp.hidden % gm::kHiddenTile != 0 ||
        mlp.layers.empty())
      return false;
    return std::ranges::all_of(mlp.layers, [&](const GmflowMlpLayer& l) {
      return l.w1.size() == h * 2 * c && l.w2.size() == c * h && l.gamma.size() == c && l.beta.size() == c;
    });
  }

  static bool sent(cudaError_t result, std::string& error) {
    if (result == cudaSuccess) return true;
    static_cast<void>(cudaGetLastError());
    error = std::string("MLP de GMFlow : ") + cudaGetErrorString(result);
    return false;
  }

  long tokens_per_block() const { return static_cast<long>(warps_) * gm::kRows; }
  // Les blocs qui tournent ensemble sur la carte.
  long resident() const { return static_cast<long>(sms_) * (warps_ == 8 ? 1 : 2); }

  // La couche cachée découpée en assez de parts pour remplir la carte une fois (8 au plus, une tranche par
  // part au moins).
  int splits_for(long blocks) const {
    int s = 1;
    while (s < kMaxSplits && 2 * s <= hidden_ / gm::kHiddenTile && blocks * s < resident()) s *= 2;
    return s;
  }

  // La place des parts : s > 1 seulement si blocks · s / 2 < resident, donc tokens · s < 2 · resident ·
  // jetons par bloc ; et s ≤ 8.
  std::int64_t partial_floats() const {
    const std::int64_t most = std::min<std::int64_t>(2 * std::int64_t{resident()} * tokens_per_block(),
                                                     std::int64_t{kMaxSplits} * max_tokens_);
    return most * gm::kChannels;
  }

  int warps_;
  long max_tokens_;
  int hidden_ = 0;
  int sms_ = 1;
  bool paired_ = false;
  bool fast_ = false;
  std::vector<Layer> layers_;
  std::int64_t partial_floats_ = 0;
  Buffer partial_;
};

}  // namespace

std::unique_ptr<GmflowMlpNet> GmflowMlpNet::create(const prepare::GmflowMlp& mlp, long max_tokens,
                                                   GmflowMlpBlock block, std::string& error, bool fast) {
  if (max_tokens < 1) {
    error = "MLP de GMFlow : au moins un jeton";
    return nullptr;
  }
  auto net = std::make_unique<Net>(block, max_tokens);
  if (!net->build(mlp, error, fast)) return nullptr;
  return net;
}

}  // namespace kirari::gpu::cuda
