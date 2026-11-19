// Le RIFE natif (rife_native_net.h) : les couches de chaque morceau planifiées une fois (tranche des canaux
// d'entrée, tuile des canaux de sortie, grille), leurs poids rangés dans l'ordre des fragments B du mma, puis
// un lancement par couche. Les activations entre deux couches alternent entre deux tampons de la zone de
// travail ; l'entrée d'un ResConv reste donc lisible pour son addition résiduelle. Une couche dont la grille
// ne remplit pas la carte découpe K ; ses sommes partielles vont dans un troisième tampon, après les deux.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/rife_native_kernels.cuh"
#include "gpu/cuda/backend/rife_native_net.h"

namespace kirari::gpu::cuda {
namespace {

namespace rn = rife_native;

constexpr int pad16(int c) {
  return (c + 15) / 16 * 16;
}
constexpr int pad8(int c) {
  return (c + 7) / 8 * 8;
}
constexpr std::int64_t align256(std::int64_t bytes) {
  return (bytes + 255) / 256 * 256;
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
  cudaError_t upload(const void* host, std::size_t bytes) {
    const cudaError_t error = cudaMalloc(&ptr_, bytes);
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

// La fin d'une couche à K découpé (rife_native::reduce_kernel), width : canaux de la grille z.
void launch_reduce(bool phase, int width, cudaStream_t stream, const rn::Params& p) {
  const long count = static_cast<long>(p.oh) * p.ow * (width / 8);
  const auto blocks = static_cast<unsigned>((count + 255) / 256);
  if (phase)
    rn::reduce_kernel<rn::Out::kPhase><<<blocks, 256, 0, stream>>>(p, width);
  else
    rn::reduce_kernel<rn::Out::kNhwc><<<blocks, 256, 0, stream>>>(p, width);
}

// Les façons de lancer une couche.
enum class Kind : std::uint8_t {
  kSame,     // 3 × 3, pas 1
  kHalf,     // 3 × 3, pas 2
  kImage,    // 3 × 3, pas 2, depuis les trois plans d'une image (première couche de l'encodeur)
  kPhase,    // transposée 4 × 4 de pas 2, une phase par blockIdx.z, sortie par pixel
  kPlanes4,  // transposée de l'encodeur, 3 × 3 aux 16 sorties phase · 4 + c, sortie en quatre plans
};

using Launch = void (*)(dim3 grid, int smem, cudaStream_t stream, const rn::Params& p);

template <int Kc, int CT, int KH, int S, int Rows, rn::In I, rn::Out O, bool Acc16>
void launch_one(dim3 grid, int smem, cudaStream_t stream, const rn::Params& p) {
  rn::conv_kernel<Kc, CT, KH, S, Rows, I, O, Acc16>
      <<<grid, rn::kThreads, static_cast<std::size_t>(smem), stream>>>(p);
}

// La tuile de 64 colonnes n'existe qu'en sommes f16 : en f32, ses 128 registres d'accumulateurs débordent.
template <int Kc, int KH, int S, int Rows, rn::In I, rn::Out O, bool Acc16>
Launch by_tile(int ct) {
  if (ct == 16) return launch_one<Kc, 16, KH, S, Rows, I, O, Acc16>;
  if (ct == 32) return launch_one<Kc, 32, KH, S, Rows, I, O, Acc16>;
  if constexpr (Acc16) {
    if (ct == 64) return launch_one<Kc, 64, KH, S, Rows, I, O, Acc16>;
  }
  return nullptr;
}

template <bool Acc16>
Launch pick(Kind kind, int kc, int ct) {
  using rn::In;
  using rn::Out;
  switch (kind) {
    case Kind::kSame:
      if (kc == 16) return by_tile<16, 3, 1, 2, In::kNhwc, Out::kNhwc, Acc16>(ct);
      if (kc == 32) return by_tile<32, 3, 1, 2, In::kNhwc, Out::kNhwc, Acc16>(ct);
      if (kc == 64) return by_tile<64, 3, 1, 2, In::kNhwc, Out::kNhwc, Acc16>(ct);
      return nullptr;
    case Kind::kHalf:
      if (kc == 16) return by_tile<16, 3, 2, 1, In::kNhwc, Out::kNhwc, Acc16>(ct);
      if (kc == 32) return by_tile<32, 3, 2, 1, In::kNhwc, Out::kNhwc, Acc16>(ct);
      return nullptr;
    case Kind::kImage:
      return kc == 16 ? by_tile<16, 3, 2, 1, In::kImage, Out::kNhwc, Acc16>(ct) : nullptr;
    case Kind::kPhase:
      if (kc == 16) return by_tile<16, 2, 1, 2, In::kNhwc, Out::kPhase, Acc16>(ct);
      if (kc == 32) return by_tile<32, 2, 1, 2, In::kNhwc, Out::kPhase, Acc16>(ct);
      if (kc == 64) return by_tile<64, 2, 1, 2, In::kNhwc, Out::kPhase, Acc16>(ct);
      return nullptr;
    case Kind::kPlanes4:
      return kc == 16 && ct == 16 ? launch_one<16, 16, 3, 1, 2, In::kNhwc, Out::kPlanes4, Acc16> : nullptr;
  }
  return nullptr;
}

struct Shape3 {
  int kh = 3;
  int s = 1;
  int rows = 2;
};

Shape3 shape_of(Kind kind) {
  switch (kind) {
    case Kind::kHalf:
    case Kind::kImage:
      return {.kh = 3, .s = 2, .rows = 1};
    case Kind::kPhase:
      return {.kh = 2, .s = 1, .rows = 2};
    default:
      return {.kh = 3, .s = 1, .rows = 2};
  }
}

// Poids FP16 [n][taps][cin] (n : les canaux d'un paquet blockIdx.z) rangés dans l'ordre des fragments B de
// mma m16n8k16 : pas k (prise, tranche de 16 entrées), paire de tuiles n8, lane (g = lane / 4, t = lane % 4)
// ; 4 mots par lane : b0 et b1 de la tuile 2·np, puis de la tuile 2·np + 1. b0 = entrées 2t et 2t + 1 de la
// sortie 8·tuile + g, b1 = entrées 2t + 8 et 2t + 9 ; la plus petite entrée dans la moitié basse du mot.
void pack(const std::vector<__half>& w, int ct, int taps, int cin, std::vector<std::uint32_t>& out) {
  const int kcs = cin / 16, nps = ct / 16;
  const auto bits = [&](int n, int tap, int c) -> std::uint32_t {
    const __half h = w[(static_cast<std::size_t>(n) * taps + tap) * cin + c];
    return __half_as_ushort(h);
  };
  for (int ks = 0; ks < taps * kcs; ++ks) {
    const int tap = ks / kcs, kc = ks % kcs;
    for (int np = 0; np < nps; ++np)
      for (int lane = 0; lane < 32; ++lane) {
        const int g = lane >> 2, t = lane & 3, k0 = kc * 16 + 2 * t;
        for (int j = 0; j < 2; ++j) {
          const int n = (2 * np + j) * 8 + g;
          out.push_back(bits(n, tap, k0) | (bits(n, tap, k0 + 1) << 16U));
          out.push_back(bits(n, tap, k0 + 8) | (bits(n, tap, k0 + 9) << 16U));
        }
      }
  }
}

struct Layer {
  Kind kind = Kind::kSame;
  Launch launch = nullptr;
  int kc = 16, ct = 16, tiles = 1, phases = 1, ksplit = 1;
  int ih = 0, iw = 0, oh = 0, ow = 0;
  int in_ld = 0, out_ld = 0, out_ch = 0, cin = 0;
  bool residual = false;
  bool activation = false;
  Buffer w, bias, slope;
  dim3 grid;
  int smem = 0;
};

// Les poids et le biais d'une couche, le facteur du résiduel replié : w[o] · s, b[o] · s ; zéros dans le
// rembourrage des canaux.
struct Folded {
  const RifeNativeConv& conv;
  float factor(int o) const { return conv.scale.empty() ? 1.F : conv.scale[static_cast<std::size_t>(o)]; }
  float weight(int o, int tap, int c) const {
    if (o >= conv.cout || c >= conv.cin) return 0.F;
    const int taps = conv.transposed ? 16 : 9;
    return conv.weights[(static_cast<std::size_t>(o) * taps + tap) * conv.cin + c] * factor(o);
  }
  float bias(int o) const {
    return o >= conv.cout ? 0.F : conv.bias[static_cast<std::size_t>(o)] * factor(o);
  }
};

class Net final : public RifeNativeNet {
 public:
  explicit Net(bool acc32) : acc32_(acc32) {
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess)
      static_cast<void>(cudaDeviceGetAttribute(&sms_, cudaDevAttrMultiProcessorCount, device));
    sms_ = std::max(sms_, 1);
  }

  // Les couches d'un morceau : entrée ih × iw à in_ld canaux par pixel (0 : l'image en plans), sortie
  // attendue out_h × out_w (celle de la transposée).
  bool plan(const RifeNativePiece& piece, int ih, int iw, int in_ld, int out_h, int out_w, bool encoder,
            std::string& error) {
    std::vector<Layer> layers(piece.size());
    for (std::size_t i = 0; i < piece.size(); ++i) {
      const RifeNativeConv& conv = piece[i];
      Layer& layer = layers[i];
      const bool last = i + 1 == piece.size();
      const bool image = encoder && i == 0;
      layer.cin = pad16(conv.cin);
      layer.in_ld = image ? 0 : in_ld;
      if (!image && layer.cin != in_ld) {
        error = "couche " + std::to_string(i) + " : " + std::to_string(conv.cin) + " canaux pour un pas de " +
                std::to_string(in_ld);
        return false;
      }
      // Le noyau de l'image échantillonne au pas 2 : une première couche d'un autre pas sortirait fausse.
      if (image && (conv.cin != 3 || conv.stride != 2 || conv.transposed)) {
        error = "première couche de l'encodeur : convolution de pas 2 sur 3 canaux attendue";
        return false;
      }
      layer.ih = ih;
      layer.iw = iw;
      if (conv.transposed) {
        layer.kind = encoder ? Kind::kPlanes4 : Kind::kPhase;
        layer.oh = ih;
        layer.ow = iw;
      } else {
        layer.kind = image ? Kind::kImage : conv.stride == 2 ? Kind::kHalf : Kind::kSame;
        layer.oh = conv.stride == 2 ? (ih - 1) / 2 + 1 : ih;
        layer.ow = conv.stride == 2 ? (iw - 1) / 2 + 1 : iw;
      }
      if (!build(conv, layer, last, error)) return false;
      if (last && (2 * layer.oh != out_h || 2 * layer.ow != out_w)) {
        error = "taille de sortie de la transposée";
        return false;
      }
      ih = layer.oh;
      iw = layer.ow;
      in_ld = layer.out_ld;
      if (!last) peak_ = std::max(peak_, align256(std::int64_t{layer.oh} * layer.ow * layer.out_ld * 2));
    }
    pieces_.push_back(std::move(layers));
    return true;
  }

  std::int64_t workspace_bytes() const override { return 2 * peak_ + partial_; }
  bool acc32() const override { return acc32_; }

  cudaError_t encode(cudaStream_t stream, const void* image, void* encoded, void* workspace) const override {
    return run(stream, 0, image, encoded, workspace);
  }
  cudaError_t block(cudaStream_t stream, int k, const void* x, void* y, void* workspace) const override {
    return run(stream, static_cast<std::size_t>(k) + 1, x, y, workspace);
  }

 private:
  // Tranche des canaux d'entrée, tuile des canaux de sortie, poids rangés, lancement.
  bool build(const RifeNativeConv& conv, Layer& layer, bool last, std::string& error) {
    const auto sent = [&](cudaError_t result) {
      if (result == cudaSuccess) return true;
      static_cast<void>(cudaGetLastError());
      error = std::string("poids du RIFE natif : ") + cudaGetErrorString(result);
      return false;
    };
    const Folded folded{conv};
    std::vector<std::uint32_t> packed;
    std::vector<float> biases;
    layer.residual = !conv.scale.empty();
    layer.activation = !conv.slope.empty();
    if (layer.kind == Kind::kPlanes4) {
      if (conv.cout != 4) {
        error = "transposée de l'encodeur : 4 canaux attendus";
        return false;
      }
      encoder_tail(folded, layer, packed, biases);
    } else {
      plain(folded, layer, last, packed, biases);
    }
    if (layer.activation) {
      std::vector<float> slope(static_cast<std::size_t>(pad16(conv.cout)), 0.F);
      std::copy(conv.slope.begin(), conv.slope.end(), slope.begin());
      if (!sent(layer.slope.upload(slope.data(), slope.size() * sizeof(float)))) return false;
    }
    if (!sent(layer.w.upload(packed.data(), packed.size() * sizeof(std::uint32_t))) ||
        !sent(layer.bias.upload(biases.data(), biases.size() * sizeof(float))))
      return false;
    layer.launch =
        acc32_ ? pick<false>(layer.kind, layer.kc, layer.ct) : pick<true>(layer.kind, layer.kc, layer.ct);
    if (layer.launch == nullptr) {
      error = "aucun noyau pour cette couche";
      return false;
    }
    const Shape3 shape = shape_of(layer.kind);
    const int th = rn::kWarps * shape.rows;
    layer.grid = dim3(static_cast<unsigned>((layer.ow + rn::kTW - 1) / rn::kTW),
                      static_cast<unsigned>((layer.oh + th - 1) / th),
                      static_cast<unsigned>(layer.phases * layer.tiles));
    layer.smem = rn::smem_bytes(layer.kc, layer.ct, shape.kh, shape.s, shape.rows);
    layer.ksplit = split_for(layer, shape.kh * shape.kh);
    if (layer.ksplit > 1) {
      partial_ = std::max(
          partial_, align256(std::int64_t{layer.ksplit} * layer.oh * layer.ow * layer.grid.z * layer.ct * 4));
      layer.grid.z *= static_cast<unsigned>(layer.ksplit);
    }
    return true;
  }

  // Les parts de K d'une couche dont la grille fait moins de deux vagues (les blocs grossiers, 30 × 17
  // pixels au 1/32 de 1080p) : assez pour deux vagues, au plus une par prise et tranche. Ni l'encodeur ni
  // une couche lue dans l'image : pleine taille, la grille remplit déjà la carte.
  int split_for(const Layer& layer, int taps) const {
    if (layer.kind == Kind::kImage || layer.kind == Kind::kPlanes4) return 1;
    const long blocks = static_cast<long>(layer.grid.x) * layer.grid.y * layer.grid.z;
    const long want = 2L * sms_;
    if (blocks >= want) return 1;
    const long units = static_cast<long>(layer.cin / layer.kc) * taps;
    return static_cast<int>(std::min((want + blocks - 1) / blocks, units));
  }

  // La transposée de l'encodeur : 16 sorties j = phase · 4 + c sur une fenêtre 3 × 3 de l'entrée. La prise
  // (dy, dx) sert la phase (py, px) si dy ∈ {py - 1, py}, au poids (3 - py - 2 ty, 3 - px - 2 tx),
  // ty = dy - py + 1.
  static void encoder_tail(const Folded& f, Layer& layer, std::vector<std::uint32_t>& packed,
                           std::vector<float>& biases) {
    const int cin = layer.cin;
    layer.kc = 16;
    layer.ct = 16;
    std::vector<__half> w(static_cast<std::size_t>(16) * 9 * cin);
    for (int j = 0; j < 16; ++j)
      for (int tap = 0; tap < 9; ++tap) {
        const int py = j >> 3, px = (j >> 2) & 1;
        const int ty = tap / 3 - py, tx = tap % 3 - px;
        const bool used = ty >= 0 && ty <= 1 && tx >= 0 && tx <= 1;
        for (int c = 0; c < cin; ++c) {
          const float v = used ? f.weight(j & 3, (3 - py - 2 * ty) * 4 + (3 - px - 2 * tx), c) : 0.F;
          w[(static_cast<std::size_t>(j) * 9 + tap) * cin + c] = __float2half_rn(v);
        }
      }
    pack(w, 16, 9, cin, packed);
    for (int j = 0; j < 16; ++j) biases.push_back(f.bias(j & 3));
    layer.out_ld = 0;
    layer.out_ch = 4;
  }

  // Une convolution 3 × 3, ou une transposée de bloc en quatre phases 2 × 2 (prises (ty, tx) aux décalages
  // py - 1 + ty, poids (3 - py - 2 ty, 3 - px - 2 tx)). Un paquet de poids par phase et par tuile.
  void plain(const Folded& f, Layer& layer, bool last, std::vector<std::uint32_t>& packed,
             std::vector<float>& biases) const {
    const int cin = layer.cin;
    const int cout = pad16(f.conv.cout);
    const bool phase = layer.kind == Kind::kPhase;
    const int taps = phase ? 4 : 9;
    layer.phases = phase ? 4 : 1;
    const bool half = layer.kind == Kind::kHalf || layer.kind == Kind::kImage;
    layer.kc = layer.kind == Kind::kImage ? 16 : (cin % 64 == 0 && !half) ? 64 : cin % 32 == 0 ? 32 : 16;
    layer.ct = choose_tile(layer, cout);
    layer.tiles = cout / layer.ct;
    layer.out_ld = last ? pad8(f.conv.cout) : cout;
    layer.out_ch = layer.out_ld;
    for (int ph = 0; ph < layer.phases; ++ph) {
      const int py = ph >> 1, px = ph & 1;
      for (int tile = 0; tile < layer.tiles; ++tile) {
        std::vector<__half> w(static_cast<std::size_t>(layer.ct) * taps * cin);
        for (int n = 0; n < layer.ct; ++n)
          for (int tap = 0; tap < taps; ++tap) {
            const int from = phase ? (3 - py - 2 * (tap >> 1)) * 4 + (3 - px - 2 * (tap & 1)) : tap;
            for (int c = 0; c < cin; ++c)
              w[(static_cast<std::size_t>(n) * taps + tap) * cin + c] =
                  __float2half_rn(f.weight(tile * layer.ct + n, from, c));
          }
        pack(w, layer.ct, taps, cin, packed);
      }
      for (int o = 0; o < cout; ++o) biases.push_back(f.bias(o));
    }
  }

  // La plus large tuile de sortie (64, ou 32 en f32) qui laisse assez de blocs pour remplir la carte (deux
  // vagues) ; les petits blocs grossiers de RIFE (30 × 18 pixels au 1/32) prennent des tuiles plus étroites.
  int choose_tile(const Layer& layer, int cout) const {
    const Shape3 shape = shape_of(layer.kind);
    const long cells = static_cast<long>((layer.ow + rn::kTW - 1) / rn::kTW) *
                       ((layer.oh + rn::kWarps * shape.rows - 1) / (rn::kWarps * shape.rows)) * layer.phases;
    for (const int ct : {64, 32}) {
      if ((acc32_ && ct == 64) || cout % ct != 0) continue;
      if (cells * (cout / ct) >= 2L * sms_) return ct;
    }
    return 16;
  }

  cudaError_t run(cudaStream_t stream, std::size_t piece, const void* in, void* out, void* workspace) const {
    if (piece >= pieces_.size()) return cudaErrorInvalidValue;
    const std::vector<Layer>& layers = pieces_[piece];
    auto* ws = static_cast<std::uint8_t*>(workspace);
    __half* const buffers[2] = {reinterpret_cast<__half*>(ws), reinterpret_cast<__half*>(ws + peak_)};
    auto* const partial = reinterpret_cast<float*>(ws + 2 * peak_);
    const __half* from = static_cast<const __half*>(in);
    for (std::size_t i = 0; i < layers.size(); ++i) {
      const Layer& layer = layers[i];
      const bool last = i + 1 == layers.size();
      __half* to = last ? static_cast<__half*>(out) : buffers[i % 2];
      rn::Params p;
      p.in = from;
      p.in_ld = layer.in_ld;
      p.in_plane = static_cast<long>(layer.ih) * layer.iw;
      p.ih = layer.ih;
      p.iw = layer.iw;
      p.out = to;
      p.out_ld = layer.out_ld;
      p.out_ch = layer.out_ch;
      p.out_plane = 4L * layer.oh * layer.ow;
      p.oh = layer.oh;
      p.ow = layer.ow;
      p.w = layer.w.as<uint4>();
      p.bias = layer.bias.as<float>();
      p.slope = layer.activation ? layer.slope.as<float>() : nullptr;
      p.skip = layer.residual ? from : nullptr;
      p.skip_ld = layer.in_ld;
      p.cin = layer.cin;
      p.tiles = layer.tiles;
      p.ksplit = layer.ksplit;
      p.partial = layer.ksplit > 1 ? partial : nullptr;
      layer.launch(layer.grid, layer.smem, stream, p);
      if (layer.ksplit > 1)
        launch_reduce(layer.kind == Kind::kPhase, layer.phases * layer.tiles * layer.ct, stream, p);
      from = to;
    }
    return cudaGetLastError();
  }

  bool acc32_;
  int sms_ = 1;
  std::int64_t peak_ = 0;
  std::int64_t partial_ = 0;  // sommes partielles de la plus grande couche à K découpé
  std::vector<std::vector<Layer>> pieces_;
};

}  // namespace

std::unique_ptr<RifeNativeNet> RifeNativeNet::create(const std::vector<RifeNativePiece>& pieces,
                                                     const std::vector<int>& reduce, int height, int width,
                                                     bool acc32, std::string& error) {
  if (pieces.size() != reduce.size() + 1 || pieces.empty()) {
    error = "un encodeur et un morceau par bloc attendus";
    return nullptr;
  }
  auto net = std::make_unique<Net>(acc32);
  if (!net->plan(pieces[0], height, width, 0, height, width, true, error)) {
    error = "encodeur : " + error;
    return nullptr;
  }
  for (std::size_t k = 0; k < reduce.size(); ++k) {
    const int d = reduce[k];
    const int channels = k == 0 ? 15 : 28;
    if (!net->plan(pieces[k + 1], height / d, width / d, pad8(channels), height / (2 * d), width / (2 * d),
                   false, error)) {
      error = "bloc " + std::to_string(k) + " : " + error;
      return nullptr;
    }
  }
  return net;
}

}  // namespace kirari::gpu::cuda
