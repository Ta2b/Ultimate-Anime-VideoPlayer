// TemporalFix natif (tfix_native_net.h) : les couches de chaque morceau planifiées une fois (tranche des
// canaux d'entrée, tuile des canaux de sortie, K découpé, grille), leurs poids rangés dans l'ordre des
// fragments B du mma (rife_native::pack), puis un lancement par couche. Les six appuis d'un bloc passent en
// un lot (Params::batch). Les activations entre deux couches alternent entre deux tampons de la zone de
// travail ; l'entrée d'un ResConv reste donc lisible pour son addition résiduelle. Les sommes partielles
// d'une couche à K découpé vont dans un troisième tampon, après les deux.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/rife_native_kernels.cuh"
#include "gpu/cuda/backend/tfix_fused.cuh"
#include "gpu/cuda/backend/tfix_native_net.h"
#include "kernels/spec/tfix_fused.h"

namespace kirari::gpu::cuda {
namespace {

namespace rn = rife_native;

constexpr int pad16(int c) {
  return (c + 15) / 16 * 16;
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

// Les façons de lancer une couche : 3 × 3 de pas 1, de pas 2, ou de pas 2 depuis les trois plans d'une image
// (la première couche de l'encodeur).
enum class Kind : std::uint8_t { kSame, kHalf, kImage };

using Launch = void (*)(dim3 grid, int smem, cudaStream_t stream, const rn::Params& p);

template <int Kc, int CT, int S, int Rows, rn::In I, bool Acc16>
void launch_one(dim3 grid, int smem, cudaStream_t stream, const rn::Params& p) {
  rn::conv_kernel<Kc, CT, 3, S, Rows, I, rn::Out::kNhwc, Acc16>
      <<<grid, rn::kThreads, static_cast<std::size_t>(smem), stream>>>(p);
}

// La tuile de 64 colonnes n'existe qu'en sommes f16 : en f32, ses 128 registres d'accumulateurs débordent.
template <int Kc, int S, int Rows, rn::In I, bool Acc16>
Launch by_tile(int ct) {
  if (ct == 16) return launch_one<Kc, 16, S, Rows, I, Acc16>;
  if (ct == 32) return launch_one<Kc, 32, S, Rows, I, Acc16>;
  if constexpr (Acc16) {
    if (ct == 64) return launch_one<Kc, 64, S, Rows, I, Acc16>;
  }
  return nullptr;
}

template <bool Acc16>
Launch pick(Kind kind, int kc, int ct) {
  using rn::In;
  switch (kind) {
    case Kind::kSame:
      if (kc == 16) return by_tile<16, 1, 2, In::kNhwc, Acc16>(ct);
      if (kc == 32) return by_tile<32, 1, 2, In::kNhwc, Acc16>(ct);
      if (kc == 64) return by_tile<64, 1, 2, In::kNhwc, Acc16>(ct);
      return nullptr;
    case Kind::kHalf:
      if (kc == 16) return by_tile<16, 2, 1, In::kNhwc, Acc16>(ct);
      if (kc == 32) return by_tile<32, 2, 1, In::kNhwc, Acc16>(ct);
      return nullptr;
    case Kind::kImage:
      return kc == 16 ? by_tile<16, 2, 1, In::kImage, Acc16>(ct) : nullptr;
  }
  return nullptr;
}

// Lignes par warp : 1 en pas 2 (tuile d'entrée de 8 × 32 sorties sous 48 Ko), 2 en pas 1.
int rows_of(Kind kind) {
  return kind == Kind::kSame ? 2 : 1;
}
int stride_of(Kind kind) {
  return kind == Kind::kSame ? 1 : 2;
}

// La fin d'une couche à K découpé (rife_native::reduce_kernel), une image du lot par blockIdx.y ; width :
// canaux de la grille z d'une image.
void launch_reduce(int width, int batch, cudaStream_t stream, const rn::Params& p) {
  const long count = static_cast<long>(p.oh) * p.ow * (width / 8);
  const dim3 grid(static_cast<unsigned>((count + 255) / 256), static_cast<unsigned>(batch));
  rn::reduce_kernel<rn::Out::kNhwc><<<grid, 256, 0, stream>>>(p, width);
}

// Les traits de l'encodeur : sa dernière convolution (par pixel au pas 8, 4 canaux, ih × iw) agrandie à
// h × w, Resize linéaire demi-pixel du graphe, en quatre plans.
__global__ void traits_kernel(const __half* in, int ih, int iw, __half* out, int h, int w) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) return;
  const tfix::Axis ax = tfix::resize_axis(x, iw, w);
  const tfix::Axis ay = tfix::resize_axis(y, ih, h);
  const tfix::View v{const_cast<__half*>(in), 1, 8};  // NOLINT(cppcoreguidelines-pro-type-const-cast) lu
  const long long plane = static_cast<long long>(h) * w;
  for (int c = 0; c < kernels::kTfixTraits; ++c)
    out[c * plane + static_cast<long long>(y) * w + x] = __float2half_rn(tfix::resized(v, c, iw, ax, ay));
}

struct Layer {
  Kind kind = Kind::kSame;
  Launch launch = nullptr;
  int kc = 16, ct = 16, tiles = 1, ksplit = 1;
  int ih = 0, iw = 0, oh = 0, ow = 0;
  int in_ld = 0, out_ld = 0, cin = 0;
  bool residual = false;
  bool activation = false;
  Buffer w, bias, slope;
  dim3 grid;
  int smem = 0;
};

// Les couches d'un morceau et sa zone de travail : deux tampons de peak octets, puis les sommes partielles.
struct Piece {
  std::vector<Layer> layers;
  int batch = 1;
  std::int64_t peak = 0;
  std::int64_t partial = 0;
  std::int64_t bytes() const { return 2 * peak + partial; }
};

// Les poids et le biais d'une couche, le facteur du résiduel replié : w[o] · s, b[o] · s ; zéros dans le
// rembourrage des canaux.
struct Folded {
  const RifeNativeConv& conv;
  float factor(int o) const { return conv.scale.empty() ? 1.F : conv.scale[static_cast<std::size_t>(o)]; }
  float weight(int o, int tap, int c) const {
    if (o >= conv.cout || c >= conv.cin) return 0.F;
    return conv.weights[(static_cast<std::size_t>(o) * 9 + tap) * conv.cin + c] * factor(o);
  }
  float bias(int o) const {
    return o >= conv.cout ? 0.F : conv.bias[static_cast<std::size_t>(o)] * factor(o);
  }
};

class Net final : public TfixNativeNet {
 public:
  explicit Net(bool acc32) : acc32_(acc32) {
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess)
      static_cast<void>(cudaDeviceGetAttribute(&sms_, cudaDevAttrMultiProcessorCount, device));
    sms_ = std::max(sms_, 1);
  }

  // Les couches d'un morceau : entrée ih × iw au pas in_ld (0 : l'image en plans), lot de batch images,
  // sortie attendue out_h × out_w à out_channels canaux.
  bool plan(const RifeNativePiece& piece, int ih, int iw, int in_ld, int batch, int out_h, int out_w,
            int out_channels, std::string& error) {
    Piece planned;
    planned.batch = batch;
    if (piece.empty()) {
      error = "morceau vide";
      return false;
    }
    planned.layers = std::vector<Layer>(piece.size());
    for (std::size_t i = 0; i < piece.size(); ++i) {
      const RifeNativeConv& conv = piece[i];
      Layer& layer = planned.layers[i];
      const bool last = i + 1 == piece.size();
      const bool image = in_ld == 0 && i == 0;
      if (conv.transposed || (conv.stride != 1 && conv.stride != 2)) {
        error = "couche " + std::to_string(i) + " : ni 3 × 3 de pas 1 ni de pas 2";
        return false;
      }
      layer.cin = pad16(conv.cin);
      layer.in_ld = image ? 0 : in_ld;
      if (image ? conv.cin != 3 || conv.stride != 2 : layer.cin != in_ld) {
        error = "couche " + std::to_string(i) + " : " + std::to_string(conv.cin) + " canaux pour un pas de " +
                std::to_string(in_ld);
        return false;
      }
      layer.kind = image ? Kind::kImage : conv.stride == 2 ? Kind::kHalf : Kind::kSame;
      layer.ih = ih;
      layer.iw = iw;
      layer.oh = conv.stride == 2 ? (ih - 1) / 2 + 1 : ih;
      layer.ow = conv.stride == 2 ? (iw - 1) / 2 + 1 : iw;
      if (!build(conv, layer, last, batch, planned.partial, error)) return false;
      ih = layer.oh;
      iw = layer.ow;
      in_ld = layer.out_ld;
      const std::int64_t bytes = std::int64_t{batch} * layer.oh * layer.ow * layer.out_ld * 2;
      if (!last) planned.peak = std::max(planned.peak, align256(bytes));
      if (last && (layer.oh != out_h || layer.ow != out_w || conv.cout != out_channels)) {
        error = "taille ou canaux de la sortie";
        return false;
      }
    }
    pieces_.push_back(std::move(planned));
    return true;
  }
  // La place d'un morceau absent (l'encodeur d'une fenêtre) : run le refuse.
  void skip() { pieces_.emplace_back(); }

  // L'encodeur : sa zone, puis la sortie de sa dernière couche (h × w au pas 8) avant son agrandissement.
  std::int64_t encoder_bytes() const {
    if (pieces_.empty() || pieces_[0].layers.empty()) return 0;
    const Layer& last = pieces_[0].layers.back();
    return pieces_[0].bytes() + align256(std::int64_t{last.oh} * last.ow * last.out_ld * 2);
  }
  std::int64_t encoder_workspace_bytes() const override { return encoder_bytes(); }
  std::int64_t workspace_bytes() const override {
    std::int64_t most = encoder_bytes();
    for (std::size_t i = 1; i < pieces_.size(); ++i) most = std::max(most, pieces_[i].bytes());
    return most;
  }
  bool acc32() const override { return acc32_; }

  cudaError_t encode(cudaStream_t stream, const void* image, void* traits, void* workspace) const override {
    if (pieces_.empty() || pieces_[0].layers.empty()) return cudaErrorInvalidValue;
    const Layer& last = pieces_[0].layers.back();
    auto* low = reinterpret_cast<__half*>(static_cast<std::uint8_t*>(workspace) + pieces_[0].bytes());
    if (cudaError_t e = run(stream, 0, image, low, workspace); e != cudaSuccess) return e;
    const int h = pieces_[0].layers.front().ih;
    const int w = pieces_[0].layers.front().iw;
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned>((w + 15) / 16), static_cast<unsigned>((h + 15) / 16));
    traits_kernel<<<grid, block, 0, stream>>>(low, last.oh, last.ow, static_cast<__half*>(traits), h, w);
    return cudaGetLastError();
  }
  cudaError_t window(cudaStream_t stream, int k, const void* x, void* y, void* workspace) const override {
    if (k < 0 || k > kernels::kTfixLevels) return cudaErrorInvalidValue;
    return run(stream, static_cast<std::size_t>(k) + 1, x, y, workspace);
  }

 private:
  // Tranche des canaux d'entrée, tuile des canaux de sortie, poids rangés, lancement, K découpé.
  bool build(const RifeNativeConv& conv, Layer& layer, bool last, int batch, std::int64_t& partial,
             std::string& error) {
    const auto sent = [&](cudaError_t result) {
      if (result == cudaSuccess) return true;
      static_cast<void>(cudaGetLastError());
      error = std::string("poids de TemporalFix natif : ") + cudaGetErrorString(result);
      return false;
    };
    const Folded folded{conv};
    layer.residual = !conv.scale.empty();
    layer.activation = !conv.slope.empty();
    const int cin = layer.cin;
    const int cout = pad16(conv.cout);
    const bool half = layer.kind != Kind::kSame;
    layer.kc = layer.kind == Kind::kImage ? 16 : (cin % 64 == 0 && !half) ? 64 : cin % 32 == 0 ? 32 : 16;
    layer.ct = choose_tile(layer, cout, batch);
    layer.tiles = cout / layer.ct;
    // La sortie du morceau au pas de 8 (kernels::tfix_out_ld), celle d'une couche intérieure au pas de 16.
    layer.out_ld = last ? (conv.cout + 7) / 8 * 8 : cout;
    if (layer.residual && layer.out_ld != layer.in_ld) {
      error = "résiduel entre deux pas différents";
      return false;
    }
    std::vector<std::uint32_t> packed;
    for (int tile = 0; tile < layer.tiles; ++tile) {
      std::vector<__half> w(static_cast<std::size_t>(layer.ct) * 9 * cin);
      for (int n = 0; n < layer.ct; ++n)
        for (int tap = 0; tap < 9; ++tap)
          for (int c = 0; c < cin; ++c)
            w[(static_cast<std::size_t>(n) * 9 + tap) * cin + c] =
                __float2half_rn(folded.weight(tile * layer.ct + n, tap, c));
      rn::pack(w, layer.ct, 9, cin, packed);
    }
    std::vector<float> biases;
    for (int o = 0; o < cout; ++o) biases.push_back(folded.bias(o));
    if (layer.activation) {
      std::vector<float> slope(static_cast<std::size_t>(cout), 0.F);
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
    const int rows = rows_of(layer.kind);
    const int th = rn::kWarps * rows;
    layer.grid = dim3(static_cast<unsigned>((layer.ow + rn::kTW - 1) / rn::kTW),
                      static_cast<unsigned>((layer.oh + th - 1) / th), static_cast<unsigned>(layer.tiles));
    layer.smem = rn::smem_bytes(layer.kc, layer.ct, 3, stride_of(layer.kind), rows);
    layer.ksplit = split_for(layer, batch);
    layer.grid.z *= static_cast<unsigned>(layer.ksplit * batch);
    if (layer.ksplit > 1) {
      partial = std::max(partial, align256(std::int64_t{batch} * layer.ksplit * layer.oh * layer.ow *
                                           layer.tiles * layer.ct * 4));
    }
    return true;
  }

  // Les parts de K d'une couche dont la grille (lot compris) fait moins de deux vagues (le bloc 0, au 1/32
  // de l'image) : assez pour deux vagues, au plus une par prise et tranche. Ni une couche lue dans l'image.
  int split_for(const Layer& layer, int batch) const {
    if (layer.kind == Kind::kImage) return 1;
    const long blocks = static_cast<long>(layer.grid.x) * layer.grid.y * layer.grid.z * batch;
    const long want = 2L * sms_;
    if (blocks >= want) return 1;
    const long units = static_cast<long>(layer.cin / layer.kc) * 9;
    return static_cast<int>(std::min((want + blocks - 1) / blocks, units));
  }

  // La plus large tuile de sortie (64, ou 32 en f32) qui laisse assez de blocs pour remplir la carte (deux
  // vagues), lot compris.
  int choose_tile(const Layer& layer, int cout, int batch) const {
    const int th = rn::kWarps * rows_of(layer.kind);
    const long cells =
        static_cast<long>((layer.ow + rn::kTW - 1) / rn::kTW) * ((layer.oh + th - 1) / th) * batch;
    for (const int ct : {64, 32}) {
      if ((acc32_ && ct == 64) || cout % ct != 0) continue;
      if (cells * (cout / ct) >= 2L * sms_) return ct;
    }
    return 16;
  }

  cudaError_t run(cudaStream_t stream, std::size_t index, const void* in, void* out, void* workspace) const {
    if (index >= pieces_.size() || pieces_[index].layers.empty()) return cudaErrorInvalidValue;
    const Piece& piece = pieces_[index];
    auto* ws = static_cast<std::uint8_t*>(workspace);
    __half* const buffers[2] = {reinterpret_cast<__half*>(ws), reinterpret_cast<__half*>(ws + piece.peak)};
    auto* const partial = reinterpret_cast<float*>(ws + 2 * piece.peak);
    const __half* from = static_cast<const __half*>(in);
    for (std::size_t i = 0; i < piece.layers.size(); ++i) {
      const Layer& layer = piece.layers[i];
      const bool last = i + 1 == piece.layers.size();
      __half* to = last ? static_cast<__half*>(out) : buffers[i % 2];
      rn::Params p;
      p.in = from;
      p.in_ld = layer.in_ld;
      p.in_plane = static_cast<long>(layer.ih) * layer.iw;
      p.ih = layer.ih;
      p.iw = layer.iw;
      p.out = to;
      p.out_ld = layer.out_ld;
      p.out_ch = layer.out_ld;
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
      p.batch = piece.batch;
      p.in_batch = static_cast<long>(layer.ih) * layer.iw * layer.in_ld;
      p.out_batch = static_cast<long>(layer.oh) * layer.ow * layer.out_ld;
      p.skip_batch = p.in_batch;
      layer.launch(layer.grid, layer.smem, stream, p);
      if (layer.ksplit > 1) launch_reduce(layer.tiles * layer.ct, piece.batch, stream, p);
      from = to;
    }
    return cudaGetLastError();
  }

  bool acc32_;
  int sms_ = 1;
  std::vector<Piece> pieces_;
};

}  // namespace

std::unique_ptr<TfixNativeNet> TfixNativeNet::create(const std::vector<RifeNativePiece>& pieces, int height,
                                                     int width, bool acc32, std::string& error) {
  if (pieces.size() != 2 + kernels::kTfixLevels || !kernels::tfix_size_valid(height, width)) {
    error = "l'encodeur, trois blocs et la tête attendus, à une taille paire";
    return nullptr;
  }
  // Un rôle à la fois : l'encodeur seul, ou les blocs et la tête seuls (les autres morceaux vides).
  const bool encoder = !pieces[0].empty();
  for (std::size_t i = 1; i < pieces.size(); ++i) {
    if (pieces[i].empty() == !encoder) continue;
    error = "l'encodeur seul, ou les trois blocs et la tête seuls, attendus";
    return nullptr;
  }
  auto net = std::make_unique<Net>(acc32);
  if (!encoder) net->skip();
  if (encoder &&
      !net->plan(pieces[0], height, width, 0, 1, height / 2, width / 2, kernels::kTfixTraits, error)) {
    error = "encodeur : " + error;
    return nullptr;
  }
  for (int b = 0; b < kernels::kTfixLevels; ++b) {
    const kernels::TfixDims x = kernels::tfix_block_x(b, height, width);
    const kernels::TfixDims y = kernels::tfix_block_y(b, height, width);
    if (!encoder && !net->plan(pieces[static_cast<std::size_t>(b) + 1], x.height, x.width,
                               kernels::tfix_in_ld(x.channels, true), kernels::kTfixSupports, y.height,
                               y.width, kernels::kTfixY, error)) {
      error = "bloc " + std::to_string(b) + " : " + error;
      return nullptr;
    }
  }
  const kernels::TfixDims half = kernels::tfix_level(2, height, width);
  if (!encoder &&
      !net->plan(pieces.back(), half.height, half.width, kernels::tfix_in_ld(kernels::kTfixConf, true),
                 kernels::kTfixSupports, half.height, half.width, 1, error)) {
    error = "tête : " + error;
    return nullptr;
  }
  return net;
}

}  // namespace kirari::gpu::cuda
