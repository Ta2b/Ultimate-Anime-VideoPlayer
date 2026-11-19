#pragma once
// La planification commune des petits réseaux natifs de GMFSS (--gmfss-native, docs/gmfss-natif.md) sur les
// noyaux de rife_native_kernels.cuh, tirée de gmfss_native_net.cu (le GridNet) : chaque convolution d'un
// réseau aux types de prepare/gmfss_grid.h planifiée une fois (tranche des canaux d'entrée, tuile des canaux
// de sortie, grille, K découpé quand la grille ne remplit pas la carte), ses poids rangés dans l'ordre des
// fragments B du mma, et les tenseurs placés dans une zone par durée de vie : deux qui vivent ensemble ne se
// recouvrent jamais (le résiduel n'est donc jamais la sortie), deux qui se suivent partagent la mémoire.
// Ajouts pour la paire (gmfss_pair_native_net.cu) : des tenseurs externes, donnés au lancement ; une image en
// trois plans lue par In::kImage en pas 2 ; la zone donnée au lancement, que deux réseaux partagent.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/rife_native_kernels.cuh"
#include "gpu/cuda/backend/rife_native_net.h"
#include "prepare/gmfss_grid.h"

namespace kirari::gpu::cuda::gmfss_plan {

namespace rn = rife_native;

constexpr int pad16(int c) {
  return (c + 15) / 16 * 16;
}
constexpr std::int64_t align256(std::int64_t bytes) {
  return (bytes + 255) / 256 * 256;
}
inline unsigned blocks_for(long count) {
  return static_cast<unsigned>((count + 255) / 256);
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

// Les façons de lancer une convolution.
enum class Kind : std::uint8_t {
  kSame,       // 3 × 3, pas 1
  kHalf,       // 3 × 3, pas 2
  kPhase,      // transposée 4 × 4 de pas 2, une phase par blockIdx.z
  kShuffle,    // 3 × 3 suivie du PixelShuffle(2) : la phase (py, px) écrite au pixel (2y + py, 2x + px)
  kImageHalf,  // 3 × 3, pas 2, sur les trois plans d'une image (In::kImage, 16 canaux dont 13 nuls)
};

using Launch = void (*)(dim3 grid, int smem, cudaStream_t stream, const rn::Params& p);

template <int Kc, int CT, int KH, int S, int Rows, rn::In I, rn::Out O, rn::Acc A>
void launch_one(dim3 grid, int smem, cudaStream_t stream, const rn::Params& p) {
  rn::conv_kernel<Kc, CT, KH, S, Rows, I, O, A>
      <<<grid, rn::kThreads, static_cast<std::size_t>(smem), stream>>>(p);
}

// La tuile de 64 colonnes n'existe qu'en sommes f16 : en f32 ou mixtes, ses accumulateurs f32 débordent.
template <int Kc, int KH, int S, int Rows, rn::In I, rn::Out O, rn::Acc A>
Launch by_tile(int ct) {
  if (ct == 16) return launch_one<Kc, 16, KH, S, Rows, I, O, A>;
  if (ct == 32) return launch_one<Kc, 32, KH, S, Rows, I, O, A>;
  if constexpr (A == rn::Acc::kF16) {
    if (ct == 64) return launch_one<Kc, 64, KH, S, Rows, I, O, A>;
  }
  return nullptr;
}

template <rn::Acc A>
Launch pick(Kind kind, int kc, int ct) {
  using rn::In;
  using rn::Out;
  switch (kind) {
    case Kind::kSame:
      if (kc == 16) return by_tile<16, 3, 1, 2, In::kNhwc, Out::kNhwc, A>(ct);
      if (kc == 64) return by_tile<64, 3, 1, 2, In::kNhwc, Out::kNhwc, A>(ct);
      return nullptr;
    case Kind::kHalf:
      return kc == 32 ? by_tile<32, 3, 2, 1, In::kNhwc, Out::kNhwc, A>(ct) : nullptr;
    case Kind::kPhase:
      return kc == 64 ? by_tile<64, 2, 1, 2, In::kNhwc, Out::kPhase, A>(ct) : nullptr;
    case Kind::kShuffle:
      return kc == 64 ? by_tile<64, 3, 1, 2, In::kNhwc, Out::kPhase, A>(ct) : nullptr;
    case Kind::kImageHalf:
      return kc == 16 ? by_tile<16, 3, 2, 1, In::kImage, Out::kNhwc, A>(ct) : nullptr;
  }
  return nullptr;
}

// Les sommes de RIFE natif (rife_native_net.h) vues par les noyaux.
inline Launch pick_for(RifeNativeSums sums, Kind kind, int kc, int ct) {
  switch (sums) {
    case RifeNativeSums::kF16:
      return pick<rn::Acc::kF16>(kind, kc, ct);
    case RifeNativeSums::kMixed:
      return pick<rn::Acc::kMixed>(kind, kc, ct);
    case RifeNativeSums::kF32:
      return pick<rn::Acc::kF32>(kind, kc, ct);
  }
  return nullptr;
}

struct Shape3 {
  int kh = 3;
  int s = 1;
  int rows = 2;
};

inline Shape3 shape_of(Kind kind) {
  switch (kind) {
    case Kind::kHalf:
    case Kind::kImageHalf:
      return {.kh = 3, .s = 2, .rows = 1};
    case Kind::kPhase:
      return {.kh = 2, .s = 1, .rows = 2};
    default:
      return {.kh = 3, .s = 1, .rows = 2};
  }
}

inline bool phased(Kind kind) {
  return kind == Kind::kPhase || kind == Kind::kShuffle;
}

// Poids FP16 [n][taps][cin] rangés dans l'ordre des fragments B de mma m16n8k16 (comme rife_native_net.cu) :
// pas k (prise, tranche de 16 entrées), paire de tuiles n8, lane ; b0 = entrées 2t et 2t + 1 de la sortie
// 8·tuile + g, b1 = entrées 2t + 8 et 2t + 9.
inline void pack(const std::vector<__half>& w, int ct, int taps, int cin, std::vector<std::uint32_t>& out) {
  const int kcs = cin / 16, nps = ct / 16;
  const auto bits = [&](int n, int tap, int c) -> std::uint32_t {
    return __half_as_ushort(w[(static_cast<std::size_t>(n) * taps + tap) * cin + c]);
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

// Le poids (o, prise, c) d'une convolution, zéro dans le rembourrage des canaux.
inline float weight_of(const prepare::GmfssGridConv& conv, int o, int tap, int c) {
  if (o >= conv.cout || c >= conv.cin) return 0.F;
  const int taps = conv.transposed ? 16 : 9;
  return conv.weights[(static_cast<std::size_t>(o) * taps + tap) * conv.cin + c];
}

struct Layer {
  Kind kind = Kind::kSame;
  Launch launch = nullptr;
  int kc = 16, ct = 16, tiles = 1, phases = 1, ksplit = 1;
  int ih = 0, iw = 0, oh = 0, ow = 0;
  int in_ld = 0, out_ld = 0, cin = 0, skip_ld = 0;
  int from = 0, to = 0, skip = -1;
  float in_slope = 1.F;
  bool activation = false;
  Buffer w, bias, slope;
  dim3 grid;
  int smem = 0;
};

struct Placed {
  int height = 0;
  int width = 0;
  int channels = 0;
  std::int64_t offset = 0;
  std::int64_t bytes = 0;
  int first = 0;      // première écriture (-1 : écrit avant la première convolution)
  int last = 0;       // dernière lecture (le nombre de convolutions : lu après la dernière)
  int external = -1;  // rang parmi les tenseurs donnés au lancement ; -1 : dans la zone
};

// Ce que le réseau prend et rend hors de ses convolutions. external : les tenseurs dont la mémoire est donnée
// au lancement (dans l'ordre de bound) ; image : l'un d'eux, trois plans FP16 (-1 : aucun) ; written : ceux
// de la zone qu'on remplit avant la première convolution ; kept : ceux qu'on lit après la dernière.
struct Io {
  std::vector<int> external;
  int image = -1;
  std::vector<int> written;
  std::vector<int> kept;
};

// Un réseau planifié pour une taille : height × width au niveau 0 (kGmfssFullLevel au double).
class ConvNet {
 public:
  ConvNet(int height, int width, RifeNativeSums sums) : height_(height), width_(width), sums_(sums) {
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess)
      static_cast<void>(cudaDeviceGetAttribute(&sms_, cudaDevAttrMultiProcessorCount, device));
    sms_ = std::max(sms_, 1);
  }

  bool plan(const prepare::GmfssGrid& grid, const Io& io, std::string& error) {
    if (grid.steps.empty()) {
      error = "réseau vide";
      return false;
    }
    place(grid, io);
    layers_ = std::vector<Layer>(grid.steps.size());
    for (std::size_t i = 0; i < grid.steps.size(); ++i) {
      if (!build(grid.steps[i], io, layers_[i], error)) {
        error = "convolution " + std::to_string(i) + " : " + error;
        return false;
      }
    }
    return true;
  }

  // Octets de la zone des tenseurs et des sommes partielles (K découpé).
  std::int64_t zone_bytes() const { return zone_bytes_; }
  std::int64_t partial_bytes() const { return partial_bytes_; }
  const Placed& tensor(int id) const { return tensors_[static_cast<std::size_t>(id)]; }

  // Un tenseur de la zone qui commence à zone.
  static __half* at(void* zone, const Placed& t) {
    return reinterpret_cast<__half*>(static_cast<std::uint8_t*>(zone) + t.offset);
  }

  // Les convolutions sur stream ; bound : la mémoire des tenseurs externes, dans l'ordre d'Io::external.
  void run(cudaStream_t stream, void* zone, float* partial, std::span<const void* const> bound) const {
    for (const Layer& layer : layers_) launch(stream, layer, zone, partial, bound);
  }

 private:
  __half* address(int id, void* zone, std::span<const void* const> bound) const {
    const Placed& t = tensor(id);
    if (t.external < 0) return at(zone, t);
    return static_cast<__half*>(const_cast<void*>(bound[static_cast<std::size_t>(t.external)]));
  }

  void lifetimes(const prepare::GmfssGrid& grid, const Io& io) {
    const auto steps = static_cast<int>(grid.steps.size());
    for (std::size_t s = 0; s < grid.steps.size(); ++s) {
      const prepare::GmfssGridStep& step = grid.steps[s];
      const auto now = static_cast<int>(s);
      Placed& to = tensors_[static_cast<std::size_t>(step.to)];
      to.first = std::min(to.first, now);
      for (const int read : {step.from, step.skip}) {
        if (read >= 0) tensors_[static_cast<std::size_t>(read)].last = now;
      }
    }
    for (const int id : io.written) tensors_[static_cast<std::size_t>(id)].first = -1;
    for (const int id : io.kept) tensors_[static_cast<std::size_t>(id)].last = steps;
    // Un tenseur écrit mais jamais relu vit au moins le temps de son écriture : sinon il pourrait recouvrir
    // l'entrée de la convolution qui l'écrit.
    for (Placed& t : tensors_) t.last = std::max(t.last, t.first);
  }

  // La durée de vie et la place de chaque tenseur de la zone : les plus grands d'abord, chacun au plus bas de
  // la zone sans recouvrir un tenseur déjà placé qui vit en même temps.
  void place(const prepare::GmfssGrid& grid, const Io& io) {
    tensors_.assign(grid.tensors.size(), Placed{});
    for (std::size_t i = 0; i < grid.tensors.size(); ++i) {
      const prepare::GmfssGridTensor& t = grid.tensors[i];
      Placed& p = tensors_[i];
      const bool full = t.level == prepare::kGmfssFullLevel;
      p.height = full ? 2 * height_ : height_ >> t.level;
      p.width = full ? 2 * width_ : width_ >> t.level;
      p.channels = t.channels;
      p.bytes = align256(std::int64_t{p.height} * p.width * p.channels * 2);
      p.first = static_cast<int>(grid.steps.size());
      p.last = -1;
    }
    for (std::size_t e = 0; e < io.external.size(); ++e)
      tensors_[static_cast<std::size_t>(io.external[e])].external = static_cast<int>(e);
    lifetimes(grid, io);
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < tensors_.size(); ++i) {
      if (tensors_[i].external < 0) order.push_back(i);
    }
    std::ranges::stable_sort(
        order, [&](std::size_t a, std::size_t b) { return tensors_[a].bytes > tensors_[b].bytes; });
    std::vector<std::size_t> placed;
    for (const std::size_t i : order) {
      Placed& p = tensors_[i];
      std::vector<std::pair<std::int64_t, std::int64_t>> taken;
      for (const std::size_t j : placed) {
        const Placed& q = tensors_[j];
        if (q.first <= p.last && p.first <= q.last) taken.emplace_back(q.offset, q.offset + q.bytes);
      }
      std::ranges::sort(taken);
      std::int64_t offset = 0;
      for (const auto& [begin, end] : taken) {
        if (offset + p.bytes <= begin) break;
        offset = std::max(offset, end);
      }
      p.offset = offset;
      zone_bytes_ = std::max(zone_bytes_, offset + p.bytes);
      placed.push_back(i);
    }
  }

  static Kind kind_of(const prepare::GmfssGridStep& step, bool image) {
    if (image) return Kind::kImageHalf;
    if (step.shuffle) return Kind::kShuffle;
    if (step.conv.transposed) return Kind::kPhase;
    return step.conv.stride == 2 ? Kind::kHalf : Kind::kSame;
  }

  bool shapes(const prepare::GmfssGridStep& step, const Io& io, Layer& layer, std::string& error) const {
    const prepare::GmfssGridConv& conv = step.conv;
    const Placed& from = tensor(step.from);
    const Placed& to = tensor(step.to);
    const bool image = step.from == io.image;
    layer.kind = kind_of(step, image);
    if (image && (conv.stride != 2 || conv.cin > 3 || conv.transposed || step.shuffle)) {
      error = "l'image ne se lit qu'en 3 × 3 de pas 2";
      return false;
    }
    if (to.external >= 0) {
      error = "une convolution n'écrit pas un tenseur externe";
      return false;
    }
    layer.phases = phased(layer.kind) ? 4 : 1;
    layer.from = step.from;
    layer.to = step.to;
    layer.skip = step.skip;
    layer.cin = pad16(conv.cin);
    layer.in_ld = from.channels;
    layer.out_ld = to.channels;
    layer.ih = from.height;
    layer.iw = from.width;
    layer.oh = phased(layer.kind) ? from.height : to.height;
    layer.ow = phased(layer.kind) ? from.width : to.width;
    layer.in_slope = conv.in_slope;
    layer.activation = conv.out_slope != 1.F;
    const int scale = phased(layer.kind) ? 2 : 1;
    const int s = shape_of(layer.kind).s;
    if (layer.cin != layer.in_ld || to.height != scale * layer.oh || to.width != scale * layer.ow ||
        (s == 2 && (from.height != 2 * to.height || from.width != 2 * to.width))) {
      error = "canaux d'entrée ou taille de sortie";
      return false;
    }
    if (phased(layer.kind) && layer.activation) {
      error = "une sortie par phases n'a pas d'activation";
      return false;
    }
    if (step.skip >= 0) {
      const Placed& skip = tensor(step.skip);
      layer.skip_ld = skip.channels;
      if (phased(layer.kind) || skip.channels != to.channels || skip.height != to.height) {
        error = "résiduel";
        return false;
      }
    }
    return true;
  }

  bool build(const prepare::GmfssGridStep& step, const Io& io, Layer& layer, std::string& error) {
    if (!shapes(step, io, layer, error)) return false;
    const prepare::GmfssGridConv& conv = step.conv;
    const int cout = pad16(step.shuffle ? conv.cout / 4 : conv.cout);
    const bool half = shape_of(layer.kind).s == 2;
    layer.kc = layer.cin % 64 == 0 && !half ? 64 : layer.cin % 32 == 0 ? 32 : 16;
    layer.ct = choose_tile(layer, cout);
    layer.tiles = cout / layer.ct;
    std::vector<std::uint32_t> packed;
    std::vector<float> biases;
    weights(step, layer, cout, packed, biases);
    const auto sent = [&](cudaError_t result) {
      if (result == cudaSuccess) return true;
      static_cast<void>(cudaGetLastError());
      error = std::string("poids : ") + cudaGetErrorString(result);
      return false;
    };
    if (layer.activation) {
      const std::vector<float> slope(static_cast<std::size_t>(cout), conv.out_slope);
      if (!sent(layer.slope.upload(slope.data(), slope.size() * sizeof(float)))) return false;
    }
    if (!sent(layer.w.upload(packed.data(), packed.size() * sizeof(std::uint32_t))) ||
        !sent(layer.bias.upload(biases.data(), biases.size() * sizeof(float))))
      return false;
    layer.launch = pick_for(sums_, layer.kind, layer.kc, layer.ct);
    if (layer.launch == nullptr) {
      error = "aucun noyau pour cette convolution";
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
      partial_bytes_ = std::max(partial_bytes_, align256(std::int64_t{layer.ksplit} * layer.oh * layer.ow *
                                                         layer.grid.z * layer.ct * 4));
      layer.grid.z *= static_cast<unsigned>(layer.ksplit);
    }
    return true;
  }

  // Un paquet de poids par phase et par tuile. Transposée : prises (ty, tx) aux décalages py - 1 + ty, poids
  // (3 - py - 2 ty, 3 - px - 2 tx). PixelShuffle : le canal c de la phase est le canal c · 4 + phase.
  static void weights(const prepare::GmfssGridStep& step, const Layer& layer, int cout,
                      std::vector<std::uint32_t>& packed, std::vector<float>& biases) {
    const prepare::GmfssGridConv& conv = step.conv;
    const int taps = layer.kind == Kind::kPhase ? 4 : 9;
    const int real_out = step.shuffle ? conv.cout / 4 : conv.cout;
    const auto source = [&](int ph, int o) { return step.shuffle ? o * 4 + ph : o; };
    for (int ph = 0; ph < layer.phases; ++ph) {
      const int py = ph >> 1, px = ph & 1;
      for (int tile = 0; tile < layer.tiles; ++tile) {
        std::vector<__half> w(static_cast<std::size_t>(layer.ct) * taps * layer.cin);
        for (int n = 0; n < layer.ct; ++n) {
          const int o = tile * layer.ct + n;
          for (int tap = 0; tap < taps; ++tap) {
            const int at =
                layer.kind == Kind::kPhase ? (3 - py - 2 * (tap >> 1)) * 4 + (3 - px - 2 * (tap & 1)) : tap;
            for (int c = 0; c < layer.cin; ++c)
              w[(static_cast<std::size_t>(n) * taps + tap) * layer.cin + c] =
                  __float2half_rn(o < real_out ? weight_of(conv, source(ph, o), at, c) : 0.F);
          }
        }
        pack(w, layer.ct, taps, layer.cin, packed);
      }
      for (int o = 0; o < cout; ++o)
        biases.push_back(o < real_out ? conv.bias[static_cast<std::size_t>(source(ph, o))] : 0.F);
    }
  }

  // La plus large tuile de sortie (64 en sommes f16, 32 sinon) qui laisse deux vagues de blocs sur la carte.
  int choose_tile(const Layer& layer, int cout) const {
    const Shape3 shape = shape_of(layer.kind);
    const long cells = static_cast<long>((layer.ow + rn::kTW - 1) / rn::kTW) *
                       ((layer.oh + rn::kWarps * shape.rows - 1) / (rn::kWarps * shape.rows)) * layer.phases;
    for (const int ct : {64, 32}) {
      if ((sums_ != RifeNativeSums::kF16 && ct == 64) || cout % ct != 0) continue;
      if (cells * (cout / ct) >= 2L * sms_) return ct;
    }
    return 16;
  }

  // K découpé pour une grille de moins de deux vagues : assez de parts pour deux, au plus une par prise et
  // tranche.
  int split_for(const Layer& layer, int taps) const {
    const long blocks = static_cast<long>(layer.grid.x) * layer.grid.y * layer.grid.z;
    const long want = 2L * sms_;
    if (blocks >= want) return 1;
    const long units = static_cast<long>(layer.cin / layer.kc) * taps;
    return static_cast<int>(std::min((want + blocks - 1) / blocks, units));
  }

  void launch(cudaStream_t stream, const Layer& layer, void* zone, float* partial,
              std::span<const void* const> bound) const {
    rn::Params p;
    p.in = address(layer.from, zone, bound);
    p.in_ld = layer.in_ld;
    p.in_plane = static_cast<long>(layer.ih) * layer.iw;
    p.ih = layer.ih;
    p.iw = layer.iw;
    p.out = address(layer.to, zone, bound);
    p.out_ld = layer.out_ld;
    p.out_ch = layer.out_ld;
    p.oh = layer.oh;
    p.ow = layer.ow;
    p.w = layer.w.as<const uint4>();
    p.bias = layer.bias.as<const float>();
    p.slope = layer.activation ? layer.slope.as<const float>() : nullptr;
    p.skip = layer.skip >= 0 ? address(layer.skip, zone, bound) : nullptr;
    p.skip_ld = layer.skip_ld;
    p.cin = layer.cin;
    p.tiles = layer.tiles;
    p.ksplit = layer.ksplit;
    p.partial = layer.ksplit > 1 ? partial : nullptr;
    p.in_slope = layer.in_slope;
    layer.launch(layer.grid, layer.smem, stream, p);
    if (layer.ksplit > 1) {
      const int width = layer.phases * layer.tiles * layer.ct;
      const long count = static_cast<long>(p.oh) * p.ow * (width / 8);
      if (phased(layer.kind))
        rn::reduce_kernel<rn::Out::kPhase><<<blocks_for(count), 256, 0, stream>>>(p, width);
      else
        rn::reduce_kernel<rn::Out::kNhwc><<<blocks_for(count), 256, 0, stream>>>(p, width);
    }
  }

  int height_;
  int width_;
  RifeNativeSums sums_;
  int sms_ = 1;
  std::vector<Placed> tensors_;
  std::vector<Layer> layers_;
  std::int64_t zone_bytes_ = 0;
  std::int64_t partial_bytes_ = 0;
};

}  // namespace kirari::gpu::cuda::gmfss_plan
