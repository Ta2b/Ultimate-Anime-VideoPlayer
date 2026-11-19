// Le squelette CNN de GMFlow natif (gmflow_backbone_net.h) : chaque opération du plan de
// prepare/gmflow_backbone.h devient un lancement, planifié une fois (noyau, tuile des canaux de sortie,
// grille, poids rangés pour le mma), et chaque tenseur une place dans une zone, par durée de vie : deux
// tenseurs qui vivent ensemble ne se recouvrent pas, deux qui se suivent partagent la mémoire. Les
// statistiques des InstanceNorm (Σx, Σx² par image et par canal, f32) ont chacune leur place, remises à zéro
// au début de chaque passage.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/gmflow_backbone_kernels.cuh"
#include "gpu/cuda/backend/gmflow_backbone_net.h"

namespace kirari::gpu::cuda {
namespace {

namespace gb = gmflow_backbone;
using prepare::GmflowBackbone;
using prepare::GmflowConv;
using prepare::GmflowOp;
using prepare::GmflowOpKind;

static_assert(gb::kStemK == prepare::kGmflowStemK && gb::kMaxChannels == prepare::kGmflowMaxChannels);
static_assert(prepare::kGmflowStemTaps <= gb::kStemK);

constexpr int kImages = 2;
constexpr int kStatsFloats = kImages * 2 * gb::kMaxChannels;  // la place d'une InstanceNorm
constexpr int kSmemLimit = 48 * 1024;

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

// Poids FP16 [n][taps][cin] dans l'ordre des fragments B de mma m16n8k16, celui que lit rn::accumulate (le
// même que rn::packed_weights) : pas k (prise, tranche de 16 entrées), paire de tuiles n8, lane ; b0 =
// entrées 2t et 2t + 1 de la sortie 8·tuile + g, b1 = entrées 2t + 8 et 2t + 9.
void pack(const std::vector<__half>& w, int ct, int taps, int cin, std::vector<std::uint32_t>& out) {
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

using Launch = void (*)(dim3 grid, int smem, cudaStream_t stream, const gb::Params& p);

template <int Kc, int CT, int KH, int S, int Rows, gb::In I>
void launch_conv(dim3 grid, int smem, cudaStream_t stream, const gb::Params& p) {
  gb::conv_kernel<Kc, CT, KH, S, Rows, I, gb::Acc::kF32>
      <<<grid, gb::kThreads, static_cast<std::size_t>(smem), stream>>>(p);
}
template <int CT>
void launch_stem(dim3 grid, int smem, cudaStream_t stream, const gb::Params& p) {
  gb::stem_kernel<CT, gb::Acc::kF32><<<grid, gb::kThreads, static_cast<std::size_t>(smem), stream>>>(p);
}

// Ce qui choisit le noyau d'une convolution : noyau, pas, tranche des canaux d'entrée, lecture normalisée.
struct Shape {
  int kernel = 3;
  int stride = 1;
  int kc = 32;
  bool norm = false;
};

template <int Kc, int KH, int S, int Rows, gb::In I>
Launch by_tile(int ct) {
  if (ct == 16) return launch_conv<Kc, 16, KH, S, Rows, I>;
  if (ct == 32) return launch_conv<Kc, 32, KH, S, Rows, I>;
  return nullptr;
}

Launch pick(const Shape& s, int ct) {
  using gb::In;
  if (s.kernel == 7) return ct == 64 ? launch_stem<64> : ct == 32 ? launch_stem<32> : nullptr;
  if (s.kernel == 3 && s.stride == 1) {
    if (s.kc == 64)
      return s.norm ? by_tile<64, 3, 1, 2, In::kNorm>(ct) : by_tile<64, 3, 1, 2, In::kPlain>(ct);
    return s.norm ? by_tile<32, 3, 1, 2, In::kNorm>(ct) : by_tile<32, 3, 1, 2, In::kPlain>(ct);
  }
  if (s.norm) return nullptr;
  if (s.kernel == 3) return s.kc == 32 ? by_tile<32, 3, 2, 1, In::kPlain>(ct) : nullptr;
  if (s.kernel != 1) return nullptr;
  if (s.stride == 1)
    return s.kc == 64 ? by_tile<64, 1, 1, 2, In::kPlain>(ct) : by_tile<32, 1, 1, 2, In::kPlain>(ct);
  return s.kc == 64 ? by_tile<64, 1, 2, 2, In::kPlain>(ct) : by_tile<32, 1, 2, 2, In::kPlain>(ct);
}

int rows_of(const Shape& s) {
  if (s.kernel == 7) return gb::kStemRows;
  return s.kernel == 3 && s.stride == 2 ? 1 : 2;
}

// La mémoire partagée d'un bloc : la tuile d'entrée, ou la sortie et la réduction des statistiques.
int smem_of(const Shape& s, int ct) {
  const int th = gb::kWarps * rows_of(s);
  const int out = th * gb::kTW * (ct + 8) * 2 + gb::kReduceBytes;
  if (s.kernel == 7)
    return std::max(gb::kThreads * gb::kStemLds * 2 + 3 * gb::kStemPatchRows * gb::kStemPatchCols * 2, out);
  const int gs = s.kernel == 1 ? 1 : s.stride;
  const int in = ((th - 1) * gs + s.kernel) * ((gb::kTW - 1) * gs + s.kernel) * (s.kc + 8) * 2;
  return std::max(in, out);
}

// Un tenseur placé : sa taille, sa place dans la zone, sa vie (opérations de la première écriture et de la
// dernière lecture), et s'il est donné au lancement (les images, les deux sorties).
struct Placed {
  int height = 0;
  int width = 0;
  int channels = 0;
  std::int64_t bytes = 0;
  std::int64_t offset = 0;
  int first = 0;
  int last = -1;
  bool external = false;
};

struct Layer {
  bool end = false;
  Launch launch = nullptr;
  gb::Params p;
  gb::EndParams e;
  int from = -1, skip = -1, to = -1;
  Buffer w, bias;
  dim3 grid;
  int smem = 0;
};

class Net final : public GmflowBackboneNet {
 public:
  Net(int height, int width) : height_(height), width_(width) {
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess)
      static_cast<void>(cudaDeviceGetAttribute(&sms_, cudaDevAttrMultiProcessorCount, device));
    sms_ = std::max(sms_, 1);
  }

  bool plan(const GmflowBackbone& b, std::string& error) {
    if (!place(b, error)) return false;
    layers_ = std::vector<Layer>(b.ops.size());
    for (std::size_t i = 0; i < b.ops.size(); ++i) {
      const bool ok = b.ops[i].kind == GmflowOpKind::kConv ? build_conv(b, b.ops[i], layers_[i], error)
                                                           : build_end(b, b.ops[i], layers_[i], error);
      if (!ok) {
        error = "opération " + std::to_string(i) + " : " + error;
        return false;
      }
    }
    return sent(zone_.allocate(static_cast<std::size_t>(zone_bytes_)), error) &&
           sent(stats_.allocate(static_cast<std::size_t>(stats_bytes_)), error);
  }

  std::int64_t workspace_bytes() const override { return zone_bytes_ + stats_bytes_; }

  cudaError_t run(cudaStream_t stream, const void* images, const GmflowFeatures& out) const override {
    if (images == nullptr || out.quarter == nullptr || out.eighth == nullptr) return cudaErrorInvalidValue;
    if (const cudaError_t e =
            cudaMemsetAsync(stats_.as<void>(), 0, static_cast<std::size_t>(stats_bytes_), stream);
        e != cudaSuccess)
      return e;
    for (const Layer& layer : layers_) {
      if (layer.end) {
        gb::EndParams e = layer.e;
        e.in = address(layer.from, images, out);
        e.skip = address(layer.skip, images, out);
        e.out = address(layer.to, images, out);
        gb::block_end_kernel<<<layer.grid, 256, 0, stream>>>(e);
      } else {
        gb::Params p = layer.p;
        p.in = address(layer.from, images, out);
        p.out = address(layer.to, images, out);
        layer.launch(layer.grid, layer.smem, stream, p);
      }
    }
    return cudaGetLastError();
  }

 private:
  static bool sent(cudaError_t result, std::string& error) {
    if (result == cudaSuccess) return true;
    static_cast<void>(cudaGetLastError());
    error = std::string("mémoire du squelette de GMFlow : ") + cudaGetErrorString(result);
    return false;
  }

  __half* address(int id, const void* images, const GmflowFeatures& out) const {
    if (id == image_) return static_cast<__half*>(const_cast<void*>(images));
    if (id == quarter_) return static_cast<__half*>(out.quarter);
    if (id == eighth_) return static_cast<__half*>(out.eighth);
    const Placed& t = tensors_[static_cast<std::size_t>(id)];
    return reinterpret_cast<__half*>(zone_.as<std::uint8_t>() + t.offset);
  }

  float* stats_at(int id) const {
    return id < 0 ? nullptr : stats_.as<float>() + std::int64_t{id} * kStatsFloats;
  }

  bool valid(const GmflowBackbone& b, const GmflowOp& op) const {
    const auto count = static_cast<int>(b.tensors.size());
    const auto in_range = [&](int id) { return id >= 0 && id < count; };
    const bool end = op.kind == GmflowOpKind::kBlockEnd;
    return in_range(op.in.tensor) && in_range(op.out) && (!end || in_range(op.skip.tensor)) &&
           (end || (op.conv >= 0 && op.conv < static_cast<int>(b.convs.size()))) && op.in.stats < b.stats &&
           op.skip.stats < b.stats && op.stats < b.stats;
  }

  // Les tailles, la vie et la place de chaque tenseur : les plus grands d'abord, chacun au plus bas de la
  // zone sans recouvrir un tenseur déjà placé qui vit en même temps.
  bool place(const GmflowBackbone& b, std::string& error) {
    image_ = b.image;
    quarter_ = b.quarter;
    eighth_ = b.eighth;
    tensors_.assign(b.tensors.size(), Placed{});
    for (std::size_t i = 0; i < b.tensors.size(); ++i) {
      Placed& t = tensors_[i];
      t.height = height_ >> b.tensors[i].level;
      t.width = width_ >> b.tensors[i].level;
      t.channels = b.tensors[i].channels;
      t.bytes = align256(std::int64_t{kImages} * t.height * t.width * t.channels * 2);
      t.first = static_cast<int>(b.ops.size());
      t.external = static_cast<int>(i) == b.image || static_cast<int>(i) == b.quarter ||
                   static_cast<int>(i) == b.eighth;
    }
    for (std::size_t s = 0; s < b.ops.size(); ++s) {
      const GmflowOp& op = b.ops[s];
      if (!valid(b, op)) {
        error = "plan : opération " + std::to_string(s) + " hors des tenseurs";
        return false;
      }
      const auto now = static_cast<int>(s);
      tensors_[static_cast<std::size_t>(op.out)].first =
          std::min(tensors_[static_cast<std::size_t>(op.out)].first, now);
      tensors_[static_cast<std::size_t>(op.in.tensor)].last = now;
      if (op.kind == GmflowOpKind::kBlockEnd) tensors_[static_cast<std::size_t>(op.skip.tensor)].last = now;
    }
    for (Placed& t : tensors_) t.last = std::max(t.last, t.first);
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < tensors_.size(); ++i) {
      if (!tensors_[i].external) order.push_back(i);
    }
    std::ranges::stable_sort(
        order, [&](std::size_t a, std::size_t c) { return tensors_[a].bytes > tensors_[c].bytes; });
    std::vector<std::size_t> placed;
    for (const std::size_t i : order) {
      Placed& t = tensors_[i];
      std::vector<std::pair<std::int64_t, std::int64_t>> taken;
      for (const std::size_t j : placed) {
        const Placed& u = tensors_[j];
        if (u.first <= t.last && t.first <= u.last) taken.emplace_back(u.offset, u.offset + u.bytes);
      }
      std::ranges::sort(taken);
      std::int64_t offset = 0;
      for (const auto& [begin, end] : taken) {
        if (offset + t.bytes <= begin) break;
        offset = std::max(offset, end);
      }
      t.offset = offset;
      zone_bytes_ = std::max(zone_bytes_, offset + t.bytes);
      placed.push_back(i);
    }
    stats_bytes_ = align256(std::int64_t{std::max(b.stats, 1)} * kStatsFloats * 4);
    return true;
  }

  // La tuile des canaux de sortie : la plus large qui laisse deux vagues de blocs sur la carte.
  int choose_tile(const Shape& s, int cout, long cells) const {
    const int widest = s.kernel == 7 ? 64 : 32;
    for (int ct = widest; ct >= 32; ct /= 2) {
      if (cout % ct == 0 && cells * (cout / ct) >= 2L * sms_) return ct;
    }
    return s.kernel == 7 ? 32 : 16;
  }

  // Les poids d'une convolution, tuile après tuile : [ct][prises][cin] FP16, rangés pour le mma. La 7 × 7 :
  // [ct][160], k = prise · 3 + plan, ce que le dépliage de stem_kernel met en face.
  static void weights(const GmflowConv& c, int ct, std::vector<std::uint32_t>& packed) {
    const bool stem = c.kernel == 7;
    const int taps = stem ? 1 : c.kernel * c.kernel;
    const int k = stem ? gb::kStemK : c.cin;
    const int real = stem ? prepare::kGmflowStemTaps : c.cin;
    for (int tile = 0; tile < c.cout / ct; ++tile) {
      std::vector<__half> w(static_cast<std::size_t>(ct) * taps * k, __float2half_rn(0.F));
      for (int n = 0; n < ct; ++n)
        for (int t = 0; t < taps; ++t)
          for (int i = 0; i < real; ++i) {
            const std::size_t from = (static_cast<std::size_t>(tile * ct + n) * taps + t) * real + i;
            w[(static_cast<std::size_t>(n) * taps + t) * k + i] = __float2half_rn(c.weights[from]);
          }
      pack(w, ct, taps, k, packed);
    }
  }

  bool build_conv(const GmflowBackbone& b, const GmflowOp& op, Layer& layer, std::string& error) {
    const GmflowConv& c = b.convs[static_cast<std::size_t>(op.conv)];
    const Placed& in = tensors_[static_cast<std::size_t>(op.in.tensor)];
    const Placed& out = tensors_[static_cast<std::size_t>(op.out)];
    const Shape shape{
        .kernel = c.kernel, .stride = op.stride, .kc = c.cin % 64 == 0 ? 64 : 32, .norm = op.in.stats >= 0};
    const int th = gb::kWarps * rows_of(shape);
    const long cells =
        static_cast<long>((out.width + gb::kTW - 1) / gb::kTW) * ((out.height + th - 1) / th) * kImages;
    const int ct = choose_tile(shape, c.cout, cells);
    layer.launch =
        c.cout % ct == 0 && c.cout <= gb::kMaxChannels && c.cout == out.channels ? pick(shape, ct) : nullptr;
    layer.smem = smem_of(shape, ct);
    const bool stem = c.kernel == 7;
    if (layer.launch == nullptr || layer.smem > kSmemLimit ||
        (stem ? op.in.tensor != image_ : c.cin != in.channels) || out.height * op.stride != in.height ||
        out.width * op.stride != in.width) {
      error = "aucun noyau pour cette convolution";
      return false;
    }
    std::vector<std::uint32_t> packed;
    weights(c, ct, packed);
    if (!sent(layer.w.upload(packed.data(), packed.size() * sizeof(std::uint32_t)), error) ||
        !sent(layer.bias.upload(c.bias.data(), c.bias.size() * sizeof(float)), error))
      return false;
    layer.from = op.in.tensor;
    layer.to = op.out;
    gb::Params& p = layer.p;
    p.in_ld = stem ? 3 : in.channels;
    p.ih = in.height;
    p.iw = in.width;
    p.in_stats = stats_at(op.in.stats);
    p.out_ld = out.channels;
    p.oh = out.height;
    p.ow = out.width;
    p.out_stats = stats_at(op.stats);
    p.w = layer.w.as<const uint4>();
    p.bias = layer.bias.as<const float>();
    p.cin = stem ? gb::kStemK : c.cin;
    p.tiles = c.cout / ct;
    p.eps = b.epsilon;
    for (int i = 0; i < 3; ++i) {
      p.pad[i] = b.pad[static_cast<std::size_t>(i)];
    }
    layer.grid =
        dim3(static_cast<unsigned>((out.width + gb::kTW - 1) / gb::kTW),
             static_cast<unsigned>((out.height + th - 1) / th), static_cast<unsigned>(kImages * p.tiles));
    return true;
  }

  bool build_end(const GmflowBackbone& b, const GmflowOp& op, Layer& layer, std::string& error) {
    const Placed& in = tensors_[static_cast<std::size_t>(op.in.tensor)];
    const Placed& skip = tensors_[static_cast<std::size_t>(op.skip.tensor)];
    const Placed& out = tensors_[static_cast<std::size_t>(op.out)];
    if (in.channels != out.channels || skip.channels != out.channels || in.height != out.height ||
        skip.height != out.height || out.channels % 8 != 0 || out.channels > gb::kMaxChannels) {
      error = "fin de bloc : formes";
      return false;
    }
    layer.end = true;
    layer.from = op.in.tensor;
    layer.skip = op.skip.tensor;
    layer.to = op.out;
    gb::EndParams& e = layer.e;
    e.in_stats = stats_at(op.in.stats);
    e.in_relu = op.in.relu;
    e.skip_stats = stats_at(op.skip.stats);
    e.skip_relu = op.skip.relu;
    e.channels = out.channels;
    e.pixels = static_cast<long>(out.height) * out.width;
    e.eps = b.epsilon;
    const long threads = e.pixels * (out.channels / 8);
    layer.grid = dim3(static_cast<unsigned>((threads + 255) / 256), kImages);
    return true;
  }

  int height_;
  int width_;
  int sms_ = 1;
  int image_ = -1, quarter_ = -1, eighth_ = -1;
  std::vector<Placed> tensors_;
  std::vector<Layer> layers_;
  std::int64_t zone_bytes_ = 0;
  std::int64_t stats_bytes_ = 0;
  Buffer zone_;
  Buffer stats_;
};

}  // namespace

std::unique_ptr<GmflowBackboneNet> GmflowBackboneNet::create(const prepare::GmflowBackbone& backbone,
                                                             int height, int width, std::string& error) {
  if (height < 32 || width < 32 || height % 32 != 0 || width % 32 != 0) {
    error = "taille des images de GMFlow : des multiples de 32 attendus";
    return nullptr;
  }
  auto net = std::make_unique<Net>(height, width);
  if (!net->plan(backbone, error)) return nullptr;
  return net;
}

}  // namespace kirari::gpu::cuda
