// Le SPAN FP16 natif (lot C du chantier AnimeJaNai, docs/span-native16.md) derrière SpanFp8Body : les
// noyaux de span_native16_kernels.cuh, sans CUTLASS. 3 · blocs + 3 lancements par image, comme le corps
// CUTLASS, mais :
//   - le dégroupage est fondu au chargement de conv_1 (un lancement de moins) ;
//   - la concaténation n'existe plus : conv_cat lit les quatre tenseurs là où leurs convolutions les ont
//     écrits (trois écritures de 2·C octets par pixel en moins) ;
//   - sommes en f16 par défaut, le débit plein des tensor cores ; en f32 avec SpanFp8Weights::sums ;
//   - le corps en bandes (SpanFp8Weights::band) : chaque couche lit ce que la précédente vient d'écrire,
//     encore en L2, au lieu de 4 Go par image en mémoire (docs/span-native16.md, « Bandes »).
// La zone de travail tient six tenseurs de C canaux FP16 (768 o par pixel à 64 canaux, comme avant).
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/span_fp8_body.h"
#include "gpu/cuda/backend/span_native_conv.cuh"

namespace kirari::gpu::cuda {
namespace {

namespace sn = span_native;

// Mémoire d'appareil possédée.
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() {
    if (ptr_ != nullptr) static_cast<void>(cudaFree(ptr_));
  }
  cudaError_t upload(const void* host, std::size_t bytes) {
    if (ptr_ != nullptr) static_cast<void>(cudaFree(std::exchange(ptr_, nullptr)));
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

// Des lignes de demi-résolution [from, to) d'une couche.
struct Rows {
  int from = 0, to = 0;
};

// Le front d'une vague de couches (les bandes) : end, la fin des lignes que la couche d'avant a faites ;
// next(r) rend les lignes que la suivante, de rayon r, peut faire maintenant, sans attendre, et avance. Une
// couche finit r lignes avant celle qu'elle lit (son halo), sauf au bas de l'image, où tout est là. done : la
// fin atteinte par chaque couche à la vague d'avant, d'où elle reprend.
struct Front {
  int fh = 0;
  int end = 0;
  int* done = nullptr;
  int layer = 0;
  Rows next(int radius) {
    const int to = std::max(done[layer], end >= fh ? fh : end - radius);
    const Rows rows{done[layer], to};
    done[layer++] = to;
    end = to;
    return rows;
  }
};

// Les bandes par défaut (SpanFp8Weights::kBandAuto) : 128 lignes là où deux tranches de 128 lignes, la
// couche lue et la couche écrite, tiennent dans les deux tiers du L2 ; 64 sinon. Au banc de la RTX 4070 Ti
// (48 Mo de L2, b8f64 en 1080p, 15,7 Mo par tranche) : 10,64 ms à 128 lignes, 10,98 à 64. Les cartes à
// moins de L2 gardent les 64 lignes d'avant, faute de mesure.
int auto_band(int width, int channels) {
  int device = 0, l2 = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return 64;
  }
  const std::int64_t slice = std::int64_t{128} * width * channels * 2;
  return 3 * (2 * slice) <= 2 * std::int64_t{l2} ? 128 : 64;
}

constexpr std::int64_t kAlign = 256;
constexpr std::int64_t align_up(std::int64_t bytes) {
  return (bytes + kAlign - 1) / kAlign * kAlign;
}

// Le décalage de la sortie (SpanFp8Body::shift_output) : un fil par pixel d'une ligne, ses trois plans.
__global__ void shift_planes(const __half* from, __half* to, int height, int width, int dx, int dy) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y);
  const int sx = x - dx, sy = y - dy;
  if (x >= width || sx < 0 || sx >= width || sy < 0 || sy >= height) return;
  const long plane = static_cast<long>(height) * width;
  const long at = static_cast<long>(y) * width + x, src = static_cast<long>(sy) * width + sx;
#pragma unroll
  for (int c = 0; c < 3; ++c) to[c * plane + at] = from[c * plane + src];
}

template <int C, sn::Acc A>
class NativeBody final : public SpanFp8Body {
  using M = sn::Mode;
  using S = sn::Src;
  // Mémoire partagée sous 48 Ko, poids partagés compris : aucun cudaFuncSetAttribute, deux blocs par SM.
  static_assert(sn::smem_bytes(C, C, 9, true) <= 48 * 1024 && sn::smem_bytes(C, C, 9) <= 48 * 1024 &&
                sn::smem_bytes(C, sn::kPixel, 9, true) <= 48 * 1024 &&
                sn::smem_bytes(C, sn::kPixel, 9) <= 48 * 1024);
  static constexpr int kBlocks = SpanFp8Weights::kBlocks;
  static constexpr int kLayers = 2 + 3 * kBlocks;  // conv_1, les blocs, conv_cat
  // S0 : conv_1, puis (sur place) conv_cat, que to_pixel lit.
  enum Tensor : int { kS0, kK1, kX0, kX1, kT1, kT2, kTensors };

 public:
  NativeBody(int height, int width) : fh_(height / 2), fw_(width / 2) {
    grid_ = dim3(static_cast<unsigned>((fw_ + sn::kTW - 1) / sn::kTW),
                 static_cast<unsigned>((fh_ + sn::kTH - 1) / sn::kTH));
  }

  bool upload(const SpanFp8Weights& weights, SpanFp8Status& status, std::string& error) override {
    used_ = weights.blocks;
    shared_ = weights.shared_weights == SpanSharedWeights::Auto
                  ? C == 48
                  : weights.shared_weights == SpanSharedWeights::On;
    band_ = weights.band == SpanFp8Weights::kBandAuto ? auto_band(fw_, C) : weights.band;
    const auto sent = [&](cudaError_t result) {
      if (result == cudaSuccess) return true;
      static_cast<void>(cudaGetLastError());
      status = result == cudaErrorMemoryAllocation ? SpanFp8Status::OutOfMemory : SpanFp8Status::Failed;
      error = std::string("poids du SPAN FP16 natif : ") + cudaGetErrorString(result);
      return false;
    };
    const auto put = [&](Conv& to, const SpanFp8ConvView& view, int cout, int taps, int cin) {
      const std::vector<std::uint32_t> packed = sn::pack(view.weights, cout, taps, cin);
      return sent(to.w.upload(packed.data(), packed.size() * sizeof(std::uint32_t))) &&
             sent(to.b.upload(view.bias, static_cast<std::size_t>(cout) * sizeof(float)));
    };
    if (!put(first_, weights.first, C, 9, sn::kImageCh) || !put(cat_, weights.concat, C, 1, 4 * C) ||
        !put(pixel_, weights.pixel, sn::kPixel, 9, C))
      return false;
    for (int b = 0; b < used_; ++b) {
      for (int i = 0; i < 3; ++i)
        if (!put(conv_[b][i], weights.conv[b][i], C, 9, C)) return false;
      if (!put(attention_[b], weights.attention[b], C, 1, C)) return false;
    }
    status = SpanFp8Status::Ok;
    return true;
  }

  std::int64_t workspace_bytes() const override {
    return kTensors * align_up(std::int64_t{fh_} * fw_ * C * 2);
  }

  void bind(void* workspace) override {
    auto* bytes = static_cast<std::uint8_t*>(workspace);
    for (int i = 0; i < kTensors; ++i)
      t_[i] = reinterpret_cast<__half*>(bytes + i * align_up(std::int64_t{fh_} * fw_ * C * 2));
    bound_ = workspace != nullptr;
  }

  bool launch_body(cudaStream_t stream, const void* input, std::string& error) override {
    if (!bound_) {
      error = "zone de travail absente";
      return false;
    }
    static_cast<void>(cudaGetLastError());
    // Les tuiles qui bougent veulent la grille de l'image entière : pas de bandes avec elles.
    const int band = region_ == nullptr && band_ > 0 ? band_ : fh_;
    std::array<int, kLayers> done{};
    for (int top = 0; top < fh_; top += band)
      wave(stream, static_cast<const __half*>(input),
           Front{.fh = fh_, .end = std::min(fh_, top + band), .done = done.data()});
    return !failed(error);
  }

  bool launch_tail(cudaStream_t stream, void* output, const SpanFp8Ring& ring, std::string& error) override {
    if (!bound_) {
      error = "zone de travail absente";
      return false;
    }
    if (output == nullptr && ring.surface.object == 0) {
      error = "ni sortie ni surface";
      return false;
    }
    sn::Params p = params(pixel_, static_cast<__half*>(output));
    p.in[0] = t_[kS0];  // la sortie de conv_cat
    p.surface = ring.surface;
    p.surf_w = ring.width;
    p.surf_h = ring.height;
    p.mask = core_;  // la queue n'écrit que le cœur ; le corps a calculé sa région
    run<C, sn::kPixel, 9, 1, S::kNhwc, M::kPixel, false>(stream, p, Rows{0, fh_});
    return !failed(error);
  }

  // Une tuile de noyau : kTH × kTW pixels de demi-résolution, 2 kTH × 2 kTW pixels d'entrée. La portée : un
  // pixel de demi-résolution par 3×3 sur le plus long chemin (conv_1, trois par bloc, to_pixel ; conv_cat et
  // la 1×1 de la porte n'en ajoutent pas), comptée sur les couches réellement lancées.
  kernels::SpanTileGrid tile_grid() const override {
    const int reach = radius(9) + used_ * 3 * radius(9) + radius(1) + radius(9);
    return kernels::SpanTileGrid{.tile_w = 2 * sn::kTW,
                                 .tile_h = 2 * sn::kTH,
                                 .cols = static_cast<int>(grid_.x),
                                 .rows = static_cast<int>(grid_.y),
                                 .reach_cols = kernels::span_reach_tiles(reach, sn::kTW),
                                 .reach_rows = kernels::span_reach_tiles(reach, sn::kTH),
                                 .shift_step = 2,
                                 .scale = 2};
  }

  void set_tile_masks(const std::uint32_t* region, const std::uint32_t* core) override {
    region_ = region;
    core_ = core;
  }

  bool shift_output(cudaStream_t stream, const void* from, void* to, int dx, int dy,
                    std::string& error) override {
    if (from == nullptr || to == nullptr || from == to) {
      error = "décalage de la sortie : tampons absents ou confondus";
      return false;
    }
    const int oh = 4 * fh_, ow = 4 * fw_;
    constexpr unsigned kLine = 256;
    shift_planes<<<dim3((static_cast<unsigned>(ow) + kLine - 1) / kLine, static_cast<unsigned>(oh)), kLine, 0,
                   stream>>>(static_cast<const __half*>(from), static_cast<__half*>(to), oh, ow, dx, dy);
    if (!failed(error)) return true;
    error = "décalage de la sortie : " + error;
    return false;
  }

  SpanFp8AccChange poll_accumulator() override { return SpanFp8AccChange::None; }
  int overflow_frames() const override { return 0; }

 private:
  struct Conv {
    Buffer w, b;
  };

  // Le rayon d'une convolution de taps positions (1 ou 9).
  static constexpr int radius(int taps) { return taps == 9 ? 1 : 0; }

  sn::Params params(const Conv& conv, __half* out) const {
    sn::Params p;
    p.mask = region_;
    p.out = out;
    p.w = conv.w.template as<uint4>();
    p.bias = conv.b.template as<float>();
    p.fh = fh_;
    p.fw = fw_;
    return p;
  }

  // Une vague : conv_1 jusqu'à la ligne front.end, puis chaque couche aussi loin que la précédente le permet.
  // Ce qu'une couche lit vient d'être écrit par la précédente et tient encore en L2. Les tampons partagés
  // entre couches (T1, T2, X0 et X1 d'un bloc à l'autre) restent justes : la couche qui réécrit un tampon
  // suit de trois lignes au moins celle qui le lit encore, qui n'y revient jamais en arrière
  // (docs/span-native16.md).
  void wave(cudaStream_t stream, const __half* input, Front front) const {
    sn::Params p = params(first_, t_[kS0]);
    p.in[0] = input;
    run<sn::kImageCh, C, 9, 1, S::kImage, M::kPlain, false>(stream, p, front.next(0));
    const int last = used_ - 1;
    __half* const x[2] = {t_[kX0], t_[kX1]};
    for (int b = 0; b < used_; ++b) {
      const __half* in = b == 0 ? t_[kS0] : b == 1 ? t_[kK1] : x[b % 2];
      __half* out = b == 0 ? t_[kK1] : b == last ? x[used_ % 2] : x[(b + 1) % 2];
      sn::Params c1 = params(conv_[b][0], t_[kT1]);
      c1.in[0] = in;
      run<C, C, 9, 1, S::kNhwc, M::kPlain, true>(stream, c1, front.next(1));
      sn::Params c2 = params(conv_[b][1], t_[kT2]);
      c2.in[0] = t_[kT1];
      run<C, C, 9, 1, S::kNhwc, M::kPlain, true>(stream, c2, front.next(1));
      sn::Params c3 = params(conv_[b][2], out);
      c3.in[0] = t_[kT2];
      c3.skip = in;
      c3.w1 = attention_[b].w.template as<uint4>();
      c3.b1 = attention_[b].b.template as<float>();
      run<C, C, 9, 1, S::kNhwc, M::kGate, false>(stream, c3, front.next(1));
    }
    // Les tranches de la concaténation du SPAN : sortie de conv_1, du dernier bloc, du premier bloc, premier
    // Relu du dernier bloc (l'ordre des poids de conv_cat). La sortie s'écrit sur S0, sur place : une 1×1 ne
    // lit que ses propres pixels, et les blocs ont fini de lire S0 sur ces lignes.
    sn::Params cat = params(cat_, t_[kS0]);
    cat.in[0] = t_[kS0];
    cat.in[1] = x[used_ % 2];
    cat.in[2] = t_[kK1];
    cat.in[3] = t_[kT1];
    run<C, C, 1, 4, S::kNhwc, M::kPlain, false>(stream, cat, front.next(0));
  }

  // Les lignes rows d'une convolution ; rien si elles sont vides.
  template <int Cin, int Cout, int Taps, int Sources, sn::Src From, sn::Mode Mode, bool Relu>
  void run(cudaStream_t stream, sn::Params p, Rows rows) const {
    if (rows.from >= rows.to) return;
    p.y0 = rows.from;
    p.y1 = rows.to;
    const dim3 grid(grid_.x, static_cast<unsigned>((rows.to - rows.from + sn::kTH - 1) / sn::kTH));
    const auto smem = static_cast<std::size_t>(sn::smem_bytes(Cin, Cout, Taps, shared_));
    if (shared_)
      sn::conv_kernel<Cin, Cout, Taps, Sources, From, Mode, Relu, A, true>
          <<<grid, sn::kThreads, smem, stream>>>(p);
    else
      sn::conv_kernel<Cin, Cout, Taps, Sources, From, Mode, Relu, A><<<grid, sn::kThreads, smem, stream>>>(p);
  }

  static bool failed(std::string& error) {
    const cudaError_t result = cudaGetLastError();
    if (result == cudaSuccess) return false;
    error = std::string("lancement des noyaux FP16 natifs : ") + cudaGetErrorString(result);
    return true;
  }

  int fh_, fw_;
  dim3 grid_;
  int used_ = kBlocks;
  int band_ = 0;         // lignes de demi-résolution par bande (SpanFp8Weights::band), 0 : l'image entière
  bool shared_ = false;  // poids partagés (SpanFp8Weights::shared_weights)
  bool bound_ = false;
  const std::uint32_t* region_ = nullptr;  // tuiles qui bougent : masques d'appareil, nuls hors d'un appel
  const std::uint32_t* core_ = nullptr;
  Conv first_, cat_, pixel_;
  std::array<std::array<Conv, 3>, kBlocks> conv_;
  std::array<Conv, kBlocks> attention_;
  std::array<__half*, kTensors> t_{};
};

template <int C>
std::unique_ptr<SpanFp8Body> native_body(int height, int width, SpanNativeSums sums) {
  if (sums == SpanNativeSums::F32) return std::make_unique<NativeBody<C, sn::Acc::kF32>>(height, width);
  return std::make_unique<NativeBody<C, sn::Acc::kF16>>(height, width);
}

}  // namespace

std::unique_ptr<SpanFp8Body> make_span_fp8_body16(int height, int width, int channels, SpanNativeSums sums) {
  return channels == 48 ? native_body<48>(height, width, sums) : native_body<64>(height, width, sums);
}

}  // namespace kirari::gpu::cuda
