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
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/span_fp8_body.h"
#include "gpu/cuda/backend/span_native_conv.cuh"
#include "gpu/cuda/backend/span_native_gate_cat.cuh"
#include "gpu/cuda/backend/span_native_wino.cuh"
#include "gpu/cuda/backend/span_native_wino_vertical.cuh"

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
  Rows next(int radius, int quantum = 1) {
    const int available = std::max(0, end - radius);
    const int to = std::max(done[layer], end >= fh ? fh : available / quantum * quantum);
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

bool ada_device() {
  int device = 0;
  cudaDeviceProp props{};
  if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&props, device) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return false;
  }
  return props.major == 8 && props.minor == 9;
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
  static constexpr int kLayers = kSpanBodyLayers;  // conv_1, les blocs, conv_cat
  // S0 : conv_1, puis (sur place) conv_cat, que to_pixel lit.
  enum Tensor : int { kS0, kK1, kX0, kX1, kT1, kT2, kTensors };

 public:
  NativeBody(int height, int width) : fh_(height / 2), fw_(width / 2) {
    grid_ = dim3(static_cast<unsigned>((fw_ + sn::kTW - 1) / sn::kTW),
                 static_cast<unsigned>((fh_ + sn::kTH - 1) / sn::kTH));
  }

  bool upload(const SpanFp8Weights& weights, SpanFp8Status& status, std::string& error) override {
    configure(weights);
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
    const auto put_wino = [&](Conv& to, const SpanFp8ConvView& view) {
      const std::vector<std::uint32_t> packed =
          vertical_ ? sn::vertical::pack<C>(view.weights) : sn::pack_wino(view.weights, C, C);
      return sent(to.w.upload(packed.data(), packed.size() * sizeof(std::uint32_t))) &&
             sent(to.b.upload(view.bias, static_cast<std::size_t>(C) * sizeof(float)));
    };
    if (!put(first_, weights.first, C, 9, sn::kImageCh) || !put(cat_, weights.concat, C, 1, 4 * C) ||
        !put(pixel_, weights.pixel, sn::kPixel, 9, C))
      return false;
    for (int b = 0; b < used_; ++b) {
      for (int i = 0; i < 3; ++i) {
        const bool use_winograd = (wino_ || vertical_) && !(second_direct_ && i == 1);
        if (!(use_winograd ? put_wino(conv_[b][i], weights.conv[b][i])
                           : put(conv_[b][i], weights.conv[b][i], C, 9, C)))
          return false;
      }
      if (!put(attention_[b], weights.attention[b], C, 1, C)) return false;
    }
    status = SpanFp8Status::Ok;
    return true;
  }

  std::int64_t workspace_bytes() const override {
    return tensor_bytes(fh_) + (kTensors - 1) * tensor_bytes(rolling_rows_ ? rolling_rows_ : fh_);
  }

  void bind(void* workspace) override {
    auto* bytes = static_cast<std::uint8_t*>(workspace);
    t_[kS0] = reinterpret_cast<__half*>(bytes);
    const auto full = tensor_bytes(fh_);
    const auto step = tensor_bytes(rolling_rows_ ? rolling_rows_ : fh_);
    for (int i = 1; i < kTensors; ++i) t_[i] = reinterpret_cast<__half*>(bytes + full + (i - 1) * step);
    bound_ = workspace != nullptr;
  }

  bool launch_body(cudaStream_t stream, const void* input, std::string& error) override {
    if (!bound_) {
      error = "zone de travail absente";
      return false;
    }
    static_cast<void>(cudaGetLastError());
    // En zone compacte, les blocs masqués restent ancrés à la grille globale.
    const bool bands = (region_ == nullptr || rolling_rows_ != 0) && band_ > 0;
    const int band = bands ? band_ : fh_;
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
    // Après les arrondis de Winograd, chaque paire lit jusqu'à deux lignes de son voisinage.
    const int vertical_reach = vertical_ ? reach + used_ * 3 : reach;
    return kernels::SpanTileGrid{.tile_w = 2 * sn::kTW,
                                 .tile_h = 2 * sn::kTH,
                                 .cols = static_cast<int>(grid_.x),
                                 .rows = static_cast<int>(grid_.y),
                                 .reach_cols = kernels::span_reach_tiles(reach, sn::kTW),
                                 .reach_rows = kernels::span_reach_tiles(vertical_reach, sn::kTH),
                                 .shift_step = 2,
                                 .scale = 2,
                                 .shift_step_y = vertical_ ? 4 : 0};
  }

  void pack_tile_layers(std::span<const std::uint8_t> core, std::uint32_t* body) const override {
    const auto grid = tile_grid();
    const int layers = 2 + 3 * used_;
    int previous_x = -1, previous_y = -1;
    for (int layer = 0; layer < layers; ++layer) {
      // conv_1, les 3×3 des blocs, conv_cat. Après conv_cat, seul to_pixel lit un halo.
      const int ahead = std::max(0, 3 * used_ - layer);
      const int rx = kernels::span_reach_tiles(1 + ahead, sn::kTW);
      const int ry = kernels::span_reach_tiles(1 + ahead * (vertical_ ? 2 : 1), sn::kTH);
      auto next = std::span(body + static_cast<std::size_t>(layer) * kernels::kSpanTileMaskWords,
                            kernels::kSpanTileMaskWords);
      if (rx == previous_x && ry == previous_y)
        std::copy_n(next.data() - kernels::kSpanTileMaskWords, kernels::kSpanTileMaskWords, next.data());
      else {
        const auto region = kernels::dilate_span_tiles(core, grid.cols, grid.rows, rx, ry);
        kernels::pack_span_tiles(region, next);
      }
      previous_x = rx;
      previous_y = ry;
    }
  }

  void set_tile_masks(const std::uint32_t* region, const std::uint32_t* core, bool layered) override {
    layered_masks_ = layered && !wino_;
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
  void configure(const SpanFp8Weights& weights) {
    used_ = weights.blocks;
    shared_ = weights.shared_weights == SpanSharedWeights::Auto
                  ? C == 48
                  : weights.shared_weights == SpanSharedWeights::On;
    wino_ = C == sn::kWinoC && A == sn::Acc::kF16 && weights.winograd == SpanWinograd::On;
    second_direct_ = C == 48 && A == sn::Acc::kF16 && !weights.compact_workspace &&
                     weights.winograd == SpanWinograd::VerticalSecondDirect;
    vertical_ = (C == 64 || C == 48) && A == sn::Acc::kF16 && !weights.compact_workspace &&
                (weights.winograd == SpanWinograd::Vertical ||
                 weights.winograd == SpanWinograd::VerticalUnrolled || second_direct_);
    unrolled_ = weights.winograd == SpanWinograd::VerticalUnrolled;
    if constexpr (A == sn::Acc::kF16 && C == 64) {
      // La porte à 64 canaux réemploie l'épilogue sur le chemin mesuré en 1080p sur Ada.
      const bool tuned = used_ == 8 && fw_ == 960 && fh_ == 540 && ada_device();
      gate_cache_ = tuned && !shared_ && !wino_ && !vertical_;
      vertical_gate_cat_ = tuned && vertical_ && unrolled_;
    }
    if constexpr (A == sn::Acc::kF16) {
      const char* fuse = std::getenv("KIRARI_AB_SPAN_GATE_CAT");  // env: banc
      if (fuse != nullptr && std::string_view(fuse) == "0") vertical_gate_cat_ = false;
      const bool compatible = C == 48 ? used_ == 5 : used_ == 8 && !shared_;
      gate_cat_ = compatible && !wino_ && !vertical_ && fuse != nullptr && std::string_view(fuse) == "1";
    }
    configure_bands(weights);
  }

  void configure_bands(const SpanFp8Weights& weights) {
    band_ = weights.band == SpanFp8Weights::kBandAuto ? auto_band(fw_, C) : weights.band;
    if constexpr (C == 64 && A == sn::Acc::kF16) {
      if (weights.compact_workspace && !shared_ && !wino_ && !vertical_ && !gate_cat_ && band_ > 0 &&
          band_ < fh_) {
        const int depth = static_cast<int>(std::bit_ceil(static_cast<unsigned>(band_ + 3 * used_ + 2)));
        if (depth < fh_) rolling_rows_ = depth;
      }
    }
  }

  struct Conv {
    Buffer w, b;
  };

  std::int64_t tensor_bytes(int rows) const { return align_up(std::int64_t{rows} * fw_ * C * 2); }

  // Le rayon d'une convolution de taps positions (1 ou 9).
  static constexpr int radius(int taps) { return taps == 9 ? 1 : 0; }

  sn::Params params(const Conv& conv, __half* out, int layer = 0) const {
    sn::Params p;
    p.mask = region_;
    if (region_ != nullptr && layered_masks_)
      p.mask += static_cast<std::size_t>(layer) * kernels::kSpanTileMaskWords;
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
    const int rolling = rolling_rows_ - 1;
    __half* const x[2] = {t_[kX0], t_[kX1]};
    sn::Params cat = params(cat_, t_[kS0], 3 * used_ + 1);
    cat.in[0] = t_[kS0];
    cat.in[1] = x[used_ % 2];
    cat.in[2] = t_[kK1];
    cat.in[3] = t_[kT1];
    cat.input_row_mask[1] = cat.input_row_mask[2] = cat.input_row_mask[3] = rolling;
    const bool fused = (gate_cat_ || vertical_gate_cat_) && region_ == nullptr;
    const int quantum =
        vertical_ ? 2 : 1;  // la paire Winograd doit avoir toutes ses entrées avant le lancement
    for (int b = 0; b < used_; ++b) {
      const __half* in = b == 0 ? t_[kS0] : b == 1 ? t_[kK1] : x[b % 2];
      __half* out = b == 0 ? t_[kK1] : b == last ? x[used_ % 2] : x[(b + 1) % 2];
      sn::Params c1 = params(conv_[b][0], t_[kT1], 3 * b + 1);
      c1.in[0] = in;
      c1.input_row_mask[0] = b == 0 ? -1 : rolling;
      c1.output_row_mask = rolling;
      block<M::kPlain, true>(stream, c1, front.next(1, quantum));
      sn::Params c2 = params(conv_[b][1], t_[kT2], 3 * b + 2);
      c2.in[0] = t_[kT1];
      c2.input_row_mask[0] = c2.output_row_mask = rolling;
      const Rows second_rows = front.next(1, quantum);
      if (second_direct_)
        run<C, C, 9, 1, S::kNhwc, M::kPlain, true>(stream, c2, second_rows);
      else
        block<M::kPlain, true>(stream, c2, second_rows);
      sn::Params c3 = params(conv_[b][2], out, 3 * b + 3);
      c3.in[0] = t_[kT2];
      c3.skip = in;
      c3.skip_row_mask = b == 0 ? -1 : rolling;
      c3.input_row_mask[0] = c3.output_row_mask = rolling;
      c3.w1 = attention_[b].w.template as<uint4>();
      c3.b1 = attention_[b].b.template as<float>();
      const Rows rows = front.next(1, quantum);
      if constexpr (A == sn::Acc::kF16) {
        if (fused && b == last) {
          const Rows cat_rows = front.next(0);  // exactement le même front que celui de la porte
          if constexpr (C == 64) {
            if (vertical_gate_cat_) {
              if (cat_rows.from < cat_rows.to) {
                c3.y0 = cat.y0 = cat_rows.from;
                c3.y1 = cat.y1 = cat_rows.to;
                const int first = cat_rows.from & ~(sn::vertical::kHeight - 1);
                const auto blocks =
                    static_cast<unsigned>((cat_rows.to - 1 - first) / sn::vertical::kHeight + 1);
                sn::vertical::gate_cat<C>
                    <<<dim3(grid_.x, blocks), sn::vertical::kWorkers<C>, sn::vertical::kShared<C>, stream>>>(
                        c3, cat);
              }
              continue;
            }
          }
          if (cat_rows.from < cat_rows.to) {
            cat.y0 = cat_rows.from;
            cat.y1 = cat_rows.to;
            const dim3 grid(grid_.x,
                            static_cast<unsigned>((cat_rows.to - cat_rows.from + sn::kTH - 1) / sn::kTH));
            if constexpr (C == 48) {
              if (shared_)
                sn::gate_cat_kernel<C, true>
                    <<<grid, sn::kThreads, sn::smem_bytes(C, C, 9, true), stream>>>(c3, cat);
              else
                sn::gate_cat_kernel<C><<<grid, sn::kThreads, sn::smem_bytes(C, C, 9), stream>>>(c3, cat);
            } else {
              sn::gate_cat_kernel<C><<<grid, sn::kThreads, sn::smem_bytes(C, C, 9), stream>>>(c3, cat);
            }
          }
          continue;
        }
      }
      block<M::kGate, false>(stream, c3, rows);
    }
    // Les tranches de la concaténation du SPAN : sortie de conv_1, du dernier bloc, du premier bloc, premier
    // Relu du dernier bloc (l'ordre des poids de conv_cat). La sortie s'écrit sur S0, sur place : une 1×1 ne
    // lit que ses propres pixels, et les blocs ont fini de lire S0 sur ces lignes.
    if (!fused) run<C, C, 1, 4, S::kNhwc, M::kPlain, false>(stream, cat, front.next(0));
  }

  // Les lignes rows d'une convolution ; rien si elles sont vides.
  template <int Cin, int Cout, int Taps, int Sources, sn::Src From, sn::Mode Mode, bool Relu>
  void run(cudaStream_t stream, sn::Params p, Rows rows) const {
    if (rows.from >= rows.to) return;
    p.y0 = rows.from;
    p.y1 = rows.to;
    const int first = rolling_rows_ != 0 && p.mask != nullptr ? rows.from & ~(sn::kTH - 1) : rows.from;
    const dim3 grid(grid_.x, static_cast<unsigned>((rows.to - first + sn::kTH - 1) / sn::kTH));
    const auto smem = static_cast<std::size_t>(sn::smem_bytes(Cin, Cout, Taps, shared_));
    if constexpr (Cout == 64 && A == sn::Acc::kF16 && Mode != M::kPixel) {
      if (rolling_rows_ != 0 && (From == S::kNhwc || p.mask != nullptr)) {
        if constexpr (Mode == M::kGate) {
          if (gate_cache_) {
            sn::conv_kernel<Cin, Cout, Taps, Sources, From, Mode, Relu, A, false, true, true>
                <<<grid, sn::kThreads, smem, stream>>>(p);
            return;
          }
        }
        sn::conv_kernel<Cin, Cout, Taps, Sources, From, Mode, Relu, A, false, false, true>
            <<<grid, sn::kThreads, smem, stream>>>(p);
        return;
      }
    }
    if constexpr (Cin == 64 && Cout == 64 && A == sn::Acc::kF16 && Mode == sn::Mode::kGate) {
      if (gate_cache_) {
        sn::conv_kernel<Cin, Cout, Taps, Sources, From, Mode, Relu, A, false, true>
            <<<grid, sn::kThreads, smem, stream>>>(p);
        return;
      }
    }
    if (shared_)
      sn::conv_kernel<Cin, Cout, Taps, Sources, From, Mode, Relu, A, true>
          <<<grid, sn::kThreads, smem, stream>>>(p);
    else
      sn::conv_kernel<Cin, Cout, Taps, Sources, From, Mode, Relu, A><<<grid, sn::kThreads, smem, stream>>>(p);
  }

  // Une 3×3 carrée d’un bloc : directe, Winograd sur les deux axes ou Winograd sur les lignes.
  template <sn::Mode Mode, bool Relu>
  void block(cudaStream_t stream, sn::Params p, Rows rows) const {
    if constexpr (A == sn::Acc::kF16) {
      if (vertical_) {
        if (rows.from >= rows.to) return;
        p.y0 = rows.from;
        p.y1 = rows.to;
        const int first = rows.from & ~(sn::vertical::kHeight - 1);
        const dim3 grid(grid_.x, static_cast<unsigned>((rows.to - 1 - first) / sn::vertical::kHeight + 1));
        if constexpr (C == 64) {
          if (unrolled_) {
            sn::vertical::convolution<C, Mode == M::kGate, true>
                <<<grid, sn::vertical::kWorkers<C>, sn::vertical::kShared<C>, stream>>>(p);
            return;
          }
        }
        sn::vertical::convolution<C, Mode == M::kGate>
            <<<grid, sn::vertical::kWorkers<C>, sn::vertical::kShared<C>, stream>>>(p);
        return;
      }
    }
    if constexpr (C == sn::kWinoC && A == sn::Acc::kF16) {
      if (wino_) {
        if (rows.from >= rows.to) return;
        p.y0 = rows.from;
        p.y1 = rows.to;
        const int first = sn::wino_first_row(rows.from);
        const dim3 grid(grid_.x, static_cast<unsigned>((rows.to - first + sn::kTH - 1) / sn::kTH));
        sn::wino_kernel<Mode, Relu>
            <<<grid, sn::kThreads, static_cast<std::size_t>(sn::wino_smem_bytes()), stream>>>(p);
        return;
      }
    }
    run<C, C, 9, 1, S::kNhwc, Mode, Relu>(stream, p, rows);
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
  int rolling_rows_ = 0;  // banc : lignes allouées pour chacun des cinq temporaires
  int band_ = 0;          // lignes de demi-résolution par bande (SpanFp8Weights::band), 0 : l'image entière
  bool shared_ = false;   // poids partagés (SpanFp8Weights::shared_weights)
  bool layered_masks_ = false;
  bool unrolled_ = false;
  bool second_direct_ = false;      // compromis vérifié sur Performance Sharp1
  bool vertical_ = false;           // Winograd le long des paires de lignes
  bool wino_ = false;               // les 3×3 des blocs par Winograd (SpanFp8Weights::winograd)
  bool gate_cache_ = false;         // conserver v float2 entre 1×1 et produit de la porte
  bool vertical_gate_cat_ = false;  // dernière porte et concaténation dans la même tuile
  bool gate_cat_ = false;           // banc de la porte finale suivie de conv_cat
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
