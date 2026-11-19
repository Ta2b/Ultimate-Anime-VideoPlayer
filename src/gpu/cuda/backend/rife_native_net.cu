// Le RIFE natif (rife_native_net.h) : les couches de chaque morceau planifiées une fois, leur forme
// (rife_native_plan.h) choisie au chargement au chronomètre, puis un lancement par couche. Les activations
// entre deux couches alternent entre deux tampons de la zone de travail ; l'entrée d'un ResConv reste donc
// lisible pour son addition résiduelle. Une couche à K découpé range ses sommes partielles dans un troisième
// tampon, après les deux.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/rife_native_layer.h"

namespace kirari::gpu::cuda {
namespace {

namespace rn = rife_native;
using rn::Config;
using rn::Kind;
using rn::Layer;

constexpr int pad16(int c) {
  return (c + 15) / 16 * 16;
}
constexpr int pad8(int c) {
  return (c + 7) / 8 * 8;
}

// Où un morceau lit et ce qu'il doit écrire : entrée ih × iw à in_ld canaux par pixel (0 : l'image en plans),
// sortie attendue out_h × out_w (celle de la transposée) ; glue_scales : pour le dernier bloc en pleine
// résolution, les échelles des blocs d'avant, dont la colle peut se refaire à la volée (vide sinon).
struct Entry {
  int ih = 0, iw = 0, in_ld = 0, out_h = 0, out_w = 0;
  bool encoder = false;
  bool finish = false;  // le dernier bloc : sa dernière couche peut faire la fin (finish_layer)
  std::vector<int> glue_scales;
  int glue_reduce = 1;
};

class Net final : public RifeNativeNet {
 public:
  Net(RifeNativePrecision precision, cudaStream_t loading) : precision_(precision), loading_(loading) {
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess)
      static_cast<void>(cudaDeviceGetAttribute(&sms_, cudaDevAttrMultiProcessorCount, device));
    sms_ = std::max(sms_, 1);
  }

  // Les couches d'un morceau, chacune réglée au banc ou reprise de tuned.
  bool plan(const RifeNativePiece& piece, Entry at, rn::Bench& bench, std::vector<RifeNativeTuned>& tuned,
            std::string& error) {
    const int index = static_cast<int>(pieces_.size());
    std::vector<Layer> layers(piece.size());
    for (std::size_t i = 0; i < piece.size(); ++i) {
      const RifeNativeConv& conv = piece[i];
      Layer& layer = layers[i];
      const bool last = i + 1 == piece.size();
      const bool image = at.encoder && i == 0;
      layer.cin = pad16(conv.cin);
      layer.in_ld = image ? 0 : at.in_ld;
      if (!image && layer.cin != at.in_ld) {
        error = "couche " + std::to_string(i) + " : " + std::to_string(conv.cin) + " canaux pour un pas de " +
                std::to_string(at.in_ld);
        return false;
      }
      // Le noyau de l'image échantillonne au pas 2 : une première couche d'un autre pas sortirait fausse.
      if (image && (conv.cin != 3 || conv.stride != 2 || conv.transposed)) {
        error = "première couche de l'encodeur : convolution de pas 2 sur 3 canaux attendue";
        return false;
      }
      layer.ih = at.ih;
      layer.iw = at.iw;
      if (conv.transposed) {
        layer.kind = at.encoder ? Kind::kPlanes4 : Kind::kPhase;
        layer.oh = at.ih;
        layer.ow = at.iw;
      } else {
        layer.kind = image ? Kind::kImage : conv.stride == 2 ? Kind::kHalf : Kind::kSame;
        layer.oh = conv.stride == 2 ? (at.ih - 1) / 2 + 1 : at.ih;
        layer.ow = conv.stride == 2 ? (at.iw - 1) / 2 + 1 : at.iw;
      }
      layer.sums = last ? precision_.head : precision_.body;
      rn::Dense dense;
      const Tuning tuning{.piece = index, .layer = static_cast<int>(i), .bench = bench, .tuned = tuned};
      if (!build(conv, layer, last, dense, error) || !tune(tuning, conv, dense, layer, error) ||
          !ends(layer, dense, at, i == 0, last, error)) {
        error = "couche " + std::to_string(i) + " : " + error;
        return false;
      }
      if (last && (2 * layer.oh != at.out_h || 2 * layer.ow != at.out_w)) {
        error = "taille de sortie de la transposée";
        return false;
      }
      at.ih = layer.oh;
      at.iw = layer.ow;
      at.in_ld = layer.out_ld;
      if (!last) peak_ = std::max(peak_, rn::align256(layer.out_bytes()));
      partial_ = std::max(partial_, layer.partial_bytes());
    }
    pieces_.push_back(std::move(layers));
    return true;
  }

  std::int64_t workspace_bytes() const override { return 2 * peak_ + partial_; }

  cudaError_t encode(cudaStream_t stream, const void* image, void* encoded, void* workspace) const override {
    return run(stream, 0, image, encoded, workspace, {}, {});
  }
  cudaError_t block(cudaStream_t stream, int k, const void* x, void* y, void* workspace,
                    const FusedGlueParams* glue, const FusedFinishParams* finish) const override {
    return run(stream, static_cast<std::size_t>(k) + 1, x, y, workspace, {},
               {.glue = glue, .finish = finish});
  }
  bool can_finish(int k, const FusedFinishParams& finish) const override {
    const auto piece = static_cast<std::size_t>(k) + 1;
    return k >= 0 && piece < pieces_.size() && !pieces_[piece].empty() &&
           rn::finish_fits(pieces_[piece].back(), finish);
  }
  bool can_glue(int k) const override {
    const auto piece = static_cast<std::size_t>(k) + 1;
    return k >= 0 && piece < pieces_.size() && !pieces_[piece].empty() && pieces_[piece][0].glue != nullptr;
  }
  bool glues(int k) const override { return can_glue(k) && !apart_.contains(k); }
  void use_glue(int k, bool on) override {
    if (on)
      apart_.erase(k);
    else
      apart_.insert(k);
  }
  int layers(int piece) const override {
    return piece >= 0 && static_cast<std::size_t>(piece) < pieces_.size()
               ? static_cast<int>(pieces_[static_cast<std::size_t>(piece)].size())
               : 0;
  }
  cudaError_t profile(cudaStream_t stream, int piece, const void* in, void* out, void* workspace,
                      std::span<const cudaEvent_t> marks, const FusedGlueParams* glue) const override {
    if (piece < 0 || marks.size() != static_cast<std::size_t>(layers(piece)) + 1)
      return cudaErrorInvalidValue;
    return run(stream, static_cast<std::size_t>(piece), in, out, workspace, marks, {.glue = glue});
  }
  std::string layer_label(int piece, int i) const override {
    if (i < 0 || i >= layers(piece)) return "?";
    const Layer& l = pieces_[static_cast<std::size_t>(piece)][static_cast<std::size_t>(i)];
    if (l.glue != nullptr && glues(piece - 1))
      return "3×3 pas 2 " + std::to_string(l.cin) + "→" + std::to_string(l.cout) + " sur " +
             std::to_string(l.ow) + "×" + std::to_string(l.oh) +
             ", entrée calculée à la volée (colle), tranche 16, tuile " + std::to_string(l.cout) +
             ", 1 ligne par warp, sommes " + rn::sums_name(l.sums);
    const std::string end =
        l.finish != nullptr ? ", fin fondue possible (première tuile de 32 seulement)" : "";
    static constexpr const char* kKinds[] = {"3×3", "3×3 pas 2", "image 3×3 pas 2", "transposée",
                                             "transposée en plans"};
    return std::string(kKinds[static_cast<int>(l.kind)]) + " " + std::to_string(l.cin) + "→" +
           std::to_string(l.cout) + " sur " + std::to_string(l.ow) + "×" + std::to_string(l.oh) +
           ", tranche " + std::to_string(l.cfg.kc) + ", tuile " + std::to_string(l.cfg.ct) + ", " +
           std::to_string(l.cfg.rows) + (l.cfg.rows > 1 ? " lignes" : " ligne") + " par warp, K en " +
           std::to_string(l.cfg.ksplit) + (l.cfg.ahead ? ", poids à l'avance" : "") + ", sommes " +
           rn::sums_name(l.sums) +
           (l.rejected > 0 ? ", " + std::to_string(l.rejected) + " formes écartées (sortie différente)"
                           : "") +
           end;
  }

 private:
  // Biais et pente envoyés, poids en FP16 ; les dimensions de la sortie.
  bool build(const RifeNativeConv& conv, Layer& layer, bool last, rn::Dense& dense, std::string& error) {
    if (layer.kind == Kind::kPlanes4 && conv.cout != 4) {
      error = "transposée de l'encodeur : 4 canaux attendus";
      return false;
    }
    layer.residual = !conv.scale.empty();
    layer.activation = !conv.slope.empty();
    dense = rn::dense_weights(conv, layer.kind, layer.cin);
    layer.phases = dense.phases;
    layer.cout = dense.cout;
    if (layer.kind == Kind::kPlanes4) {
      layer.out_ld = 0;
      layer.out_ch = 4;
    } else {
      layer.out_ld = last ? pad8(conv.cout) : layer.cout;
      layer.out_ch = layer.out_ld;
    }
    if (layer.activation) {
      std::vector<float> slope(static_cast<std::size_t>(pad16(conv.cout)), 0.F);
      std::copy(conv.slope.begin(), conv.slope.end(), slope.begin());
      if (!sent(layer.slope.upload(slope.data(), slope.size() * sizeof(float), loading_), error))
        return false;
    }
    return sent(layer.bias.upload(dense.bias.data(), dense.bias.size() * sizeof(float), loading_), error);
  }

  static bool sent(cudaError_t result, std::string& error) {
    if (result == cudaSuccess) return true;
    static_cast<void>(cudaGetLastError());
    error = std::string("poids du RIFE natif : ") + cudaGetErrorString(result);
    return false;
  }

  bool upload_weights(rn::Buffer& to, const rn::Dense& dense, int ct, std::string& error) const {
    const std::vector<std::uint32_t> packed = rn::packed_weights(dense, ct);
    return sent(to.upload(packed.data(), packed.size() * sizeof(std::uint32_t), loading_), error);
  }

  // La première couche d'un bloc dont l'entrée se calcule à la volée (scales : les échelles des blocs
  // d'avant, vide sinon ; reduce : celle du bloc, 1 ou 2) : son noyau, s'il en est un pour ses sommes et ses
  // canaux (X de 28 canaux, 32 rangés ; une convolution de pas 2 sans résiduel, toute en une tuile), et ses
  // poids rangés pour lui.
  bool glue_layer(Layer& layer, const rn::Dense& dense, const std::vector<int>& scales, int reduce,
                  std::string& error) const {
    if (scales.empty() || layer.kind != Kind::kHalf || layer.cin != 32 || layer.residual) return true;
    // Le noyau est dimensionné pour des sources plus grossières que X d'un facteur 2 au moins (au plus 15 Ko
    // de relèvement en pleine résolution, 10 Ko à moitié) : un découpage qui en aurait une plus fine garde la
    // colle à part. La borne de 48 Ko sur toute sa mémoire partagée (tuile et relèvement), que ces échelles
    // tiennent déjà, n'est qu'une garde : le lancement ne demande pas plus que la part par défaut.
    if (std::ranges::any_of(scales, [reduce](int d) { return d < 2 * reduce; }) ||
        rn::glue_smem(scales, reduce) > 48 * 1024)
      return true;
    layer.glue = rn::pick_glue(layer.sums, static_cast<int>(scales.size()), reduce, layer.cout);
    if (layer.glue == nullptr) return true;
    layer.glue_scales = scales;
    layer.glue_reduce = reduce;
    layer.glue_grid = dim3(static_cast<unsigned>((layer.ow + rn::kTW - 1) / rn::kTW),
                           static_cast<unsigned>((layer.oh + rn::kWarps - 1) / rn::kWarps), 1U);
    layer.glue_smem = rn::glue_smem(scales, reduce);
    return upload_weights(layer.glue_w, dense, layer.cout, error);
  }

  // Les noyaux d'une première ou d'une dernière couche qui finissent autrement : la colle à la volée
  // (glue_layer), la fin dans l'épilogue (finish_layer).
  bool ends(Layer& layer, const rn::Dense& dense, const Entry& at, bool first, bool last,
            std::string& error) const {
    return (!first || glue_layer(layer, dense, at.glue_scales, at.glue_reduce, error)) &&
           (!last || !at.finish || finish_layer(layer, dense, error));
  }

  // La dernière couche du dernier bloc : son noyau qui fait la fin dans son épilogue, s'il en est un pour la
  // forme réglée (même tranche et mêmes lignes par warp, sans découpe de K), et ses poids rangés pour une
  // tuile de 32 (le flot et le masque sont dans la première).
  bool finish_layer(Layer& layer, const rn::Dense& dense, std::string& error) const {
    if (layer.kind != Kind::kPhase || layer.cfg.ksplit != 1 || layer.cout % 32 != 0 || layer.out_ld < 20)
      return true;
    layer.finish = rn::pick_finish(layer.sums, layer.cfg.kc, layer.cfg.rows);
    if (layer.finish == nullptr) return true;
    layer.finish_tiles = layer.cout / 32;
    const int th = rn::kWarps * layer.cfg.rows;
    layer.finish_grid = dim3(static_cast<unsigned>((layer.ow + rn::kTW - 1) / rn::kTW),
                             static_cast<unsigned>((layer.oh + th - 1) / th), 4U);
    layer.finish_smem = rn::smem_bytes(layer.cfg.kc, 32, rn::kernel_size(layer.kind), 1, layer.cfg.rows);
    if (layer.finish_smem > 48 * 1024) {  // le lancement ne demande pas plus que la part par défaut
      layer.finish = nullptr;
      return true;
    }
    return upload_weights(layer.finish_w, dense, 32, error);
  }

  static bool same_shape(const Config& a, const Config& b) {
    return a.kc == b.kc && a.ct == b.ct && a.rows == b.rows && a.ksplit == b.ksplit && a.ahead == b.ahead;
  }

  struct Tuning {
    int piece = 0, layer = 0;
    rn::Bench& bench;
    std::vector<RifeNativeTuned>& tuned;
  };

  // La forme de la couche : celle que tuned garde déjà pour ces sommes, sinon la plus rapide au banc parmi
  // celles qui rendent la sortie de la forme de référence (tranche et tuile de 16, sans découpe de K : la
  // plus simple), chacune avec ses poids rangés pour sa tuile.
  bool tune(const Tuning& at, const RifeNativeConv& conv, const rn::Dense& dense, Layer& layer,
            std::string& error) const {
    const auto known = std::ranges::find_if(at.tuned, [&](const RifeNativeTuned& t) {
      return t.piece == at.piece && t.layer == at.layer && t.sums == layer.sums;
    });
    if (known != at.tuned.end() && shape(layer, Config{.kc = known->kc,
                                                       .ct = known->ct,
                                                       .rows = known->rows,
                                                       .ksplit = known->ksplit,
                                                       .ahead = known->ahead}))
      return upload_weights(layer.w, dense, layer.cfg.ct, error);
    std::array<rn::Buffer, 3> packs;  // tuiles de 16, 32 et 64
    const auto packed = [&](int ct) -> const void* {
      rn::Buffer& pack = packs[ct == 16 ? 0 : ct == 32 ? 1 : 2];
      if (pack.as<void>() == nullptr && !upload_weights(pack, dense, ct, error)) return nullptr;
      return pack.as<void>();
    };
    const Config reference{.kc = 16, .ct = 16, .rows = rn::stride_of(layer.kind) == 2 ? 1 : 2, .ksplit = 1};
    if (!shape(layer, reference)) {
      error = "pas de forme de référence";
      return false;
    }
    const void* w = packed(16);
    if (w == nullptr || !at.bench.reference(layer, w, error)) return false;
    const rn::Extent extent{
        .cin = layer.cin, .cout = layer.cout, .oh = layer.oh, .ow = layer.ow, .phases = layer.phases};
    Config best = reference;
    float fastest = std::numeric_limits<float>::infinity();
    layer.rejected = 0;
    for (const Config& cfg : rn::configs(layer.sums, layer.kind, extent, sms_)) {
      const bool simplest = same_shape(cfg, reference);
      if (!shape(layer, cfg)) continue;
      w = packed(cfg.ct);
      if (w == nullptr) return false;
      const std::optional<rn::Bench::Timing> timed = at.bench.time(layer, w, error);
      if (!timed) return false;
      if (timed->differs && !simplest) {
        ++layer.rejected;
        continue;
      }
      if (timed->ms < fastest) {
        fastest = timed->ms;
        best = cfg;
      }
    }
    if (fastest == std::numeric_limits<float>::infinity() || !shape(layer, best)) {
      error = "aucun noyau ne fait la couche " + std::to_string(conv.cin) + "→" + std::to_string(conv.cout);
      return false;
    }
    at.tuned.push_back({.piece = at.piece,
                        .layer = at.layer,
                        .sums = layer.sums,
                        .kc = best.kc,
                        .ct = best.ct,
                        .rows = best.rows,
                        .ksplit = best.ksplit,
                        .ahead = best.ahead});
    return upload_weights(layer.w, dense, best.ct, error);
  }

  // Ce que block fait à la place de la première ou de la dernière couche : la colle (la première couche la
  // refait, rn::glue_fits), la fin (la dernière la fait, rn::finish_fits ; Y n'est pas écrit). Nuls : rien.
  struct Ends {
    const FusedGlueParams* glue = nullptr;
    const FusedFinishParams* finish = nullptr;
  };

  // Les couches d'un morceau ; marks (vide hors profil) : un événement avant la première et après chacune.
  cudaError_t run(cudaStream_t stream, std::size_t piece, const void* in, void* out, void* workspace,
                  std::span<const cudaEvent_t> marks, Ends ends) const {
    if (piece >= pieces_.size() || pieces_[piece].empty()) return cudaErrorInvalidValue;
    const std::vector<Layer>& layers = pieces_[piece];
    const FusedGlueParams* const glue = ends.glue;
    if (glue != nullptr && !rn::glue_fits(layers[0], *glue)) return cudaErrorInvalidValue;
    if (ends.finish != nullptr && !rn::finish_fits(layers.back(), *ends.finish)) return cudaErrorInvalidValue;
    auto* ws = static_cast<std::uint8_t*>(workspace);
    __half* const buffers[2] = {reinterpret_cast<__half*>(ws), reinterpret_cast<__half*>(ws + peak_)};
    auto* const partial = reinterpret_cast<float*>(ws + 2 * peak_);
    const auto mark = [&](std::size_t i) {
      return marks.empty() ? cudaSuccess
                           : cudaEventRecordWithFlags(marks[i], stream, cudaEventRecordExternal);
    };
    cudaError_t result = mark(0);
    const __half* from = static_cast<const __half*>(in);
    for (std::size_t i = 0; i < layers.size() && result == cudaSuccess; ++i) {
      __half* to = i + 1 == layers.size() ? static_cast<__half*>(out) : buffers[i % 2];
      if (i == 0 && glue != nullptr) {
        const Layer& l = layers[0];
        l.glue(l.glue_grid, l.glue_smem, stream, rn::params(l, from, to, partial, l.glue_w.as<const void>()),
               *glue);
      } else if (i + 1 == layers.size() && ends.finish != nullptr) {
        const Layer& l = layers[i];
        rn::Params p = rn::params(l, from, to, partial, l.finish_w.as<const void>());
        p.tiles = l.finish_tiles;
        l.finish(l.finish_grid, l.finish_smem, stream, p, with_window(*ends.finish));
      } else {
        rn::enqueue(layers[i], stream,
                    rn::params(layers[i], from, to, partial, layers[i].w.as<const void>()));
      }
      from = to;
      result = mark(i + 1);
    }
    return result != cudaSuccess ? result : cudaGetLastError();
  }

  RifeNativePrecision precision_;
  cudaStream_t loading_;  // la voie du chargement : envois et banc
  int sms_ = 1;
  std::int64_t peak_ = 0;
  std::int64_t partial_ = 0;  // sommes partielles de la plus grande couche à K découpé
  std::set<int> apart_;       // les blocs qui gardent la colle à part (use_glue)
  std::vector<std::vector<Layer>> pieces_;
};

}  // namespace

std::unique_ptr<RifeNativeNet> RifeNativeNet::create(const std::vector<RifeNativePiece>& pieces,
                                                     const std::vector<int>& reduce, int height, int width,
                                                     RifeNativePrecision precision, cudaStream_t stream,
                                                     std::vector<RifeNativeTuned>& tuned,
                                                     std::string& error) {
  if (pieces.size() != reduce.size() + 1 || pieces.empty()) {
    error = "un encodeur et un morceau par bloc attendus";
    return nullptr;
  }
  rn::Bench bench(stream);
  if (!bench.ready()) {
    error = "événements CUDA du réglage";
    return nullptr;
  }
  auto net = std::make_unique<Net>(precision, stream);
  const Entry image{.ih = height,
                    .iw = width,
                    .in_ld = 0,
                    .out_h = height,
                    .out_w = width,
                    .encoder = true,
                    .glue_scales = {}};
  if (!net->plan(pieces[0], image, bench, tuned, error)) {
    error = "encodeur : " + error;
    return nullptr;
  }
  for (std::size_t k = 0; k < reduce.size(); ++k) {
    const int d = reduce[k];
    const int channels = k == 0 ? 15 : 28;
    Entry at{.ih = height / d,
             .iw = width / d,
             .in_ld = pad8(channels),
             .out_h = height / (2 * d),
             .out_w = width / (2 * d),
             .encoder = false,
             .glue_scales = {}};
    at.finish = k + 1 == reduce.size();
    // L'entrée calculée à la volée : le dernier bloc en pleine résolution, l'avant-dernier à moitié.
    const bool last = k + 1 == reduce.size();
    if (k > 0 && ((last && d == 1) || (k + 2 == reduce.size() && d == 2))) {
      at.glue_scales.assign(reduce.begin(), reduce.begin() + static_cast<std::ptrdiff_t>(k));
      at.glue_reduce = d;
    }
    if (!net->plan(pieces[k + 1], at, bench, tuned, error)) {
      error = "bloc " + std::to_string(k) + " : " + error;
      return nullptr;
    }
  }
  // Les envois sur stream finis avant qu'une autre voie (la lecture) lance le réseau.
  if (const cudaError_t result = cudaStreamSynchronize(stream); result != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    error = std::string("chargement : ") + cudaGetErrorString(result);
    return nullptr;
  }
  return net;
}

}  // namespace kirari::gpu::cuda
