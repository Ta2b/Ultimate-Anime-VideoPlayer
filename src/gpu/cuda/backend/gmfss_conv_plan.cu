// La planification des réseaux natifs de GMFSS (gmfss_conv_plan.cuh) : les tenseurs placés par durée de vie,
// puis chaque convolution rangée (rn::Dense, les poids de chaque tuile par rn::packed_weights) et réglée au
// banc du RIFE natif pour les sommes f16, mixtes et f32 (gmfss_conv_tune.cu), comme celles du RIFE natif
// (rife_native_net.cu). Une convolution qui n'a qu'une forme n'est pas mesurée ; une convolution de même
// noyau et de même géométrie qu'une autre déjà réglée (les blocs d'une ligne du GridNet) en reprend les
// formes.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/gmfss_conv_plan.cuh"

namespace kirari::gpu::cuda::gmfss_plan {
namespace {

using rn::Kind;

Kind kind_of(const prepare::GmfssGridStep& step, bool image) {
  if (image) return Kind::kImage;
  if (step.shuffle) return Kind::kShuffle;
  if (step.conv.transposed) return Kind::kPhase;
  return step.conv.stride == 2 ? Kind::kHalf : Kind::kSame;
}

const char* kind_name(Kind kind) {
  switch (kind) {
    case Kind::kSame:
      return "3×3";
    case Kind::kHalf:
      return "3×3 pas 2";
    case Kind::kImage:
      return "image 3×3 pas 2";
    case Kind::kPhase:
      return "transposée";
    case Kind::kShuffle:
      return "3×3 et PixelShuffle";
    case Kind::kPlanes4:
      break;
  }
  return "?";
}

// Le poids (o, prise, c) d'une convolution, zéro dans le rembourrage des canaux.
float weight_of(const prepare::GmfssGridConv& conv, int o, int tap, int c) {
  if (o >= conv.cout || c >= conv.cin) return 0.F;
  const int taps = conv.transposed ? 16 : 9;
  return conv.weights[(static_cast<std::size_t>(o) * taps + tap) * conv.cin + c];
}

// Les poids FP16 [phase][cout][prise][cin] et les biais [phase][cout] d'une convolution, que
// rn::packed_weights range pour chaque tuile. Transposée : prises (ty, tx) aux décalages py - 1 + ty, poids
// (3 - py - 2 ty, 3 - px - 2 tx). PixelShuffle : le canal c de la phase est le canal c · 4 + phase.
rn::Dense dense_of(const prepare::GmfssGridStep& step, const Layer& layer) {
  const prepare::GmfssGridConv& conv = step.conv;
  rn::Dense d;
  d.phases = layer.phases;
  d.cout = layer.cout;
  d.taps = layer.kind == Kind::kPhase ? 4 : 9;
  d.cin = layer.cin;
  const int real_out = step.shuffle ? conv.cout / 4 : conv.cout;
  const auto source = [&](int ph, int o) { return step.shuffle ? o * 4 + ph : o; };
  d.w.reserve(static_cast<std::size_t>(d.phases) * static_cast<std::size_t>(d.cout) *
              static_cast<std::size_t>(d.taps) * static_cast<std::size_t>(d.cin));
  for (int ph = 0; ph < d.phases; ++ph) {
    const int py = ph >> 1, px = ph & 1;
    for (int o = 0; o < d.cout; ++o)
      for (int tap = 0; tap < d.taps; ++tap) {
        const int at =
            layer.kind == Kind::kPhase ? (3 - py - 2 * (tap >> 1)) * 4 + (3 - px - 2 * (tap & 1)) : tap;
        for (int c = 0; c < d.cin; ++c)
          d.w.push_back(__float2half_rn(o < real_out ? weight_of(conv, source(ph, o), at, c) : 0.F));
      }
    for (int o = 0; o < d.cout; ++o)
      d.bias.push_back(o < real_out ? conv.bias[static_cast<std::size_t>(source(ph, o))] : 0.F);
  }
  return d;
}

bool same_geometry(const Layer& a, const Layer& b) {
  return a.kind == b.kind && a.cin == b.cin && a.cout == b.cout && a.ih == b.ih && a.iw == b.iw &&
         a.oh == b.oh && a.ow == b.ow && a.in_ld == b.in_ld && a.out_ld == b.out_ld &&
         (a.skip >= 0) == (b.skip >= 0) && a.skip_ld == b.skip_ld && a.activation == b.activation;
}

// « convolution i (genre cin→cout sur ow×oh), sommes … : », le début d'une note du réglage.
std::string label(const Layer& layer, std::size_t index, std::size_t s) {
  return "convolution " + std::to_string(index) + " (" + kind_name(layer.kind) + " " +
         std::to_string(layer.cin) + "→" + std::to_string(layer.cout) + " sur " + std::to_string(layer.ow) +
         "×" + std::to_string(layer.oh) + "), sommes " + rn::sums_name(kSums[s]) + " : ";
}

}  // namespace

bool phased(rn::Kind kind) {
  return kind == rn::Kind::kPhase || kind == rn::Kind::kShuffle;
}

std::size_t pack_index(int ct) {
  return ct == 16 ? 0 : ct == 32 ? 1 : 2;
}

std::int64_t partial_of(const Layer& layer, const rn::Config& cfg, int oh) {
  return cfg.ksplit > 1
             ? rn::align256(std::int64_t{cfg.ksplit} * oh * layer.ow * layer.phases * layer.cout * 4)
             : 0;
}

rn::Params params_of(const Layer& layer, Rows rows, const Shape& shape, const __half* from, __half* to,
                     const __half* skip, float* partial, const void* w) {
  rn::Params p;
  p.in = from;
  p.in_ld = layer.in_ld;
  p.in_plane = static_cast<long>(rows.ih) * layer.iw;
  p.ih = rows.ih;
  p.iw = layer.iw;
  p.out = to;
  p.out_ld = layer.out_ld;
  p.out_ch = layer.out_ld;
  p.oh = rows.oh;
  p.ow = layer.ow;
  p.w = static_cast<const uint4*>(w);
  p.bias = layer.bias.as<const float>();
  p.slope = layer.activation ? layer.slope.as<const float>() : nullptr;
  p.skip = skip;
  p.skip_ld = layer.skip_ld;
  p.cin = layer.cin;
  p.tiles = layer.cout / shape.cfg.ct;
  p.ksplit = shape.cfg.ksplit;
  p.partial = shape.cfg.ksplit > 1 ? partial : nullptr;
  p.in_slope = layer.in_slope;
  return p;
}

void enqueue(const Layer& layer, const Shape& shape, cudaStream_t stream, const rn::Params& p) {
  shape.launch(shape.grid, shape.smem, stream, p);
  if (shape.cfg.ksplit > 1) rn::launch_reduce(phased(layer.kind), layer.phases * layer.cout, stream, p);
}

ConvNet::ConvNet(int height, int width) : height_(height), width_(width) {
  int device = 0;
  if (cudaGetDevice(&device) == cudaSuccess)
    static_cast<void>(cudaDeviceGetAttribute(&sms_, cudaDevAttrMultiProcessorCount, device));
  sms_ = std::max(sms_, 1);
}

bool ConvNet::plan(const prepare::GmfssGrid& grid, const Io& io, std::span<const int> pieces,
                   const Tuner& tuner, std::string& error) {
  if (grid.steps.empty() || (!pieces.empty() && pieces.size() != grid.steps.size())) {
    error = "réseau vide, ou morceaux pour une autre liste de convolutions";
    return false;
  }
  place(grid, io);
  layers_ = std::vector<Layer>(grid.steps.size());
  for (std::size_t i = 0; i < grid.steps.size(); ++i) {
    Layer& layer = layers_[i];
    layer.piece = pieces.empty() ? 0 : pieces[i];
    layer.head = std::ranges::find(io.kept, grid.steps[i].to) != io.kept.end();
    if (!layout(grid.steps[i], io, layer, error) || !build(grid.steps[i], tuner, layer, error)) {
      error = "convolution " + std::to_string(i) + " : " + error;
      return false;
    }
    for (std::size_t s = 0; s < kSums.size(); ++s) {
      const Shape& shape = layer.shapes[s];
      if (shape.rejected > 0)
        tuner.tuning.notes.push_back(label(layer, i, s) + std::to_string(shape.rejected) +
                                     " formes écartées (sortie différente)");
      if (!shape.stream.empty()) tuner.tuning.notes.push_back(label(layer, i, s) + shape.stream);
    }
  }
  tuner.tuning.layers += static_cast<int>(layers_.size());
  return true;
}

bool ConvNet::has_head(int piece) const {
  return std::ranges::any_of(layers_, [&](const Layer& l) { return l.piece == piece && l.head; });
}

void ConvNet::use(std::span<const RifeNativePrecision> precisions) {
  for (Layer& layer : layers_) {
    if (layer.piece < 0 || static_cast<std::size_t>(layer.piece) >= precisions.size()) continue;
    const RifeNativePrecision& p = precisions[static_cast<std::size_t>(layer.piece)];
    layer.sums = layer.head ? p.head : p.body;
  }
}

void ConvNet::trim() {
  for (Layer& layer : layers_)
    for (std::size_t k = 0; k < layer.packs.size(); ++k)
      if (k != pack_index(layer.shape().cfg.ct)) layer.packs[k].release();
}

void ConvNet::run(cudaStream_t stream, void* zone, float* partial, std::span<const void* const> bound,
                  int piece) const {
  for (const Layer& layer : layers_) {
    if (piece >= 0 && layer.piece != piece) continue;
    const Shape& shape = layer.shape();
    const __half* skip = layer.skip >= 0 ? address(layer.skip, zone, bound) : nullptr;
    enqueue(layer, shape, stream,
            params_of(layer, {.ih = layer.ih, .oh = layer.oh}, shape, address(layer.from, zone, bound),
                      address(layer.to, zone, bound), skip, partial,
                      layer.packs[pack_index(shape.cfg.ct)].as<const void>()));
  }
}

__half* ConvNet::address(int id, void* zone, std::span<const void* const> bound) const {
  const Placed& t = tensor(id);
  if (t.external < 0) return at(zone, t);
  return static_cast<__half*>(const_cast<void*>(bound[static_cast<std::size_t>(t.external)]));
}

void ConvNet::lifetimes(const prepare::GmfssGrid& grid, const Io& io) {
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
void ConvNet::place(const prepare::GmfssGrid& grid, const Io& io) {
  tensors_.assign(grid.tensors.size(), Placed{});
  for (std::size_t i = 0; i < grid.tensors.size(); ++i) {
    const prepare::GmfssGridTensor& t = grid.tensors[i];
    Placed& p = tensors_[i];
    const bool full = t.level == prepare::kGmfssFullLevel;
    p.height = full ? 2 * height_ : height_ >> t.level;
    p.width = full ? 2 * width_ : width_ >> t.level;
    p.channels = t.channels;
    p.bytes = rn::align256(std::int64_t{p.height} * p.width * p.channels * 2);
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

bool ConvNet::layout(const prepare::GmfssGridStep& step, const Io& io, Layer& layer,
                     std::string& error) const {
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
  const bool phases = phased(layer.kind);
  layer.phases = phases ? 4 : 1;
  layer.cout = pad16(step.shuffle ? conv.cout / 4 : conv.cout);
  layer.from = step.from;
  layer.to = step.to;
  layer.skip = step.skip;
  layer.cin = pad16(conv.cin);
  layer.in_ld = from.channels;
  layer.out_ld = to.channels;
  layer.ih = from.height;
  layer.iw = from.width;
  layer.oh = phases ? from.height : to.height;
  layer.ow = phases ? from.width : to.width;
  layer.in_slope = conv.in_slope;
  layer.activation = conv.out_slope != 1.F;
  const int scale = phases ? 2 : 1;
  if (layer.cin != layer.in_ld || to.height != scale * layer.oh || to.width != scale * layer.ow ||
      (rn::stride_of(layer.kind) == 2 && (from.height != 2 * to.height || from.width != 2 * to.width))) {
    error = "canaux d'entrée ou taille de sortie";
    return false;
  }
  if (phases && layer.activation) {
    error = "une sortie par phases n'a pas d'activation";
    return false;
  }
  if (step.skip >= 0) {
    const Placed& skip = tensor(step.skip);
    layer.skip_ld = skip.channels;
    if (phases || skip.channels != to.channels || skip.height != to.height) {
      error = "résiduel";
      return false;
    }
  }
  return true;
}

// Biais, pente et poids de chaque tuile envoyés ; les formes réglées au banc, ou reprises d'une convolution
// jumelle (same) ; les poids des tuiles qu'aucune forme n'emploie, libérés.
bool ConvNet::build(const prepare::GmfssGridStep& step, const Tuner& tuner, Layer& layer,
                    std::string& error) {
  const auto sent = [&](cudaError_t result) {
    if (result == cudaSuccess) return true;
    static_cast<void>(cudaGetLastError());
    error = std::string("poids : ") + cudaGetErrorString(result);
    return false;
  };
  const rn::Dense dense = dense_of(step, layer);
  if (layer.activation) {
    const std::vector<float> slope(static_cast<std::size_t>(layer.cout), step.conv.out_slope);
    if (!sent(layer.slope.upload(slope.data(), slope.size() * sizeof(float), tuner.stream))) return false;
  }
  if (!sent(layer.bias.upload(dense.bias.data(), dense.bias.size() * sizeof(float), tuner.stream)))
    return false;
  for (const int ct : {16, 32, 64}) {
    if (layer.cout % ct != 0) continue;
    const std::vector<std::uint32_t> packed = rn::packed_weights(dense, ct);
    if (!sent(layer.packs[pack_index(ct)].upload(packed.data(), packed.size() * sizeof(std::uint32_t),
                                                 tuner.stream)))
      return false;
  }
  if (const Layer* same = twin(layer); same != nullptr) {
    layer.shapes = same->shapes;
    for (Shape& shape : layer.shapes) {
      shape.rejected = 0;
      shape.stream.clear();
    }
    ++tuner.tuning.reused;
  } else {
    for (std::size_t s = 0; s < kSums.size(); ++s)
      if (!tune(tuner, s, layer, error)) return false;
  }
  for (const Shape& shape : layer.shapes)
    partial_bytes_ = std::max(partial_bytes_, partial_of(layer, shape.cfg, layer.oh));
  for (std::size_t k = 0; k < layer.packs.size(); ++k)
    if (std::ranges::none_of(layer.shapes, [&](const Shape& s) { return pack_index(s.cfg.ct) == k; }))
      layer.packs[k].release();
  return true;
}

// Une convolution déjà réglée de même noyau et de même géométrie, ou nulle.
const Layer* ConvNet::twin(const Layer& layer) const {
  for (const Layer& other : layers_) {
    if (&other == &layer) return nullptr;
    if (same_geometry(other, layer)) return &other;
  }
  return nullptr;
}

}  // namespace kirari::gpu::cuda::gmfss_plan
