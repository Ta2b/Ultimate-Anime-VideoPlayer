// La planification des réseaux natifs de GMFSS (gmfss_conv_plan.cuh) : les tenseurs placés par durée de vie,
// puis chaque convolution rangée (rn::Dense, les poids de chaque tuile par rn::packed_weights) et réglée au
// banc du RIFE natif pour les sommes f16, mixtes et f32 (gmfss_conv_tune.cu), comme celles du RIFE natif
// (rife_native_net.cu). Une convolution qui n'a qu'une forme n'est pas mesurée ; une convolution de même
// noyau et de même géométrie qu'une autre déjà réglée (les blocs d'une ligne du GridNet) en reprend les
// formes.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/gmfss_conv_plan.cuh"
#include "gpu/cuda/backend/gmfss_grid_folded.cuh"
#include "gpu/cuda/backend/gmfss_grid_head.cuh"
#include "gpu/cuda/backend/gmfss_grid_join.cuh"
#include "gpu/cuda/backend/gmfss_grid_rgb.cuh"
#include "gpu/cuda/backend/gmfss_grid_ring.cuh"
#include "gpu/cuda/backend/gmfss_grid_tail.cuh"

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
  return a.kind == b.kind && a.blocked8 == b.blocked8 && a.cin == b.cin && a.cout == b.cout && a.ih == b.ih &&
         a.iw == b.iw && a.oh == b.oh && a.ow == b.ow && a.in_ld == b.in_ld && a.out_ld == b.out_ld &&
         (a.skip >= 0) == (b.skip >= 0) && a.skip_ld == b.skip_ld && a.activation == b.activation &&
         (a.in_slope != 1.F) == (b.in_slope != 1.F);
}

int find_planar_head(const prepare::GmfssGrid& grid, std::span<const Layer> layers, int height, int width,
                     bool eligible) {
  if (!eligible || std::ranges::count_if(grid.steps, [](const auto& step) {
                     return step.from == 0 || step.skip == 0 || step.to == 0;
                   }) != 1)
    return -1;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const Layer& l = layers[i];
    if (l.from == 0 && l.skip < 0 && l.kind == Kind::kSame && !l.blocked8 && grid.steps[i].conv.cin == 9 &&
        l.cin == 16 && l.cout == 64 && l.in_ld == 16 && l.out_ld == 64 && l.iw == width && l.ow == width &&
        l.ih == height && l.oh == height)
      return static_cast<int>(i);
  }
  return -1;
}

bool direct_ring_shape(const Shape& shape) {
  const rn::Config& c = shape.cfg;
  return c.kc == 64 && c.ct == 64 && (c.rows == 1 || c.rows == 2) && c.ksplit == 1 && c.bands == 0 &&
         shape.launch != nullptr &&
         shape.launch == rn::pick_for(RifeNativeSums::kF16, Kind::kSame, c.kc, c.ct, c.rows, c.ahead);
}

bool same_c64(const Layer& layer, int height, int width) {
  return layer.kind == Kind::kSame && layer.phases == 1 && !layer.blocked8 && layer.cin == 64 &&
         layer.cout == 64 && layer.in_ld == 64 && layer.out_ld == 64 && layer.ih == height &&
         layer.oh == height && layer.iw == width && layer.ow == width;
}

bool join_shape(const Layer& layer, int kc, int rows, bool ahead) {
  constexpr auto f16 = RifeNativeSums::kF16;
  const Shape& shape = layer.shapes[sums_index(f16)];
  const rn::Config& c = shape.cfg;
  return c.kc == kc && c.ct == 64 && c.rows == rows && c.ahead == ahead && c.ksplit == 1 && c.bands == 0 &&
         shape.launch != nullptr &&
         shape.launch == rn::pick_for(f16, layer.kind, c.kc, c.ct, c.rows, c.ahead);
}

bool up_c64(const Layer& up, int height, int width) {
  return up.kind == Kind::kPhase && up.phases == 4 && !up.blocked8 && up.cin == 128 && up.in_ld == 128 &&
         up.cout == 64 && up.out_ld == 64 && 2 * up.ih == height && 2 * up.iw == width && up.oh == up.ih &&
         up.ow == up.iw && up.skip < 0 && !up.activation;
}

bool separate(const Placed& zone, std::int64_t bytes, const Placed& t) {
  return t.external < 0 && (zone.offset + bytes <= t.offset || t.offset + t.bytes <= zone.offset);
}

bool on_ada() {
  int device = 0, major = 0, minor = 0;
  return cudaGetDevice(&device) == cudaSuccess &&
         cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
         cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess &&
         major == 8 && minor == 9;
}

std::string ring_shape_name(const Shape& shape) {
  const rn::Config& c = shape.cfg;
  return "Kc" + std::to_string(c.kc) + "/CT" + std::to_string(c.ct) + "/R" + std::to_string(c.rows) +
         (c.ahead ? "/avance" : "") + (c.ksplit > 1 ? "/K découpé" : "") + (c.bands > 0 ? "/flux" : "");
}

// « convolution i (genre cin→cout sur ow×oh), sommes … : », le début d'une note du réglage.
std::string label(const Layer& layer, std::size_t index, std::size_t s) {
  return "convolution " + std::to_string(index) + " (" + kind_name(layer.kind) + " " +
         std::to_string(layer.cin) + "→" + std::to_string(layer.cout) + " sur " + std::to_string(layer.ow) +
         "×" + std::to_string(layer.oh) + "), sommes " + rn::sums_name(kSums[s]) + " : ";
}

void record_tuning(const Layer& layer, std::size_t index, std::vector<std::string>& notes) {
  for (std::size_t s = 0; s < kSums.size(); ++s) {
    const Shape& shape = layer.shapes[s];
    if (shape.rejected > 0)
      notes.push_back(label(layer, index, s) + std::to_string(shape.rejected) +
                      " formes écartées (sortie différente)");
    if (!shape.stream.empty()) notes.push_back(label(layer, index, s) + shape.stream);
  }
}

template <rn::Acc A>
void launch_rgb(cudaStream_t stream, const Shape& shape, rn::Params p) {
  switch (shape.cfg.kc) {
    case 16:
      gmfss_grid_rgb::conv_rgb<16, 2, A><<<shape.grid, rn::kThreads, shape.smem, stream>>>(p);
      break;
    case 32:
      gmfss_grid_rgb::conv_rgb<32, 2, A><<<shape.grid, rn::kThreads, shape.smem, stream>>>(p);
      break;
    case 64:
      gmfss_grid_rgb::conv_rgb<64, 2, A><<<shape.grid, rn::kThreads, shape.smem, stream>>>(p);
      break;
  }
}

bool supports_rgb(const Layer& layer, const Shape& shape) {
  const rn::Config& cfg = shape.cfg;
  return layer.rgb && cfg.ct == 16 && cfg.rows == 2 && cfg.ksplit == 1 && cfg.bands == 0 && !cfg.ahead &&
         (cfg.kc == 16 || cfg.kc == 32 || cfg.kc == 64);
}

void enqueue_rgb(cudaStream_t stream, const Shape& shape, RifeNativeSums sums, const rn::Params& p) {
  switch (sums) {
    case RifeNativeSums::kF16:
      return launch_rgb<rn::Acc::kF16>(stream, shape, p);
    case RifeNativeSums::kMixed:
      return launch_rgb<rn::Acc::kMixed>(stream, shape, p);
    case RifeNativeSums::kF32:
      return launch_rgb<rn::Acc::kF32>(stream, shape, p);
  }
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
  join_tensors_.clear();
  join_on_ = false;
  planar_head_ = -1;
  tail_on_ = false;
  folded_tail_on_ = false;
  folded_weights_.release();
  folded_edges_.release();
  folded_bias_.release();
  const char* tail = std::getenv("KIRARI_AB_GMFSS_GRID_TAIL");  // env: banc
  try_tail_ = io.rgb >= 0 && width_ <= 960 && height_ >= gmfss_grid_tail::kRingRows &&
              (tail == nullptr || std::string_view(tail) != "0") && on_ada();
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
    // Les autres couches lisent encore leur sortie au format NHWC.
    layer.rgb = layer.rgb && i + 1 == grid.steps.size();
    record_tuning(layer, i, tuner.tuning.notes);
  }
  static const bool try_ring = [] {
    const char* value = std::getenv("KIRARI_AB_GMFSS_GRID_RING");  // env: banc
    return value != nullptr && std::string_view(value) == "1";
  }();
  const bool try_join = io.rgb >= 0 && grid.steps.size() == 45 && width_ == 960 && height_ == 576 && on_ada();
  planar_head_ = find_planar_head(grid, layers_, height_, width_, try_join);
  const auto residual_pair = [&](std::size_t i) {
    const Layer& a = layers_[i];
    const Layer& b = layers_[i + 1];
    const Placed& temp = tensor(a.to);
    const auto now = static_cast<int>(i);
    return a.piece == 0 && b.piece == 0 && same_c64(a, height_, width_) && same_c64(b, height_, width_) &&
           a.skip < 0 && a.activation && !b.activation && b.skip == a.from && b.skip_ld == 64 &&
           b.from == a.to && b.in_slope == 1.F && a.to != b.to && temp.external < 0 && temp.first == now &&
           temp.last == now + 1;
  };
  if (try_ring && io.rgb >= 0 && height_ >= gmfss_grid_ring::kRingRows) {
    constexpr auto f16 = sums_index(RifeNativeSums::kF16);
    for (std::size_t i = 0; i + 1 < layers_.size(); ++i) {
      Layer& a = layers_[i];
      const Layer& b = layers_[i + 1];
      if (!residual_pair(i)) continue;
      a.ring_next = direct_ring_shape(a.shapes[f16]) && direct_ring_shape(b.shapes[f16]);
      tuner.tuning.notes.push_back(
          "anneau C64, convolutions " + std::to_string(i) + "–" + std::to_string(i + 1) + " : " +
          ring_shape_name(a.shapes[f16]) + ", " + ring_shape_name(b.shapes[f16]) +
          (a.ring_next ? " : forme compatible en sommes f16" : " : forme directe gardée"));
    }
  }
  if (try_join) {
    const auto bytes = std::int64_t{gmfss_grid_join::kWorkspaceRows} * width_ * 64 * 2;
    std::vector<std::size_t> candidates;
    for (std::size_t i = 0; i + 3 < layers_.size(); ++i) {
      const Layer& a = layers_[i];
      const Layer& b = layers_[i + 1];
      const Layer& up = layers_[i + 2];
      const Layer& add = layers_[i + 3];
      if (!residual_pair(i) || up.piece != 0 || add.piece != 0 || !up_c64(up, height_, width_) ||
          !same_c64(add, height_, width_) || up.to != add.from || b.to != add.skip || add.skip_ld != 64 ||
          add.activation)
        continue;

      // Le résiduel J n'est lu que par l'addition. Son allocation existante peut garder les trois anneaux.
      const int now = static_cast<int>(i);
      const Placed& j = tensor(b.to);
      const Placed& t = tensor(up.to);
      const Placed& residual_in = tensor(a.from);
      const Placed& up_in = tensor(up.from);
      const Placed& out = tensor(add.to);
      const bool fits = j.external < 0 && j.first == now + 1 && j.last == now + 3 && j.bytes >= bytes &&
                        t.external < 0 && t.first == now + 2 && t.last == now + 3 &&
                        residual_in.first < now && up_in.first < now && out.first == now + 3;
      constexpr auto f16 = sums_index(RifeNativeSums::kF16);
      const auto& ac = a.shapes[f16].cfg;
      const auto& bc = b.shapes[f16].cfg;
      const auto& uc = up.shapes[f16].cfg;
      const auto& dc = add.shapes[f16].cfg;
      const bool shape = join_shape(a, 64, 2, ac.ahead) && (bc.kc == 32 || bc.kc == 64) &&
                         join_shape(b, bc.kc, 1, false) && (uc.kc == 32 || uc.kc == 64) &&
                         (uc.rows == 1 || uc.rows == 2) && join_shape(up, uc.kc, uc.rows, false) &&
                         join_shape(add, 64, 2, dc.ahead);
      if (fits && shape) candidates.push_back(i);
      tuner.tuning.notes.push_back(
          "anneaux C64 avec addition, convolutions " + std::to_string(i) + "–" + std::to_string(i + 3) +
          " : " + ring_shape_name(b.shapes[f16]) + ", " + ring_shape_name(up.shapes[f16]) + ", " +
          ring_shape_name(add.shapes[f16]) +
          (fits && shape ? " : forme compatible en sommes f16" : " : forme réglée gardée"));
    }
    if (!candidates.empty()) {
      auto original_tensors = tensors_;
      const auto original_bytes = zone_bytes_;
      for (const auto i : candidates) {
        // B et I doivent rester intacts jusqu'à la dernière bande. Sans cela le plan initial peut partager
        // la même adresse entre B et la sortie J2 dans laquelle l'addition écrit maintenant dès le début.
        for (const int from : {layers_[i].from, layers_[i + 2].from}) {
          Placed& t = tensors_[static_cast<std::size_t>(from)];
          t.last = std::max(t.last, static_cast<int>(i + 3));
        }
      }
      pack();
      bool safe = zone_bytes_ <= original_bytes;
      for (const auto i : candidates) {
        const Placed& j = tensor(layers_[i + 1].to);
        const Placed& b = tensor(layers_[i].from);
        const Placed& up_in = tensor(layers_[i + 2].from);
        const Placed& out = tensor(layers_[i + 3].to);
        safe = safe && separate(j, bytes, b) && separate(j, bytes, up_in) && separate(j, bytes, out) &&
               separate(out, out.bytes, b) && separate(out, out.bytes, up_in);
      }
      if (safe) {
        join_tensors_ = std::move(tensors_);
        for (const auto i : candidates) layers_[i].join_next = true;
      } else {
        tuner.tuning.notes.emplace_back(
            "anneaux C64 avec addition : zone incompatible, anneau résiduel gardé");
      }
      // Le mélange de précisions et l'IFNet sont choisis avec les adresses ordinaires et sans joints.
      tensors_ = std::move(original_tensors);
      zone_bytes_ = original_bytes;
    }
  }
  const char* folded = std::getenv("KIRARI_AB_GMFSS_FOLDED_TAIL");  // env: banc
  if (grid.folded_tail_verified && (folded == nullptr || std::string_view(folded) != "0") && try_join) {
    const std::size_t up_index = layers_.size() - 2;
    const Layer& up = layers_[up_index];
    const Layer& rgb = layers_.back();
    const auto& first = grid.steps[up_index].conv;
    const auto& last = grid.steps.back().conv;
    const Placed& intermediate = tensor(up.to);
    if (up.kind == Kind::kShuffle && first.stride == 1 && !first.transposed && first.cin == 64 &&
        first.cout == 256 && up.ih == height_ && up.iw == width_ && up.in_ld == 64 && !up.blocked8 &&
        up.in_slope == 1.F && !up.activation && up.skip < 0 && up.piece == rgb.piece && rgb.rgb &&
        rgb.from == up.to && rgb.in_slope == 1.F && rgb.oh == 2 * height_ && rgb.ow == 2 * width_ &&
        intermediate.first == static_cast<int>(up_index) &&
        intermediate.last == static_cast<int>(up_index + 1)) {
      const auto weights = gmfss_grid_folded::pack(first, last);
      auto status = folded_weights_.upload(weights.interior.data(), weights.interior.size() * sizeof(uint2),
                                           tuner.stream);
      if (status == cudaSuccess)
        status =
            folded_edges_.upload(weights.edges.data(), weights.edges.size() * sizeof(float), tuner.stream);
      if (status == cudaSuccess)
        status = folded_bias_.upload(weights.bias.data(), weights.bias.size() * sizeof(float), tuner.stream);
      // Les poids côté hôte restent vivants jusqu'à la fin de toute copie déjà soumise.
      if (const auto synced = cudaStreamSynchronize(tuner.stream); synced != cudaSuccess) status = synced;
      if (status != cudaSuccess) {
        folded_weights_.release();
        folded_edges_.release();
        folded_bias_.release();
        if (status != cudaErrorMemoryAllocation) {
          error = std::string("queue composée : ") + cudaGetErrorString(status);
          return false;
        }
        static_cast<void>(cudaGetLastError());
        tuner.tuning.notes.emplace_back("queue composée indisponible : mémoire CUDA");
      }
    }
  }
  tuner.tuning.layers += static_cast<int>(layers_.size());
  return true;
}

bool ConvNet::has_head(int piece) const {
  return std::ranges::any_of(layers_, [&](const Layer& l) { return l.piece == piece && l.head; });
}

bool ConvNet::planar_head_ready() const {
  if (planar_head_ < 0) return false;
  const Layer& l = layers_[static_cast<std::size_t>(planar_head_)];
  const Shape& shape = l.shape();
  const auto& c = shape.cfg;
  return shape.launch != nullptr &&
         shape.launch == rn::pick_for(l.sums, l.kind, c.kc, c.ct, c.rows, c.ahead) &&
         gmfss_grid_head::pick(c, l.sums) != nullptr;
}

bool ConvNet::tail_ready() const {
  if (!try_tail_ || layers_.size() < 2) return false;
  const std::size_t i = layers_.size() - 2;
  const Layer& first = layers_[i];
  const Layer& last = layers_[i + 1];
  const Placed& scratch = tensor(first.to);
  const auto bytes =
      std::int64_t{gmfss_grid_tail::kRingRows} * last.iw * 64 * static_cast<std::int64_t>(sizeof(__half));
  const auto& c = first.shape().cfg;
  return first.kind == Kind::kShuffle && first.piece == last.piece && first.cin == 64 && first.in_ld == 64 &&
         first.cout == 64 && first.out_ld == 64 && first.phases == 4 && first.in_slope == 1.F &&
         !first.activation && first.skip < 0 && first.sums == RifeNativeSums::kF16 && c.kc == 64 &&
         c.ksplit == 1 && first.packs[pack_index(64)].as<const void>() != nullptr && last.from == first.to &&
         last.ih == 2 * first.oh && last.iw == 2 * first.ow && last.cin == 64 && last.in_ld == 64 &&
         last.skip < 0 && last.in_slope == 1.F && last.sums == RifeNativeSums::kF32 &&
         supports_rgb(last, last.shape()) && scratch.external < 0 && scratch.first == static_cast<int>(i) &&
         scratch.last == static_cast<int>(i + 1) && scratch.bytes >= bytes;
}

bool ConvNet::joins_at(std::size_t i) const {
  constexpr auto f16 = RifeNativeSums::kF16;
  return layers_[i].join_next && layers_[i].sums == f16 && layers_[i + 1].sums == f16 &&
         layers_[i + 2].sums == f16 && layers_[i + 3].sums == f16;
}

bool ConvNet::join_ready() const {
  if (join_tensors_.empty()) return false;
  for (std::size_t i = 0; i + 3 < layers_.size(); ++i)
    if (joins_at(i)) return true;
  return false;
}

void ConvNet::use_join(bool enabled) {
  if (join_on_ == enabled || join_tensors_.empty()) return;
  tensors_.swap(join_tensors_);
  join_on_ = enabled;
}

void ConvNet::use(std::span<const RifeNativePrecision> precisions) {
  for (Layer& layer : layers_) {
    if (layer.piece < 0 || static_cast<std::size_t>(layer.piece) >= precisions.size()) continue;
    const RifeNativePrecision& p = precisions[static_cast<std::size_t>(layer.piece)];
    layer.sums = layer.head ? p.head : p.body;
  }
}

void ConvNet::trim() {
  const Layer* shuffle = tail_ready() ? &layers_[layers_.size() - 2] : nullptr;
  for (Layer& layer : layers_)
    for (std::size_t k = 0; k < layer.packs.size(); ++k)
      if (k != pack_index(layer.shape().cfg.ct) && (&layer != shuffle || k != pack_index(64)))
        layer.packs[k].release();
}

bool ConvNet::run(cudaStream_t stream, void* zone, float* partial, std::span<const void* const> bound,
                  int piece, __half* rgb, const std::array<const __half*, 3>* planar_head) const {
  const auto parameters = [&](const Layer& layer, __half* out) {
    const Shape& shape = layer.shape();
    const __half* skip = layer.skip >= 0 ? address(layer.skip, zone, bound) : nullptr;
    return params_of(layer, {.ih = layer.ih, .oh = layer.oh}, shape, address(layer.from, zone, bound), out,
                     skip, partial, layer.packs[pack_index(shape.cfg.ct)].as<const void>());
  };
  bool written = false;
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    const Layer& layer = layers_[i];
    if (piece >= 0 && layer.piece != piece) continue;
    const Shape& shape = layer.shape();
    const bool direct = rgb != nullptr && supports_rgb(layer, shape);
    const rn::Params p = parameters(layer, direct ? rgb : address(layer.to, zone, bound));
    constexpr auto f16 = RifeNativeSums::kF16;
    if (planar_head && static_cast<int>(i) == planar_head_) {
      const auto launch = gmfss_grid_head::pick(shape.cfg, layer.sums);
      launch(shape.grid, shape.smem, stream, p, {(*planar_head)[0], (*planar_head)[1], (*planar_head)[2]});
    } else if (folded_tail_on_ && i + 2 == layers_.size() && rgb != nullptr &&
               (piece < 0 || rgb != address(layers_.back().to, zone, bound))) {
      gmfss_grid_folded::enqueue({.input = p.in,
                                  .output = rgb,
                                  .weights = folded_weights_.as<const uint2>(),
                                  .bias = folded_bias_.as<const float>(),
                                  .edges = folded_edges_.as<const float>(),
                                  .height = height_,
                                  .width = width_},
                                 stream);
      ++i;
      written = true;
    } else if (tail_on_ && i + 2 == layers_.size() && rgb != nullptr &&
               (piece < 0 || rgb != address(layers_.back().to, zone, bound))) {
      const Layer& head = layers_[++i];
      const rn::Params q = parameters(head, rgb);
      rn::Params shuffle = p;
      shuffle.w = layer.packs[pack_index(64)].as<const uint4>();
      shuffle.tiles = 1;
      switch (head.shape().cfg.kc) {
        case 16:
          gmfss_grid_tail::enqueue<16>(shuffle, q, stream);
          break;
        case 32:
          gmfss_grid_tail::enqueue<32>(shuffle, q, stream);
          break;
        case 64:
          gmfss_grid_tail::enqueue<64>(shuffle, q, stream);
          break;
      }
      written = true;
    } else if (piece == -1 && join_on_ && joins_at(i)) {
      const Layer& b = layers_[i + 1];
      const Layer& up = layers_[i + 2];
      const Layer& add = layers_[i + 3];
      const auto ring = std::int64_t{gmfss_grid_join::kRingRows} * width_ * 64;
      auto* const storage = address(b.to, zone, bound);
      gmfss_grid_join::Join args{p, parameters(b, storage + ring), parameters(up, storage + 2 * ring),
                                 parameters(add, address(add.to, zone, bound))};
      args.residual_first.out = storage;
      args.residual_second.in = storage;
      args.up_second.in = storage + 2 * ring;
      args.up_second.skip = storage + ring;
      const auto& uc = up.shape().cfg;
      const gmfss_grid_join::Config cfg{shape.cfg.ahead, b.shape().cfg.kc, uc.kc, uc.rows,
                                        add.shape().cfg.ahead};
      gmfss_grid_join::join(args, cfg, stream);
      i += 3;
    } else if (layer.ring_next && layer.sums == f16 && layers_[i + 1].sums == f16) {
      const Layer& second = layers_[++i];
      const auto& c = second.shape().cfg;
      const auto q = parameters(second, address(second.to, zone, bound));
      gmfss_grid_ring::enqueue_residual(p, q, shape.cfg.rows, shape.cfg.ahead, c.rows, c.ahead, stream);
    } else if (direct) {
      enqueue_rgb(stream, shape, layer.sums, p);
      written = true;
    } else {
      enqueue(layer, shape, stream, p);
    }
  }
  return written;
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
  pack();
}

void ConvNet::pack() {
  zone_bytes_ = 0;
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
  layer.blocked8 = io.features_c8 && step.from > 0 && step.from < prepare::kGmfssGridInputs;
  if (layer.blocked8 && (layer.kind != Kind::kSame || conv.cin < 128 || conv.cin % 8 != 0)) {
    error = "entrée des caractéristiques GridNet par groupes de huit";
    return false;
  }
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
  layer.rgb = step.to == io.rgb && layer.head && layer.kind == Kind::kSame && conv.cin == 64 &&
              conv.cout == 3 && layer.out_ld == 8 && step.skip < 0 && !layer.activation;
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
