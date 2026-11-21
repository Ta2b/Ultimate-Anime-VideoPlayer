// Le réglage au banc des convolutions des réseaux natifs de GMFSS (gmfss_conv_plan.cuh), à part de leur
// planification (gmfss_conv_plan.cu) : pour chacune des sommes, la forme de référence (la plus simple) lancée
// d'abord, puis chaque forme que les noyaux font, sur une sortie remplie de NaN, comparée à la référence aux
// tolérances de native_sums.h et chronométrée en graphe ; la plus rapide de celles qui rendent la sortie de
// la référence est gardée. Une grande couche ne passe au banc que sur une bande de ses lignes. Aux formes du
// RIFE natif s'ajoutent celles en flux (rn::stream_configs, conv_stream_kernels.cuh), mesurées de même.
// Les formes retenues servent ensuite à choisir et placer les anneaux du GridNet.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/conv_stream.h"
#include "gpu/cuda/backend/gmfss_conv_plan.cuh"
#include "gpu/cuda/backend/gmfss_shuffle_reuse.cuh"

namespace kirari::gpu::cuda::gmfss_plan {
namespace {

using rn::Kind;

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

std::string ring_shape_name(const Shape& shape) {
  const rn::Config& c = shape.cfg;
  return "Kc" + std::to_string(c.kc) + "/CT" + std::to_string(c.ct) + "/R" + std::to_string(c.rows) +
         (c.ahead ? "/avance" : "") + (c.ksplit > 1 ? "/K découpé" : "") + (c.bands > 0 ? "/flux" : "");
}

bool join_shapes(const Layer& a, const Layer& b, const Layer& up, const Layer& add) {
  constexpr auto f16 = sums_index(RifeNativeSums::kF16);
  const auto& ac = a.shapes[f16].cfg;
  const auto& bc = b.shapes[f16].cfg;
  const auto& uc = up.shapes[f16].cfg;
  const auto& dc = add.shapes[f16].cfg;
  return join_shape(a, 64, 2, ac.ahead) && (bc.kc == 32 || bc.kc == 64) && join_shape(b, bc.kc, 1, false) &&
         (uc.kc == 32 || uc.kc == 64) && (uc.rows == 1 || uc.rows == 2) &&
         join_shape(up, uc.kc, uc.rows, false) && join_shape(add, 64, 2, dc.ahead);
}

// Les octets que lit (l'entrée, qui sert aussi de résiduel au banc) et qu'écrit une convolution sur rows.
std::pair<std::int64_t, std::int64_t> bytes_of(const Layer& layer, Rows rows) {
  const std::int64_t cells = std::int64_t{rows.ih} * layer.iw;
  const std::int64_t in = layer.kind == Kind::kImage ? 3 * cells * 2 : cells * layer.in_ld * 2;
  const std::int64_t out = (phased(layer.kind) ? 4 : 1) * std::int64_t{rows.oh} * layer.ow * layer.out_ld * 2;
  return {std::max(in, layer.skip >= 0 ? out : 0), out};
}

// Au banc : au plus kBenchBytes par tampon (entrée, sortie, référence), des bandes de lignes par kBandStep.
constexpr std::int64_t kBenchBytes = std::int64_t{64} << 20;
constexpr int kBandStep = 16;

// La bande du banc : toute la couche si une forme y découpe K (moins de deux vagues), sinon les lignes de
// sortie, par kBandStep, dont l'entrée et la sortie tiennent chacune kBenchBytes (docs/gmfss-natif.md).
Rows band_of(const Layer& layer, const std::vector<rn::Config>& configs) {
  const int stride = rn::stride_of(layer.kind);
  const auto [in, out] = bytes_of(layer, Rows{.ih = stride * kBandStep, .oh = kBandStep});
  const std::int64_t oh = std::max<std::int64_t>(kBenchBytes / std::max(in, out), 1) * kBandStep;
  if (oh >= layer.oh || std::ranges::any_of(configs, [](const rn::Config& c) { return c.ksplit > 1; }))
    return {.ih = layer.ih, .oh = layer.oh};
  return {.ih = stride * static_cast<int>(oh), .oh = static_cast<int>(oh)};
}

// La forme cfg d'une convolution pour des sommes, sur oh lignes de sortie : lancement, grille, mémoire
// partagée. En flux, les colonnes de blocs de la couche entière (cfg.bands bandes au plus chacune) : sur la
// bande du banc, plus courtes mais aussi nombreuses, au même nombre de blocs que la couche entière.
Shape shape_for(const Layer& layer, RifeNativeSums sums, const rn::Config& cfg, int oh) {
  Shape s;
  s.cfg = cfg;
  const bool stream = cfg.bands > 0;
  if (layer.blocked8)
    s.launch = rn::pick_blocked8(sums, cfg);
  else
    s.launch = stream ? rn::pick_stream(sums, layer.kind, cfg)
                      : rn::pick_for(sums, layer.kind, cfg.kc, cfg.ct, cfg.rows, cfg.ahead);
  const int th = rn::kWarps * cfg.rows, tiles = (oh + th - 1) / th;
  const int strips =
      stream ? std::min(conv_stream::strips_for((layer.oh + th - 1) / th, cfg.bands), tiles) : tiles;
  s.grid = dim3(static_cast<unsigned>((layer.ow + rn::kTW - 1) / rn::kTW), static_cast<unsigned>(strips),
                static_cast<unsigned>(layer.phases * (layer.cout / cfg.ct) * cfg.ksplit));
  s.smem = stream ? conv_stream::smem_bytes(cfg.kc, cfg.rows)
                  : rn::smem_bytes(cfg.kc, cfg.ct, rn::kernel_size(layer.kind), rn::stride_of(layer.kind),
                                   cfg.rows);
  return s;
}

// La forme de référence du banc : tuile de 16, ni K découpé ni poids à l'avance ni flux, la plus petite
// tranche, deux lignes par warp en pas 1 et une en pas 2 ; la plus simple, comme celle du RIFE natif.
const rn::Config* reference_of(const std::vector<rn::Config>& configs, Kind kind) {
  const int rows = rn::stride_of(kind) == 2 ? 1 : 2;
  const rn::Config* out = nullptr;
  for (const rn::Config& c : configs) {
    if (c.ct != 16 || c.ksplit != 1 || c.ahead || c.bands > 0) continue;
    if (out == nullptr || c.kc < out->kc || (c.kc == out->kc && c.rows == rows)) out = &c;
  }
  return out;
}

// Un lancement de la forme au banc, sur la bande rows : l'entrée du banc sert aussi de résiduel.
rn::Bench::Shot shot_of(const Layer& layer, Rows rows, const Shape& shape, RifeNativeSums sums, const void* w,
                        bool bit_exact = false) {
  const auto [in, out] = bytes_of(layer, rows);
  return rn::Bench::Shot{
      .in_bytes = in,
      .out_bytes = out,
      .partial_bytes = partial_of(layer, shape.cfg, rows.oh),
      .sums = sums,
      .bit_exact = bit_exact,
      .enqueue = [&layer, rows, &shape, w](cudaStream_t stream, const __half* from, __half* to,
                                           float* partial) {
        enqueue(layer, shape, stream,
                params_of(layer, rows, shape, from, to, layer.skip >= 0 ? from : nullptr, partial, w));
      }};
}

// Le flux face à la tuile au banc, en microsecondes : la meilleure forme de chacun, et laquelle est gardée.
// Une forme qui ne se lance pas a un temps infini (rn::Bench), que la conversion en entier ne prend pas.
std::string versus(const rn::Config& stream, float streamed, float tiled) {
  const auto us = [](float ms) {
    return std::isfinite(ms) ? std::to_string(static_cast<long>(ms * 1000.F + 0.5F)) + " µs"
                             : std::string("sans lancement");
  };
  return "en flux au mieux " + us(streamed) + " (" + std::to_string(stream.bands) +
         " bandes par bloc, tuile " + std::to_string(stream.ct) + "), en tuile au mieux " + us(tiled) +
         (streamed < tiled ? " : flux retenu" : " : tuile gardée");
}

template <int RowsPerWarp, int Phases>
void launch_shuffle(dim3 grid, int smem, cudaStream_t stream, const rn::Params& p) {
  gmfss_shuffle_reuse::conv_shuffle<RowsPerWarp, Phases, rn::Acc::kF16>
      <<<grid, rn::kThreads, static_cast<std::size_t>(smem), stream>>>(p);
}

unsigned shuffle_tiles_y(int lines, int rows_per_warp) {
  const int tile = rn::kWarps * rows_per_warp;
  return static_cast<unsigned>((lines + tile - 1) / tile);
}

template <int RowsPerWarp, int Phases>
Shape shuffle_shape(const Layer& layer, int lines) {
  constexpr int smem = gmfss_shuffle_reuse::smem_bytes<RowsPerWarp>();
  if constexpr (smem > 48 * 1024) {
    const auto result =
        cudaFuncSetAttribute(gmfss_shuffle_reuse::conv_shuffle<RowsPerWarp, Phases, rn::Acc::kF16>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (result != cudaSuccess) {
      static_cast<void>(cudaGetLastError());  // la mémoire partagée de cette forme peut être indisponible
      return {};
    }
  }
  return Shape{
      .cfg = {.kc = 64, .ct = 64, .rows = RowsPerWarp},
      .launch = launch_shuffle<RowsPerWarp, Phases>,
      .grid = {static_cast<unsigned>((layer.ow + rn::kTW - 1) / rn::kTW), shuffle_tiles_y(lines, RowsPerWarp),
               static_cast<unsigned>(layer.phases / Phases)},
      .smem = smem,
      .stream = {},
  };
}

// Le halo groupé doit rendre les mêmes bits que la meilleure forme déjà choisie. Comparer à la tuile
// de référence du réglage général laisserait changer les sommes si la meilleure a découpé K.
bool tune_shuffle(const Tuner& tuner, const Layer& layer, Rows band, Shape& best, std::string& error) {
  if (layer.kind != Kind::kShuffle || layer.phases != 4 || layer.cin != 64 || layer.in_ld != 64 ||
      layer.cout != 64 || layer.out_ld != 64 || best.cfg.ksplit != 1)
    return true;
  const std::array candidates{shuffle_shape<1, 2>(layer, band.oh), shuffle_shape<1, 4>(layer, band.oh),
                              shuffle_shape<2, 2>(layer, band.oh), shuffle_shape<2, 4>(layer, band.oh)};
  constexpr RifeNativeSums sums = RifeNativeSums::kF16;
  const Shape direct = shape_for(layer, sums, best.cfg, band.oh);
  const auto shot = [&](const Shape& shape) {
    return shot_of(layer, band, shape, sums, layer.packs[pack_index(shape.cfg.ct)].as<const void>(), true);
  };
  if (!tuner.bench.reference(shot(direct), error)) return false;
  Shape grouped;
  grouped.ms = std::numeric_limits<float>::infinity();
  for (const Shape& candidate : candidates) {
    if (candidate.launch == nullptr) continue;
    const auto timed = tuner.bench.time(shot(candidate), error);
    if (!timed) return false;
    ++tuner.tuning.measured;
    if (timed->differs) {
      ++best.rejected;
      ++tuner.tuning.rejected;
    } else if (timed->ms < grouped.ms) {
      grouped = candidate;
      grouped.ms = timed->ms;
    }
  }
  // Une marge, comme à l'essai du réseau entier : le plus rapide des trois rejeux peut varier.
  if (grouped.ms < 0.98F * best.ms) {
    const auto us = [](float ms) { return std::to_string(static_cast<long>(ms * 1000.F + 0.5F)); };
    grouped.grid.y = shuffle_tiles_y(layer.oh, grouped.cfg.rows);
    grouped.rejected = best.rejected;
    grouped.stream = "halo partagé retenu, " + std::to_string(layer.phases / grouped.grid.z) + " phases, " +
                     std::to_string(grouped.cfg.rows) + " lignes par warp : " + us(grouped.ms) +
                     " µs contre " + us(best.ms) + " µs (sortie exacte)";
    best = std::move(grouped);
  }
  return true;
}

}  // namespace

// La forme de la convolution pour les sommes kSums[s] : la seule possible, sinon la plus rapide au banc (sur
// la bande de band_of) parmi celles qui rendent la sortie de la forme de référence.
bool ConvNet::tune(const Tuner& tuner, std::size_t s, Layer& layer, std::string& error) {
  const RifeNativeSums sums = kSums[s];
  const rn::Extent extent{
      .cin = layer.cin, .cout = layer.cout, .oh = layer.oh, .ow = layer.ow, .phases = layer.phases};
  std::vector<rn::Config> configs = rn::configs(sums, layer.kind, extent, sms_);
  const std::vector<rn::Config> stream = rn::stream_configs(sums, layer.kind, extent, sms_);
  configs.insert(configs.end(), stream.begin(), stream.end());
  const rn::Config* reference = reference_of(configs, layer.kind);
  if (reference == nullptr) {
    error = std::string("aucune forme en sommes ") + rn::sums_name(sums);
    return false;
  }
  Shape& best = layer.shapes[s];
  best = shape_for(layer, sums, *reference, layer.oh);
  if (configs.size() == 1) {
    ++tuner.tuning.single;
    return true;
  }
  const auto weights = [&](const rn::Config& cfg) {
    return layer.packs[pack_index(cfg.ct)].as<const void>();
  };
  const Rows band = band_of(layer, configs);
  const Shape first = shape_for(layer, sums, *reference, band.oh);
  if (!tuner.bench.reference(shot_of(layer, band, first, sums, weights(*reference)), error)) return false;
  best.ms = std::numeric_limits<float>::infinity();
  int rejected = 0;
  float tiled = best.ms, streamed = best.ms;
  const rn::Config* flow = nullptr;  // la plus rapide des formes en flux
  for (const rn::Config& cfg : configs) {
    const Shape shape = shape_for(layer, sums, cfg, band.oh);
    const std::optional<rn::Bench::Timing> timed =
        tuner.bench.time(shot_of(layer, band, shape, sums, weights(cfg)), error);
    if (!timed) return false;
    ++tuner.tuning.measured;
    if (timed->differs && &cfg != reference) {
      ++rejected;
      continue;
    }
    if (cfg.bands == 0) tiled = std::min(tiled, timed->ms);
    if (cfg.bands > 0 && timed->ms < streamed) {
      streamed = timed->ms;
      flow = &cfg;
    }
    if (timed->ms < best.ms) {
      best = shape_for(layer, sums, cfg, layer.oh);
      best.ms = timed->ms;
    }
  }
  best.rejected = rejected;
  if (flow != nullptr) best.stream = versus(*flow, streamed, tiled);
  tuner.tuning.rejected += rejected;
  if (!(best.ms < std::numeric_limits<float>::infinity())) {
    error = std::string("aucune forme ne se lance en sommes ") + rn::sums_name(sums);
    return false;
  }
  return sums != RifeNativeSums::kF16 || tune_shuffle(tuner, layer, band, best, error);
}

// Le premier filtre applique le PReLU. Le deuxième ajoute le résiduel et il est le seul
// à relire l'intermédiaire, dont la zone peut alors servir d'anneau.
bool ConvNet::residual_pair(std::size_t i) const {
  const Layer& a = layers_[i];
  const Layer& b = layers_[i + 1];
  const Placed& temp = tensor(a.to);
  const auto now = static_cast<int>(i);
  return a.piece == 0 && b.piece == 0 && same_c64(a, height_, width_) && same_c64(b, height_, width_) &&
         a.skip < 0 && a.activation && !b.activation && b.skip == a.from && b.skip_ld == 64 &&
         b.from == a.to && b.in_slope == 1.F && a.to != b.to && temp.external < 0 && temp.first == now &&
         temp.last == now + 1;
}

void ConvNet::plan_residual_rings(const Tuner& tuner) {
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

bool ConvNet::join_lifetimes(std::size_t i, std::int64_t bytes) const {
  // Le résiduel J n'est lu que par l'addition. Son allocation existante peut garder les trois anneaux.
  const int now = static_cast<int>(i);
  const Placed& j = tensor(layers_[i + 1].to);
  const Layer& up = layers_[i + 2];
  const Placed& t = tensor(up.to);
  const Placed& residual_in = tensor(layers_[i].from);
  const Placed& up_in = tensor(up.from);
  const Placed& out = tensor(layers_[i + 3].to);
  return j.external < 0 && j.first == now + 1 && j.last == now + 3 && j.bytes >= bytes && t.external < 0 &&
         t.first == now + 2 && t.last == now + 3 && residual_in.first < now && up_in.first < now &&
         out.first == now + 3;
}

bool ConvNet::join_candidate(std::size_t i, std::int64_t bytes, const Tuner& tuner) const {
  const Layer& a = layers_[i];
  const Layer& b = layers_[i + 1];
  const Layer& up = layers_[i + 2];
  const Layer& add = layers_[i + 3];
  if (!residual_pair(i) || up.piece != 0 || add.piece != 0 || !up_c64(up, height_, width_) ||
      !same_c64(add, height_, width_) || up.to != add.from || b.to != add.skip || add.skip_ld != 64 ||
      add.activation)
    return false;
  const bool fits = join_lifetimes(i, bytes);
  const bool shape = join_shapes(a, b, up, add);
  constexpr auto f16 = sums_index(RifeNativeSums::kF16);
  tuner.tuning.notes.push_back(
      "anneaux C64 avec addition, convolutions " + std::to_string(i) + "–" + std::to_string(i + 3) + " : " +
      ring_shape_name(b.shapes[f16]) + ", " + ring_shape_name(up.shapes[f16]) + ", " +
      ring_shape_name(add.shapes[f16]) +
      (fits && shape ? " : forme compatible en sommes f16" : " : forme réglée gardée"));
  return fits && shape;
}

void ConvNet::plan_join_rings(std::int64_t bytes, const Tuner& tuner) {
  std::vector<std::size_t> candidates;
  for (std::size_t i = 0; i + 3 < layers_.size(); ++i)
    if (join_candidate(i, bytes, tuner)) candidates.push_back(i);
  if (candidates.empty()) return;
  const char* phases = std::getenv("KIRARI_AB_GMFSS_JOIN_PHASES");  // env: banc
  join_phases_ = phases == nullptr || std::string_view(phases) != "0";
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
    tuner.tuning.notes.emplace_back("anneaux C64 avec addition : zone incompatible, anneau résiduel gardé");
  }
  // Le mélange de précisions et l'IFNet sont choisis avec les adresses ordinaires et sans joints.
  tensors_ = std::move(original_tensors);
  zone_bytes_ = original_bytes;
}

}  // namespace kirari::gpu::cuda::gmfss_plan
