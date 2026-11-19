// Le réglage au banc des convolutions des réseaux natifs de GMFSS (gmfss_conv_plan.cuh), à part de leur
// planification (gmfss_conv_plan.cu) : pour chacune des sommes, la forme de référence (la plus simple) lancée
// d'abord, puis chaque forme que les noyaux font, sur une sortie remplie de NaN, comparée à la référence aux
// tolérances de native_sums.h et chronométrée en graphe ; la plus rapide de celles qui rendent la sortie de
// la référence est gardée. Une grande couche ne passe au banc que sur une bande de ses lignes. Aux formes du
// RIFE natif s'ajoutent celles en flux (rn::stream_configs, conv_stream_kernels.cuh), mesurées de même.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/conv_stream.h"
#include "gpu/cuda/backend/gmfss_conv_plan.cuh"

namespace kirari::gpu::cuda::gmfss_plan {
namespace {

using rn::Kind;

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
// partagée. En flux, une colonne de blocs prend cfg.bands bandes au plus.
Shape shape_for(const Layer& layer, RifeNativeSums sums, const rn::Config& cfg, int oh) {
  Shape s;
  s.cfg = cfg;
  const bool stream = cfg.bands > 0;
  s.launch = stream ? rn::pick_stream(sums, layer.kind, cfg)
                    : rn::pick_for(sums, layer.kind, cfg.kc, cfg.ct, cfg.rows, cfg.ahead);
  const int th = rn::kWarps * cfg.rows, tiles = (oh + th - 1) / th;
  s.grid = dim3(static_cast<unsigned>((layer.ow + rn::kTW - 1) / rn::kTW),
                static_cast<unsigned>(stream ? conv_stream::strips_for(tiles, cfg.bands) : tiles),
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
rn::Bench::Shot shot_of(const Layer& layer, Rows rows, const Shape& shape, RifeNativeSums sums,
                        const void* w) {
  const auto [in, out] = bytes_of(layer, rows);
  return rn::Bench::Shot{
      .in_bytes = in,
      .out_bytes = out,
      .partial_bytes = partial_of(layer, shape.cfg, rows.oh),
      .sums = sums,
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
  return true;
}

}  // namespace kirari::gpu::cuda::gmfss_plan
