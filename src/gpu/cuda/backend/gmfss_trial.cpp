// Le choix commun des essais au chargement du GMFSS natif (gmfss_trial.h, docs/gmfss-natif.md, « Essai par
// morceaux ») : une passe TensorRT (la référence), une passe native f32 partout (la base), le temps seul de
// chaque option de chaque morceau, puis le mélange de native_mix.h, dont l'effet de chaque option est mesuré
// contre la base sur l'unité entière. Les crans f16 et mixtes de tout le réseau, les variantes de l'ancien
// essai, sont mesurés tels quels : le mélange retenu fait au moins aussi bien qu'elles. Chaque sortie de
// l'unité (l'image de la fusion ; les caractéristiques, les mouvements de la paire) est jugée à part, l'écart
// rapporté à son étendue (au moins 1).
#include "gpu/cuda/backend/gmfss_trial.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/gmfss_mix.h"
#include "gpu/cuda/backend/native_mix.h"

namespace kirari::gpu::cuda::gmfss_trial {
namespace {

using Planes = std::vector<std::vector<__half>>;
using native_mix::Options;

// Trois passes (directe, capture du graphe de TensorRT, rejeu) ; la sortie relue est celle de la dernière.
std::expected<void, Refusal> settled(const Unit& u, bool native) {
  for (int i = 0; i < 3; ++i) {
    if (auto ok = u.pass(native); !ok) return ok;
    if (auto ok = cuda_check(cudaStreamSynchronize(u.stream), "GMFSS natif : essai"); !ok) return ok;
  }
  return {};
}

std::expected<Planes, Refusal> fetch_all(const Unit& u) {
  Planes out;
  for (std::size_t i = 0; i < u.outputs; ++i) {
    auto got = u.fetch(i);
    if (!got) return std::unexpected(std::move(got.error()));
    out.push_back(std::move(*got));
  }
  return out;
}

double peak_of(const std::vector<__half>& values) {
  double peak = 1.0;
  for (const __half v : values)
    if (const double a = std::abs(static_cast<double>(__half2float(v))); std::isfinite(a))
      peak = std::max(peak, a);
  return peak;
}

// L'erreur quadratique moyenne de got contre want ; infinie sur une valeur non finie d'un côté ou de l'autre.
double mse_of(const std::vector<__half>& want, const std::vector<__half>& got) {
  double sum = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double a = __half2float(want[i]);
    const double b = __half2float(got[i]);
    if (!std::isfinite(a) || !std::isfinite(b)) return std::numeric_limits<double>::infinity();
    sum += (b - a) * (b - a);
  }
  return sum / static_cast<double>(std::max<std::size_t>(want.size(), 1));
}

// Une passe native aux sommes posées, ses sorties contre TensorRT et contre la base.
std::expected<native_mix::Eval, Refusal> evaluate(const Unit& u, const Planes& reference,
                                                  const Planes& base) {
  if (auto ok = settled(u, true); !ok) return std::unexpected(std::move(ok.error()));
  native_mix::Eval eval;
  for (std::size_t i = 0; i < u.outputs; ++i) {
    auto got = u.fetch(i);
    if (!got) return std::unexpected(std::move(got.error()));
    eval.want.push_back(mse_of(reference[i], *got));
    eval.base.push_back(mse_of(base[i], *got));
  }
  return eval;
}

// Le temps seul de chaque option de chaque morceau (cands), les autres morceaux au f32 ; une option hors de
// cands : l'infini.
std::expected<native_mix::Grid, Refusal> piece_times(const Unit& u,
                                                     const std::vector<std::vector<int>>& cands) {
  std::vector<std::pair<std::size_t, int>> variants;
  for (std::size_t p = 0; p < cands.size(); ++p)
    for (const int o : cands[p]) variants.emplace_back(p, o);
  std::vector<RifeNativePrecision> precisions(cands.size(), gmfss_mix::kLadder[gmfss_mix::kF32]);
  const auto pass = [&](std::size_t v) {
    const auto [p, o] = variants[v];
    std::ranges::fill(precisions, gmfss_mix::kLadder[gmfss_mix::kF32]);
    precisions[p] = gmfss_mix::kLadder[static_cast<std::size_t>(o)];
    u.parts->use(precisions);
    return u.piece(static_cast<int>(p));
  };
  auto ms = times(u.stream, variants.size(), pass, kPieceRounds);
  if (!ms) return std::unexpected(std::move(ms.error()));
  native_mix::Grid grid(
      cands.size(), std::vector<double>(gmfss_mix::kLadder.size(), std::numeric_limits<double>::infinity()));
  for (std::size_t v = 0; v < variants.size(); ++v)
    grid[variants[v].first][static_cast<std::size_t>(variants[v].second)] = (*ms)[v];
  return grid;
}

std::string tuning_line(const Unit& u, double seconds) {
  const GmfssTuning& t = u.parts->tuning();
  return std::format(
      "GMFSS natif, {} : {} convolutions réglées au banc pour les sommes f16, mixtes et f32 en {:.2f} s ({} "
      "formes mesurées, {} écartées ; {} sans choix ; {} convolutions reprises d'une jumelle)",
      u.name, t.layers, seconds, t.measured, t.rejected, t.single, t.reused);
}

// Une ligne par morceau : le temps seul de chaque option et son effet sur l'unité contre la base.
std::vector<std::string> piece_lines(const Unit& u, const native_mix::Problem& problem,
                                     const native_mix::Mix& mix) {
  std::vector<std::string> lines;
  for (std::size_t p = 0; p < problem.cands.size(); ++p) {
    std::string line =
        std::format("GMFSS natif, {}, {} seul :", u.name, u.parts->piece_name(static_cast<int>(p)));
    for (const int o : problem.cands[p]) {
      const auto at = static_cast<std::size_t>(o);
      line += std::format("{} {} {:.3f} ms", o == problem.cands[p].front() ? "" : " ;",
                          gmfss_mix::kLadderNames[at], problem.times[p][at]);
      line +=
          o == problem.base[p] ? std::string(" (base)") : std::format(" ({:.1f} dB)", mix.effect_db[p][at]);
    }
    lines.push_back(std::move(line));
  }
  return lines;
}

std::vector<std::string> piece_names(const GmfssNativeParts& parts) {
  std::vector<std::string> names(static_cast<std::size_t>(parts.pieces()));
  for (std::size_t p = 0; p < names.size(); ++p) names[p] = parts.piece_name(static_cast<int>(p));
  return names;
}

}  // namespace

std::expected<Outcome, Refusal> try_mix(const Unit& u, double min_db, double tuning) {
  const auto started = std::chrono::steady_clock::now();
  GmfssNativeParts& parts = *u.parts;
  const auto pieces = static_cast<std::size_t>(parts.pieces());
  Outcome out;
  out.lines.push_back(tuning_line(u, tuning));
  for (const std::string& note : parts.tuning().notes)
    out.lines.push_back(std::format("GMFSS natif, {}, réglage : {}", u.name, note));
  const auto give_up = [&](std::string why) {
    out.lines.push_back(
        std::format("GMFSS natif écarté, TensorRT garde la {} ({}×{} : {})", u.name, u.width, u.height, why));
    return out;
  };
  if (auto ok = settled(u, false); !ok) return std::unexpected(std::move(ok.error()));
  auto reference = fetch_all(u);
  if (!reference) return std::unexpected(std::move(reference.error()));
  parts.use(gmfss_mix::precisions(gmfss_mix::uniform(pieces, gmfss_mix::kF32)));
  if (auto ok = settled(u, true); !ok) return std::unexpected(std::move(ok.error()));
  auto base = fetch_all(u);
  if (!base) return std::unexpected(std::move(base.error()));
  std::vector<double> peak, floor, base_peak;
  for (std::size_t i = 0; i < u.outputs; ++i) {
    peak.push_back(peak_of((*reference)[i]));
    base_peak.push_back(peak_of((*base)[i]));
    floor.push_back(mse_of((*reference)[i], (*base)[i]));
  }
  if (!std::ranges::all_of(floor, [](double m) { return std::isfinite(m); }))
    return give_up("valeur non finie, de TensorRT ou du natif f32");
  std::vector<bool> heads(pieces);
  for (std::size_t p = 0; p < pieces; ++p) heads[p] = parts.has_head(static_cast<int>(p));
  auto times_of = piece_times(u, gmfss_mix::candidates(heads));
  if (!times_of) return std::unexpected(std::move(times_of.error()));
  const native_mix::Problem problem = gmfss_mix::problem(std::move(*times_of), heads, peak, floor, min_db);
  auto mix = native_mix::choose(problem, [&](const Options& o) {
    parts.use(gmfss_mix::precisions(o));
    return evaluate(u, *reference, *base);
  });
  if (!mix) return std::unexpected(std::move(mix.error()));
  std::ranges::move(piece_lines(u, problem, *mix), std::back_inserter(out.lines));
  out.lines.push_back(std::format(
      "GMFSS natif, {}, tout le réseau contre TensorRT : f32 {:.1f} dB, f16 {:.1f} dB, mixtes {:.1f} dB",
      u.name, native_mix::worst_db(peak, floor), mix->seed_db.at(0), mix->seed_db.at(1)));
  if (!mix->held) return give_up(std::format("aucun mélange ne tient {:.0f} dB", min_db));
  parts.use(gmfss_mix::precisions(mix->options));
  auto pass_ms = times(u.stream, 2, [&](std::size_t v) { return u.pass(v == 1); });
  if (!pass_ms) return std::unexpected(std::move(pass_ms.error()));
  out.keep = gmfss_mix::beats((*pass_ms)[0], (*pass_ms)[1]);
  out.lines.push_back(std::format(
      "{} ({}×{}, {} ; contre TensorRT {:.1f} dB (prévus {:.1f}), seuil {:.0f} dB ; contre le natif f32 : "
      "mélange {:.1f} dB, TensorRT {:.1f} dB ; une passe : TensorRT {:.2f} ms, mélange {:.2f} ms ; zone {} "
      "Mio ; "
      "réglage {:.1f} s, essai {:.1f} s)",
      out.keep ? std::format("GMFSS natif retenu pour la {}", u.name)
               : std::format("GMFSS natif écarté, TensorRT plus rapide pour la {}", u.name),
      u.width, u.height, gmfss_mix::mix_name(piece_names(parts), mix->options), mix->db, mix->predicted,
      min_db, native_mix::worst_db(base_peak, mix->base_mse), native_mix::worst_db(base_peak, floor),
      (*pass_ms)[0], (*pass_ms)[1], parts.workspace_bytes() >> 20, tuning,
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()));
  if (out.keep) parts.trim();
  return out;
}

}  // namespace kirari::gpu::cuda::gmfss_trial
