// Le choix du mélange du RIFE natif, en C++ pur (rife_mix_choice.h).
#include "gpu/cuda/backend/rife_mix_choice.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <optional>

namespace kirari::gpu::cuda::rife_mix {
namespace {

// L'effet de chaque option sur la chaîne : [morceau][option], l'erreur quadratique de chaque groupe (vide
// pour une option que le morceau ne prend pas ; nulle pour base).
using Effects = std::vector<std::vector<std::vector<double>>>;

// Les erreurs d'un mélange, groupe par groupe, à partir de l'effet de ses morceaux : leur somme (des erreurs
// indépendantes s'ajoutent ainsi) et le carré de la somme de leurs racines (des erreurs de même signe aux
// mêmes pixels) ; le nombre de morceaux hors de base.
struct Spread {
  std::vector<double> power;
  std::vector<double> amplitude;
  int apart = 0;
};

Spread spread_of(const Options& options, const Effects& effects, std::size_t groups, int base) {
  Spread s{.power = std::vector<double>(groups, 0.0), .amplitude = std::vector<double>(groups, 0.0)};
  for (std::size_t p = 0; p < options.size(); ++p) {
    const auto o = static_cast<std::size_t>(options[p]);
    s.apart += options[p] == base ? 0 : 1;
    for (std::size_t g = 0; g < groups; ++g) {
      s.power[g] += effects[p][o][g];
      s.amplitude[g] += std::sqrt(effects[p][o][g]);
    }
  }
  for (double& a : s.amplitude) a *= a;
  return s;
}

// Ce qu'un mélange doit tenir. L'erreur prévue d'un groupe va de la somme (rho 0, au départ) au carré de la
// somme des racines (rho 1) ; un mélange d'au plus un morceau hors de base a été mesuré tel quel (son effet)
// et tient bar ou non. Les autres tiennent bar plus multi : 0, puis kSlack après une vérification manquée,
// qui recale aussi rho ; les mélanges manqués sont écartés.
struct Bar {
  std::vector<double> bar;
  double multi = 0.0;
  std::vector<double> rho;
  std::vector<Options> failed;
};

std::vector<double> predicted(const Spread& s, const std::vector<double>& rho) {
  std::vector<double> mse = s.power;
  if (s.apart <= 1) return mse;
  for (std::size_t g = 0; g < mse.size(); ++g) {
    const double extra = s.amplitude[g] - s.power[g];
    if (extra > 0.0 && rho[g] > 0.0) mse[g] += rho[g] * extra;
  }
  return mse;
}

bool holds(const Bar& bar, const std::vector<double>& peak, const std::vector<double>& mse, int apart) {
  for (std::size_t g = 0; g < peak.size(); ++g)
    if (db_of(peak[g], mse[g]) < bar.bar[g] + (apart <= 1 ? 0.0 : bar.multi)) return false;
  return true;
}

// Le mélange le plus rapide parmi cands (la somme des temps seuls) dont l'erreur prévue tient bar ; à temps
// égal, le plus juste. Ses options et son erreur prévue ; rien si aucun ne tient.
std::optional<std::pair<Options, std::vector<double>>> fastest_within(
    const Grid& times, const std::vector<std::vector<int>>& cands, const Effects& effects,
    const std::vector<double>& peak, const Bar& bar, int base) {
  const std::size_t pieces = cands.size();
  std::vector<std::size_t> at(pieces, 0);  // compteur : l'indice dans cands de chaque morceau
  std::optional<std::pair<Options, std::vector<double>>> best;
  double best_ms = std::numeric_limits<double>::infinity();
  double best_db = -std::numeric_limits<double>::infinity();
  for (;;) {
    double ms = 0.0;
    Options options(pieces);
    for (std::size_t p = 0; p < pieces; ++p) {
      options[p] = cands[p][at[p]];
      ms += times[p][static_cast<std::size_t>(options[p])];
    }
    if (ms <= best_ms && std::ranges::find(bar.failed, options) == bar.failed.end()) {
      const Spread s = spread_of(options, effects, peak.size(), base);
      std::vector<double> mse = predicted(s, bar.rho);
      const double db = worst_db(peak, mse);
      if ((ms < best_ms || db > best_db) && holds(bar, peak, mse, s.apart)) {
        best_ms = ms;
        best_db = db;
        best = std::pair{std::move(options), std::move(mse)};
      }
    }
    std::size_t p = 0;
    while (p < pieces && ++at[p] == cands[p].size()) at[p++] = 0;
    if (p == pieces) return best;
  }
}

// Après une vérification manquée : rho de chaque groupe recalé sur l'erreur mesurée (sans jamais descendre ;
// infini sur une erreur non finie), la marge des mélanges de plusieurs morceaux hors de base à slack, le
// mélange écarté.
void miss(Bar& bar, const Spread& s, const std::vector<double>& measured, Options options, double slack) {
  for (std::size_t g = 0; g < bar.rho.size(); ++g) {
    const double extra = s.amplitude[g] - s.power[g];
    if (extra <= 0.0) continue;
    const double rho = std::isfinite(measured[g]) ? (measured[g] - s.power[g]) / extra
                                                  : std::numeric_limits<double>::infinity();
    bar.rho[g] = std::max(bar.rho[g], rho);
  }
  bar.multi = slack;
  bar.failed.push_back(std::move(options));
}

bool within(const std::vector<double>& db, const std::vector<double>& bar) {
  for (std::size_t g = 0; g < db.size(); ++g)
    if (!(db[g] >= bar[g])) return false;
  return true;
}

}  // namespace

std::string piece_name(int piece) {
  return piece == 0 ? std::string("encodeur") : std::format("bloc {}", piece - 1);
}

std::vector<std::pair<std::size_t, std::size_t>> groups_of(bool motion) {
  if (motion) return {{0, 4}, {4, 5}};
  return {{0, 3}};
}

std::vector<double> group_peak(const std::vector<float>& want, std::size_t plane, bool motion) {
  std::vector<double> peak;
  for (const auto& [first, end] : groups_of(motion)) {
    double top = 1.0;
    for (std::size_t i = first * plane; i < end * plane; ++i)
      if (std::isfinite(want[i])) top = std::max(top, static_cast<double>(std::abs(want[i])));
    peak.push_back(top);
  }
  return peak;
}

std::vector<double> group_mse(const std::vector<float>& want, const std::vector<float>& got,
                              std::size_t plane, bool motion) {
  std::vector<double> mse;
  for (const auto& [first, end] : groups_of(motion)) {
    double sum = 0.0;
    for (std::size_t i = first * plane; i < end * plane && std::isfinite(sum); ++i) {
      const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
      sum = std::isfinite(got[i]) && std::isfinite(want[i]) ? sum + d * d
                                                            : std::numeric_limits<double>::infinity();
    }
    mse.push_back(sum / static_cast<double>((end - first) * plane));
  }
  return mse;
}

double db_of(double peak, double mse) {
  if (!std::isfinite(mse)) return -std::numeric_limits<double>::infinity();
  return mse == 0.0 ? 200.0 : 10.0 * std::log10(peak * peak / mse);
}

double worst_db(const std::vector<double>& peak, const std::vector<double>& mse) {
  double worst = std::numeric_limits<double>::infinity();
  for (std::size_t g = 0; g < peak.size(); ++g) worst = std::min(worst, db_of(peak[g], mse[g]));
  return worst;
}

std::vector<double> group_db(const std::vector<float>& want, const std::vector<float>& got, std::size_t plane,
                             bool motion) {
  const std::vector<double> peak = group_peak(want, plane, motion);
  const std::vector<double> mse = group_mse(want, got, plane, motion);
  std::vector<double> db(peak.size());
  for (std::size_t g = 0; g < peak.size(); ++g) db[g] = db_of(peak[g], mse[g]);
  return db;
}

double worst_psnr(const std::vector<float>& want, const std::vector<float>& got, std::size_t plane,
                  bool motion) {
  return worst_db(group_peak(want, plane, motion), group_mse(want, got, plane, motion));
}

std::vector<std::vector<int>> candidates(const Grid& times) {
  std::vector<std::vector<int>> out(times.size());
  for (std::size_t p = 0; p < times.size(); ++p) {
    out[p].push_back(0);
    for (std::size_t o = 1; o < times[p].size(); ++o)
      if (times[p][o] < times[p][0]) out[p].push_back(static_cast<int>(o));
  }
  return out;
}

bool all_trt(const Options& options) {
  return std::ranges::all_of(options, [](int o) { return o == 0; });
}

Options fastest_options(const Grid& times) {
  Options options(times.size());
  for (std::size_t p = 0; p < times.size(); ++p)
    options[p] = static_cast<int>(std::ranges::min_element(times[p]) - times[p].begin());
  return options;
}

Reference reference_of(const std::vector<double>& pieces_db, const std::vector<double>& trt_db,
                       double min_db) {
  Reference ref{.bar_db = std::vector<double>(trt_db.size(), min_db)};
  if (pieces_db.empty()) {
    ref.why = "aucun morceau du natif f32 mesuré";
    return ref;
  }
  for (std::size_t p = 0; p < pieces_db.size(); ++p)
    if (!(pieces_db[p] >= kPieceFloorDb)) {
      ref.why = std::format("{} du natif f32 à {:.1f} dB de TensorRT, moins de {:.0f}",
                            piece_name(static_cast<int>(p)), pieces_db[p], kPieceFloorDb);
      return ref;
    }
  // Groupe par groupe : un NaN ne passe pas (std::ranges::min le laisserait derrière une valeur finie).
  double trt = std::numeric_limits<double>::infinity();
  for (const double db : trt_db) trt = std::isnan(db) || std::isnan(trt) ? std::nan("") : std::min(trt, db);
  if (trt_db.empty() || !(trt >= kChainFloorDb)) {
    ref.why =
        std::format("TensorRT à {:.1f} dB du natif f32 sur la chaîne, moins de {:.0f}", trt, kChainFloorDb);
    return ref;
  }
  ref.native = true;
  for (std::size_t g = 0; g < trt_db.size(); ++g) ref.bar_db[g] = std::min(min_db, trt_db[g]);
  ref.why = std::format("TensorRT à {:.1f} dB du natif f32", trt);
  return ref;
}

std::expected<Mix, Refusal> choose_mix(const Grid& times, const Judge& judge, const Chain& chain) {
  constexpr int kAttempts = 4;
  constexpr double kSlack = 0.5;
  const std::vector<float>& want = *judge.want;
  const auto cands = candidates(times);
  const std::vector<double> peak = group_peak(want, judge.plane, judge.motion);
  Effects effects(times.size(), std::vector<std::vector<double>>(times.front().size()));
  Mix mix{.options = Options(times.size(), 0)};
  for (std::size_t p = 0; p < times.size(); ++p) {
    effects[p][static_cast<std::size_t>(judge.base)].assign(peak.size(), 0.0);
    for (const int o : cands[p]) {
      if (o == judge.base) continue;
      Options alone(times.size(), judge.base);
      alone[p] = o;
      auto got = chain(alone);
      if (!got) return std::unexpected(std::move(got.error()));
      auto& effect = effects[p][static_cast<std::size_t>(o)];
      effect = group_mse(want, *got, judge.plane, judge.motion);
      mix.effects.push_back({.piece = static_cast<int>(p), .option = o, .db = worst_db(peak, effect)});
    }
  }
  Bar bar{.bar = judge.bar_db, .rho = std::vector<double>(peak.size(), 0.0)};
  for (int attempt = 0; attempt < kAttempts; ++attempt) {
    if (attempt + 1 == kAttempts) bar.multi = std::numeric_limits<double>::infinity();
    auto pick = fastest_within(times, cands, effects, peak, bar, judge.base);
    if (!pick || all_trt(pick->first)) break;
    auto got = chain(pick->first);
    if (!got) return std::unexpected(std::move(got.error()));
    const std::vector<double> mse = group_mse(want, *got, judge.plane, judge.motion);
    std::vector<double> db(peak.size());
    for (std::size_t g = 0; g < peak.size(); ++g) db[g] = db_of(peak[g], mse[g]);
    if (within(db, judge.bar_db)) {
      mix.options = std::move(pick->first);
      mix.out = std::move(*got);
      mix.db = worst_db(peak, mse);
      mix.predicted = worst_db(peak, pick->second);
      return mix;
    }
    const Spread spread = spread_of(pick->first, effects, peak.size(), judge.base);
    miss(bar, spread, mse, std::move(pick->first), kSlack);
  }
  mix.options.assign(times.size(), 0);
  return mix;
}

}  // namespace kirari::gpu::cuda::rife_mix
