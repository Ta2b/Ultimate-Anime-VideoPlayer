// Le mélange du RIFE natif sur native_mix.h (rife_mix.h) choisit comme le choix d'avant, recopié ici tel
// qu'il était dans rife_native_mix.cpp : mêmes passes sur la chaîne, dans le même ordre, même mélange retenu,
// mêmes dB mesurés et prévus, mêmes effets, sur des problèmes tirés au hasard (temps, effets, corrélation et
// écarts de la chaîne).
#include "gpu/cuda/backend/rife_mix.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <expected>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/cuda/backend/native_mix.h"

namespace kirari::gpu::cuda {
namespace {

// Le choix d'avant native_mix.h, à la chaîne près : chain rend l'erreur de chaque groupe contre TensorRT.
namespace before {

using Options = std::vector<int>;
using Grid = std::vector<std::vector<double>>;
using Effects = std::vector<std::vector<std::vector<double>>>;

double db_of(double peak, double mse) {
  if (!std::isfinite(mse)) return -std::numeric_limits<double>::infinity();
  return mse == 0.0 ? 200.0 : 10.0 * std::log10(peak * peak / mse);
}

double worst_db(const std::vector<double>& peak, const std::vector<double>& mse) {
  double worst = std::numeric_limits<double>::infinity();
  for (std::size_t g = 0; g < peak.size(); ++g) worst = std::min(worst, db_of(peak[g], mse[g]));
  return worst;
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

struct Spread {
  std::vector<double> power;
  std::vector<double> amplitude;
  int natives = 0;
};

Spread spread_of(const Options& options, const Effects& effects, std::size_t groups) {
  Spread s{.power = std::vector<double>(groups, 0.0), .amplitude = std::vector<double>(groups, 0.0)};
  for (std::size_t p = 0; p < options.size(); ++p) {
    const auto o = static_cast<std::size_t>(options[p]);
    s.natives += o == 0 ? 0 : 1;
    for (std::size_t g = 0; g < groups; ++g) {
      s.power[g] += effects[p][o][g];
      s.amplitude[g] += std::sqrt(effects[p][o][g]);
    }
  }
  for (double& a : s.amplitude) a *= a;
  return s;
}

struct Bar {
  double min_db = 0.0;
  double multi_db = 0.0;
  std::vector<double> rho;
  std::vector<Options> failed;
};

std::vector<double> predicted(const Spread& s, const std::vector<double>& rho) {
  std::vector<double> mse = s.power;
  if (s.natives <= 1) return mse;
  for (std::size_t g = 0; g < mse.size(); ++g) {
    const double extra = s.amplitude[g] - s.power[g];
    if (extra > 0.0 && rho[g] > 0.0) mse[g] += rho[g] * extra;
  }
  return mse;
}

bool holds(const Bar& bar, const std::vector<double>& peak, const std::vector<double>& mse, int natives) {
  for (std::size_t g = 0; g < peak.size(); ++g)
    if (db_of(peak[g], mse[g]) < (natives <= 1 ? bar.min_db : bar.multi_db)) return false;
  return true;
}

std::optional<std::pair<Options, std::vector<double>>> fastest_within(
    const Grid& times, const std::vector<std::vector<int>>& cands, const Effects& effects,
    const std::vector<double>& peak, const Bar& bar) {
  const std::size_t pieces = cands.size();
  std::vector<std::size_t> at(pieces, 0);
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
      const Spread s = spread_of(options, effects, peak.size());
      std::vector<double> mse = predicted(s, bar.rho);
      const double db = worst_db(peak, mse);
      if ((ms < best_ms || db > best_db) && holds(bar, peak, mse, s.natives)) {
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

void miss(Bar& bar, const Spread& s, const std::vector<double>& measured, Options options, double slack) {
  for (std::size_t g = 0; g < bar.rho.size(); ++g) {
    const double extra = s.amplitude[g] - s.power[g];
    if (extra <= 0.0) continue;
    const double rho = std::isfinite(measured[g]) ? (measured[g] - s.power[g]) / extra
                                                  : std::numeric_limits<double>::infinity();
    bar.rho[g] = std::max(bar.rho[g], rho);
  }
  bar.multi_db = bar.min_db + slack;
  bar.failed.push_back(std::move(options));
}

struct Mix {
  Options options;
  double db = 0.0;
  double predicted = 0.0;
  std::vector<double> effect_db;  // dans l'ordre de la ligne de journal des effets
};

template <class Chain>
Mix choose_mix(const Grid& times, const std::vector<double>& peak, double min_db, Chain&& chain) {
  constexpr int kAttempts = 4;
  constexpr double kSlack = 0.5;
  const auto cands = candidates(times);
  Effects effects(times.size(), std::vector<std::vector<double>>(times.front().size()));
  Mix mix{.options = Options(times.size(), 0)};
  for (std::size_t p = 0; p < times.size(); ++p) {
    effects[p][0].assign(peak.size(), 0.0);
    for (const int o : cands[p]) {
      if (o == 0) continue;
      Options alone(times.size(), 0);
      alone[p] = o;
      auto& effect = effects[p][static_cast<std::size_t>(o)];
      effect = chain(alone);
      mix.effect_db.push_back(worst_db(peak, effect));
    }
  }
  Bar bar{.min_db = min_db, .multi_db = min_db, .rho = std::vector<double>(peak.size(), 0.0)};
  for (int attempt = 0; attempt < kAttempts; ++attempt) {
    if (attempt + 1 == kAttempts) bar.multi_db = std::numeric_limits<double>::infinity();
    auto pick = fastest_within(times, cands, effects, peak, bar);
    if (!pick || all_trt(pick->first)) break;
    const std::vector<double> mse = chain(pick->first);
    if (worst_db(peak, mse) >= min_db) {
      mix.options = std::move(pick->first);
      mix.db = worst_db(peak, mse);
      mix.predicted = worst_db(peak, pick->second);
      return mix;
    }
    const Spread spread = spread_of(pick->first, effects, peak.size());
    miss(bar, spread, mse, std::move(pick->first), kSlack);
  }
  mix.options.assign(times.size(), 0);
  return mix;
}

}  // namespace before

// Une chaîne tirée au hasard, mais fixe : l'erreur d'un mélange est celle de ses morceaux seuls, mêlée
// selon une corrélation (de la somme au carré de la somme des racines), et un écart propre au mélange ; le
// même mélange rend toujours la même erreur.
struct Chain {
  native_mix::Effects effect;  // [morceau][option][groupe], nul pour TensorRT
  double rho = 0.0;
  unsigned salt = 0;

  std::vector<double> operator()(const native_mix::Options& options) const {
    const std::size_t groups = effect.front().front().size();
    std::vector<double> mse(groups, 0.0);
    unsigned h = salt;
    for (const int o : options) h = h * 31U + static_cast<unsigned>(o) + 7U;
    for (std::size_t g = 0; g < groups; ++g) {
      double power = 0.0, amplitude = 0.0;
      for (std::size_t p = 0; p < options.size(); ++p) {
        const double e = effect[p][static_cast<std::size_t>(options[p])][g];
        power += e;
        amplitude += std::sqrt(e);
      }
      const double noise = 0.6 + 0.8 * static_cast<double>((h >> (3 * g)) % 101U) / 100.0;
      mse[g] = (power + rho * (amplitude * amplitude - power)) * noise;
    }
    return mse;
  }
};

TEST_CASE("RIFE natif : le mélange sur native_mix.h choisit comme celui d'avant", "[gpu][native_mix]") {
  std::mt19937 rng(4242U);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  int natives = 0, misses = 0, refusals = 0;
  for (int trial = 0; trial < 600; ++trial) {
    const auto pieces = static_cast<std::size_t>(1 + trial % 5);
    const std::size_t groups = trial % 3 == 0 ? 2 : 1;
    native_mix::Grid times(pieces, std::vector<double>(5));
    Chain chain{.effect = native_mix::Effects(pieces, native_mix::Grid(5, std::vector<double>(groups, 0.0))),
                .rho = unit(rng),
                .salt = static_cast<unsigned>(trial)};
    for (std::size_t p = 0; p < pieces; ++p)
      for (std::size_t o = 0; o < 5; ++o) {
        times[p][o] = 0.1 + unit(rng);  // un réseau peut être plus lent que TensorRT
        for (std::size_t g = 0; g < groups && o > 0; ++g)
          chain.effect[p][o][g] = std::pow(10.0, -(3.4 + 3.4 * unit(rng)));  // de 34 à 68 dB
      }
    std::vector<double> peak(groups, 1.0);
    if (groups == 2) peak[0] = 1.0 + 20.0 * unit(rng);  // un flot en pixels
    const double min_db = 45.0;

    std::vector<native_mix::Options> old_calls;
    const before::Mix old = before::choose_mix(times, peak, min_db, [&](const native_mix::Options& o) {
      old_calls.push_back(o);
      return chain(o);
    });

    std::vector<native_mix::Options> new_calls;
    const native_mix::Problem problem = rife_mix::problem(times, peak, min_db);
    const auto chosen = native_mix::choose(
        problem, [&](const native_mix::Options& o) -> std::expected<native_mix::Eval, std::string> {
          new_calls.push_back(o);
          return rife_mix::eval(chain(o));
        });
    REQUIRE(chosen);
    const native_mix::Options options = rife_mix::kept(*chosen, pieces);
    INFO("essai " << trial);
    CHECK(new_calls == old_calls);
    CHECK(options == old.options);
    std::vector<double> effect_db;
    for (std::size_t p = 0; p < pieces; ++p)
      for (const int o : problem.cands[p])
        if (o != 0) effect_db.push_back(chosen->effect_db[p][static_cast<std::size_t>(o)]);
    CHECK(effect_db == old.effect_db);
    if (!before::all_trt(options)) {
      ++natives;
      CHECK(chosen->db == old.db);
      CHECK(chosen->predicted == old.predicted);
    } else {
      ++refusals;
    }
    const auto attempts = static_cast<int>(new_calls.size() - old.effect_db.size());
    misses += attempts - (before::all_trt(options) ? 0 : 1);
  }
  // Le tirage couvre les trois issues : un mélange retenu, des vérifications manquées, rien qui tienne.
  CHECK(natives > 50);
  CHECK(refusals > 20);
  CHECK(misses > 20);
}

}  // namespace
}  // namespace kirari::gpu::cuda
