// Le choix d'un mélange de morceaux natifs (native_mix.h).
#include "gpu/cuda/backend/native_mix.h"

#include <algorithm>
#include <cmath>

namespace kirari::gpu::cuda::native_mix {

std::vector<double> group_peak(const std::vector<float>& want, const Spans& spans) {
  std::vector<double> peak;
  for (const auto& [first, end] : spans) {
    double top = 1.0;
    for (std::size_t i = first; i < end; ++i)
      if (std::isfinite(want[i])) top = std::max(top, static_cast<double>(std::abs(want[i])));
    peak.push_back(top);
  }
  return peak;
}

std::vector<double> group_mse(const std::vector<float>& want, const std::vector<float>& got,
                              const Spans& spans) {
  std::vector<double> mse;
  for (const auto& [first, end] : spans) {
    double sum = 0.0;
    for (std::size_t i = first; i < end && std::isfinite(sum); ++i) {
      const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
      sum = std::isfinite(got[i]) && std::isfinite(want[i]) ? sum + d * d
                                                            : std::numeric_limits<double>::infinity();
    }
    mse.push_back(sum / static_cast<double>(std::max<std::size_t>(end - first, 1)));
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

Spread spread_of(const Options& options, const Effects& effects, std::size_t groups) {
  Spread s{.power = std::vector<double>(groups, 0.0), .amplitude = std::vector<double>(groups, 0.0)};
  for (std::size_t p = 0; p < options.size(); ++p) {
    const std::vector<double>& effect = effects[p][static_cast<std::size_t>(options[p])];
    for (std::size_t g = 0; g < groups; ++g) {
      s.power[g] += effect[g];
      s.amplitude[g] += std::sqrt(effect[g]);
    }
  }
  for (double& a : s.amplitude) a *= a;
  return s;
}

std::vector<double> predicted(const Spread& s, const std::vector<double>& floor,
                              const std::vector<double>& rho) {
  std::vector<double> mse(s.power.size());
  for (std::size_t g = 0; g < mse.size(); ++g) {
    mse[g] = floor[g] + s.power[g];
    const double extra = s.amplitude[g] - s.power[g];
    if (extra > 0.0 && rho[g] > 0.0) mse[g] += rho[g] * extra;
  }
  return mse;
}

namespace {

const Measured* measured_of(const Bar& bar, const Options& options) {
  const auto found = std::ranges::find(bar.measured, options, &Measured::options);
  return found == bar.measured.end() ? nullptr : &*found;
}

}  // namespace

std::optional<Pick> fastest_within(const Problem& problem, const Effects& effects, const Bar& bar) {
  const std::size_t pieces = problem.cands.size();
  std::vector<std::size_t> at(pieces, 0);  // compteur : l'indice dans cands de chaque morceau
  std::optional<Pick> best;
  double best_ms = std::numeric_limits<double>::infinity();
  double best_db = -std::numeric_limits<double>::infinity();
  for (;;) {
    double ms = 0.0;
    Options options(pieces);
    for (std::size_t p = 0; p < pieces; ++p) {
      options[p] = problem.cands[p][at[p]];
      ms += problem.times[p][static_cast<std::size_t>(options[p])];
    }
    if (ms <= best_ms) {
      const Measured* known = measured_of(bar, options);
      std::vector<double> mse = known != nullptr ? known->mse
                                                 : predicted(spread_of(options, effects, problem.peak.size()),
                                                             problem.floor, bar.rho);
      const double db = worst_db(problem.peak, mse);
      if ((ms < best_ms || db > best_db) && db >= (known != nullptr ? bar.min_db : bar.multi_db)) {
        best_ms = ms;
        best_db = db;
        best = Pick{.options = std::move(options), .mse = std::move(mse)};
      }
    }
    std::size_t p = 0;
    while (p < pieces && ++at[p] == problem.cands[p].size()) at[p++] = 0;
    if (p == pieces) return best;
  }
}

void miss(Bar& bar, const Problem& problem, const Effects& effects, Pick pick, std::vector<double> measured,
          double slack) {
  const Spread s = spread_of(pick.options, effects, problem.peak.size());
  for (std::size_t g = 0; g < bar.rho.size(); ++g) {
    const double extra = s.amplitude[g] - s.power[g];
    if (extra <= 0.0) continue;
    const double rho = std::isfinite(measured[g]) ? (measured[g] - problem.floor[g] - s.power[g]) / extra
                                                  : std::numeric_limits<double>::infinity();
    bar.rho[g] = std::max(bar.rho[g], rho);
  }
  bar.multi_db = bar.min_db + slack;
  bar.measured.push_back({.options = std::move(pick.options), .mse = std::move(measured)});
}

}  // namespace kirari::gpu::cuda::native_mix
