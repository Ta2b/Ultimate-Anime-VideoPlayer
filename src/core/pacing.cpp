// Réécrit d'après :
//   - the-database/mpv@5afefa76 player/video.c:706-716 (calc_best_speed) : la vitesse
//     ratio / rint(ratio) qui cale une image sur un nombre entier de vsyncs, bornée
//     par un écart maximal. Seul le multiple 1 est gardé : un motif 1-2 saccade ;
//   - Ta2b/mpv-animejanai-hdr@55a8ae53 portable_config/hdr-tools/rife-sync.lua:8-50 :
//     petite fraction d'abord (dénominateur <= 4), sinon réduite de fraction continue.
// Comportement fixé par tests/unit/core/pacing_test.cpp.
#include "core/pacing.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "core/cadence.h"

namespace ajp::core {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::expected<void, Refusal> check_rates(const DisplayInfo& display, double source_fps) {
  // Écrit en positif pour qu'un NaN soit refusé lui aussi.
  if (!(source_fps > 0.0 && source_fps <= limits::kFps.hi)) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("fps source {} hors de ]0, 1000]", source_fps));
  }
  if (!display.vrr && !(display.refresh_hz > 0.0 && display.refresh_hz <= limits::kFps.hi)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("rafraîchissement {} Hz hors de ]0, 1000]", display.refresh_hz));
  }
  return {};
}

// a·b réduit, ou rien si un terme déborde.
std::optional<Ratio> multiply(Ratio a, Ratio b) {
  const std::int64_t g1 = std::gcd(a.num, b.den);
  const std::int64_t g2 = std::gcd(b.num, a.den);
  const std::int64_t n1 = a.num / g1;
  const std::int64_t n2 = b.num / g2;
  const std::int64_t d1 = a.den / g2;
  const std::int64_t d2 = b.den / g1;
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  if (n1 > kMax / n2 || d1 > kMax / d2) return std::nullopt;
  return Ratio{n1 * n2, d1 * d2};
}

// La réduite la plus fine de x (x >= 1) dont les deux termes restent dans kRatioTerm.
// Un bruit d'arrondi de x ne crée qu'un quotient partiel immense, coupé par la borne.
Ratio continued_fraction(double x) {
  std::int64_t h0 = 0;
  std::int64_t h1 = 1;
  std::int64_t k0 = 1;
  std::int64_t k1 = 0;
  double y = x;
  for (int i = 0; i < 64; ++i) {
    const double a = std::floor(y);
    if (a > static_cast<double>(limits::kRatioTerm.hi)) break;
    const auto ai = static_cast<std::int64_t>(a);
    const std::int64_t h2 = ai * h1 + h0;
    const std::int64_t k2 = ai * k1 + k0;
    if (h2 > limits::kRatioTerm.hi || k2 > limits::kRatioTerm.hi) break;
    h0 = std::exchange(h1, h2);
    k0 = std::exchange(k1, k2);
    const double rest = y - a;
    if (rest < 1e-12) break;
    y = 1.0 / rest;
  }
  return {h1, k1};
}

// Le facteur qui cale la cible : une petite fraction à kMaxSpeedChange près,
// sinon la fraction exacte, ou rien.
std::optional<Ratio> lock_factor(double target) {
  for (std::int64_t q = 1; q <= pacing::kMaxSimpleDen; ++q) {
    const Ratio simple{std::llround(target * static_cast<double>(q)), q};
    if (simple.num > q && std::abs(target / simple.value() - 1.0) <= pacing::kMaxSpeedChange) {
      return reduced(simple);
    }
  }
  const Ratio exact = continued_fraction(target);
  if (exact.num > exact.den && std::abs(target / exact.value() - 1.0) <= pacing::kExactTolerance) {
    return exact;
  }
  return std::nullopt;
}

CadenceChoice on_timestamps(Ratio factor, double source_fps) {
  return CadenceChoice{.factor = factor, .output_fps = source_fps * factor.value()};
}

// Le plus proche du facteur demandé, en rapport ; à égalité, le moins coûteux.
bool closer(const CadenceChoice& a, const CadenceChoice& b, double requested) {
  const double da = std::abs(std::log(a.factor.value() / requested));
  const double db = std::abs(std::log(b.factor.value() / requested));
  if (std::abs(da - db) > 1e-12) return da < db;
  return a.factor.value() < b.factor.value();
}

}  // namespace

double vsync_speed(double refresh_hz, double output_fps) {
  if (!(refresh_hz > 0.0 && output_fps > 0.0)) return 1.0;
  const long long vsyncs = std::llround(refresh_hz / output_fps);
  if (vsyncs < 1 || vsyncs > pacing::kMaxVsyncsPerFrame) return 1.0;
  const double speed = refresh_hz / (static_cast<double>(vsyncs) * output_fps);
  return std::abs(speed - 1.0) <= pacing::kMaxSpeedChange ? speed : 1.0;
}

std::expected<CadenceChoice, Refusal> fit_cadence(const DisplayInfo& display, double source_fps,
                                                  Ratio factor) {
  if (auto ok = check_rates(display, source_fps); !ok) return std::unexpected(ok.error());
  if (factor.den < 1 || factor.num < factor.den || factor.value() > limits::kMaxFactorProduct) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("facteur {}/{} hors de [1, 16]", factor.num, factor.den));
  }
  CadenceChoice choice = on_timestamps(factor, source_fps);
  if (display.vrr) return choice;

  const long long vsyncs = std::llround(display.refresh_hz / choice.output_fps);
  if (vsyncs < 1 || vsyncs > pacing::kMaxVsyncsPerFrame) return choice;
  const double speed = display.refresh_hz / (static_cast<double>(vsyncs) * choice.output_fps);
  if (std::abs(speed - 1.0) > pacing::kMaxSpeedChange) return choice;
  choice.grid = Grid::Vsync;
  choice.vsyncs_per_frame = static_cast<int>(vsyncs);
  choice.speed = speed;
  return choice;
}

std::expected<CadenceChoice, Refusal> choose_cadence(const DisplayInfo& display, double source_fps,
                                                     Ratio requested, double max_factor) {
  if (auto ok = check_rates(display, source_fps); !ok) return std::unexpected(ok.error());
  const std::string where = std::format("facteur {}/{}", requested.num, requested.den);
  if (!limits::kRatioTerm.contains(requested.num) || !limits::kRatioTerm.contains(requested.den) ||
      requested.value() <= limits::kInterpFactor.lo || requested.value() > limits::kInterpFactor.hi) {
    return refuse(RefusalCode::FieldOutOfRange, where + " hors de ]1, 8]");
  }
  if (requested.value() > max_factor) {
    return refuse(RefusalCode::FactorProductTooLarge, std::format("{} au-delà de {}", where, max_factor));
  }
  if (display.vrr) return on_timestamps(requested, source_fps);

  const double limit = std::min(max_factor, limits::kInterpFactor.hi);
  std::optional<CadenceChoice> best;
  for (int k = 1; k <= pacing::kMaxVsyncsPerFrame; ++k) {
    const double target = display.refresh_hz / (k * source_fps);
    // La source se cale déjà sur k vsyncs : au-delà, plus rien à interpoler.
    if (target <= 1.0 + pacing::kMaxSpeedChange) break;
    const std::optional<Ratio> factor = lock_factor(target);
    if (!factor || factor->value() > limit) continue;
    const CadenceChoice candidate{.factor = *factor,
                                  .grid = Grid::Vsync,
                                  .vsyncs_per_frame = k,
                                  .speed = target / factor->value(),
                                  .output_fps = source_fps * factor->value()};
    if (!best || closer(candidate, *best, requested.value())) best = candidate;
  }
  if (best) return *best;
  return fit_cadence(display, source_fps, requested);
}

std::expected<PacedRule, Refusal> pace_rule(const Rule& rule, const DisplayInfo& display, double source_fps) {
  std::size_t interpolations = 0;
  std::size_t index = 0;
  double upscale = 1.0;
  Ratio product{1, 1};
  for (std::size_t i = 0; i < rule.steps.size(); ++i) {
    const Step& step = rule.steps[i];
    if (step.kind == StepKind::Downscale) continue;
    if (!limits::kRatioTerm.contains(step.factor.num) || !limits::kRatioTerm.contains(step.factor.den)) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("étape {} : terme de facteur hors bornes", i + 1));
    }
    if (step.kind == StepKind::Upscale) {
      upscale *= step.factor.value();
      continue;
    }
    ++interpolations;
    index = i;
    const std::optional<Ratio> next = multiply(product, step.factor);
    if (!next) return refuse(RefusalCode::FactorProductTooLarge, "produit des facteurs non représentable");
    product = *next;
  }

  PacedRule paced{.rule = rule};
  if (interpolations != 1) {
    auto choice = fit_cadence(display, source_fps, product);
    if (!choice) return std::unexpected(choice.error());
    paced.cadence = *choice;
    return paced;
  }
  Step& step = paced.rule.steps[index];
  auto choice = choose_cadence(display, source_fps, step.factor, limits::kMaxFactorProduct / upscale);
  if (!choice) return std::unexpected(choice.error());
  step.factor = choice->factor;
  paced.cadence = *choice;
  return paced;
}

}  // namespace ajp::core
