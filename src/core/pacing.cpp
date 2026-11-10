// Réécrit d'après :
//   - the-database/mpv@5afefa76 player/video.c:706-716 (calc_best_speed) : la vitesse
//     ratio / rint(ratio) qui cale une image sur un nombre entier de vsyncs, bornée
//     par un écart maximal. Seul le multiple 1 est gardé : un motif 1-2 saccade ;
//   - Ta2b/mpv-animejanai-hdr@55a8ae53 portable_config/hdr-tools/rife-sync.lua:8-50 :
//     petite fraction d'abord (dénominateur <= 4), sinon réduite de fraction continue.
// Comportement fixé par tests/unit/core/pacing_test.cpp.
#include "core/pacing.h"

#include <cmath>
#include <optional>
#include <utility>

#include "core/cadence.h"

namespace ajp::core {
namespace {

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

}  // namespace

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

std::optional<double> vsync_lock(double refresh_hz, double output_fps) {
  if (!(refresh_hz > 0.0 && output_fps > 0.0)) return std::nullopt;
  const long long vsyncs = std::llround(refresh_hz / output_fps);
  if (vsyncs < 1 || vsyncs > pacing::kMaxVsyncsPerFrame) return std::nullopt;
  const double speed = refresh_hz / (static_cast<double>(vsyncs) * output_fps);
  if (std::abs(speed - 1.0) > pacing::kMaxSpeedChange) return std::nullopt;
  return speed;
}

double vsync_speed(double refresh_hz, double output_fps) {
  return vsync_lock(refresh_hz, output_fps).value_or(1.0);
}

}  // namespace ajp::core
