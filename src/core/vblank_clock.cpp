#include "core/vblank_clock.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ajp::core {

void VblankClock::reset() {
  times_.clear();
  latencies_.clear();
  period_guess_us_ = 0.0;
  rejects_ = 0;
  have_last_ = false;
}

void VblankClock::set_nominal_period(double period_us) {
  if (!(period_us > 0.0)) period_us = 0.0;
  const bool moved = nominal_us_ > 0.0 ? std::abs(period_us / nominal_us_ - 1.0) > 0.005 : period_us > 0.0;
  nominal_us_ = period_us;
  if (moved && have_last_) reset();
}

void VblankClock::restart(std::int64_t visible_us) {
  times_.clear();
  rejects_ = 0;
  last_us_ = visible_us;
  last_k_ = 0;
  have_last_ = true;
  times_.push_back({visible_us, 0});
}

void VblankClock::push(std::int64_t visible_us, std::int64_t steps) {
  rejects_ = 0;
  last_k_ += steps;
  last_us_ = visible_us;
  times_.push_back({visible_us, last_k_});
  if (times_.size() > kWindow) times_.pop_front();
}

void VblankClock::observe(std::int64_t visible_us, std::uint64_t /*id*/, std::int64_t target_us) {
  if (target_us > 0) {
    latencies_.push_back(visible_us - target_us);
    if (latencies_.size() > kWindow) latencies_.pop_front();
  }
  if (!have_last_) {
    restart(visible_us);
    return;
  }
  if (visible_us <= last_us_) return;  // hors d'ordre ou doublon : sans effet sur la grille
  if (visible_us - last_us_ > kMaxGapUs) {
    restart(visible_us);
    return;
  }
  const double gap = static_cast<double>(visible_us - last_us_);
  if (nominal_us_ > 0.0) {
    // Le pas : la période annoncée, affinée par la régression tant qu'elle en reste proche (164,78 Hz
    // pour 165 annoncés ; jamais la cadence du contenu, à des multiples de là).
    double period = nominal_us_;
    if (const auto fitted = grid()) period = fitted->period_us;
    const double ratio = gap / period;
    const std::int64_t steps = std::max<std::int64_t>(1, std::llround(ratio));
    if (std::abs(ratio - static_cast<double>(steps)) > kOffGrid) {
      if (++rejects_ >= kMaxRejects) restart(visible_us);
      return;
    }
    push(visible_us, steps);
    return;
  }
  // Sans période annoncée : le plus petit écart vu (un vblank), que la régression affine sans jamais le
  // remplacer par plus long. Un écart nettement plus court dément les numéros déjà donnés : on repart.
  if (period_guess_us_ == 0.0) period_guess_us_ = gap;
  if (gap < period_guess_us_ * 0.75) {
    period_guess_us_ = gap;
    const std::int64_t previous = last_us_;
    restart(previous);
    push(visible_us, 1);
    return;
  }
  const std::int64_t steps = std::max<std::int64_t>(1, std::llround(gap / period_guess_us_));
  push(visible_us, steps);
  if (const auto fitted = grid(); fitted && std::abs(fitted->period_us / period_guess_us_ - 1.0) < 0.1)
    period_guess_us_ = fitted->period_us;
}

std::optional<VblankGrid> VblankClock::grid() const {
  if (times_.size() < kMinSamples) return std::nullopt;
  const double n = static_cast<double>(times_.size());
  const std::int64_t k0 = times_.front().k;
  const std::int64_t t0 = times_.front().visible_us;
  double sk = 0.0;
  double st = 0.0;
  double skk = 0.0;
  double skt = 0.0;
  for (const Sample& s : times_) {
    const double k = static_cast<double>(s.k - k0);
    const double t = static_cast<double>(s.visible_us - t0);
    sk += k;
    st += t;
    skk += k * k;
    skt += k * t;
  }
  const double denom = n * skk - sk * sk;
  if (denom <= 0.0) return std::nullopt;
  const double period = (n * skt - sk * st) / denom;
  if (period <= 0.0) return std::nullopt;
  // Loin de la période annoncée : des numéros faux (dates hors grille), pas une grille.
  if (nominal_us_ > 0.0 && std::abs(period / nominal_us_ - 1.0) > kNominalTolerance) return std::nullopt;
  const double phase = (st - period * sk) / n;
  return VblankGrid{.phase_us = static_cast<double>(t0) + phase - period * static_cast<double>(k0),
                    .period_us = period};
}

double VblankClock::hz() const {
  const auto fitted = grid();
  return fitted ? 1e6 / fitted->period_us : 0.0;
}

std::int64_t VblankClock::output_latency_us() const {
  if (latencies_.empty()) return 0;
  std::vector<std::int64_t> sorted(latencies_.begin(), latencies_.end());
  std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2),
                   sorted.end());
  return sorted[sorted.size() / 2];
}

}  // namespace ajp::core
