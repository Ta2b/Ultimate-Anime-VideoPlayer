#include "core/vblank_clock.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ajp::core {

void VblankClock::reset() {
  times_.clear();
  latencies_.clear();
  fit_.reset();
  rejects_ = 0;
  have_last_ = false;
}

void VblankClock::set_nominal_period(double period_us) {
  if (!(period_us > 0.0)) period_us = 0.0;
  const bool moved = nominal_us_ > 0.0 ? std::abs(period_us / nominal_us_ - 1.0) > 0.005 : period_us > 0.0;
  nominal_us_ = period_us;
  if (moved && have_last_) reset();
  refit();  // la tolérance à la période annoncée a pu changer
}

void VblankClock::restart(std::int64_t visible_us) {
  times_.clear();
  rejects_ = 0;
  last_us_ = visible_us;
  last_k_ = 0;
  have_last_ = true;
  times_.push_back({visible_us, 0});
  refit();
}

void VblankClock::push(std::int64_t visible_us, std::int64_t steps) {
  rejects_ = 0;
  last_k_ += steps;
  last_us_ = visible_us;
  times_.push_back({visible_us, last_k_});
  if (times_.size() > kWindow) times_.pop_front();
  refit();
}

void VblankClock::observe(std::int64_t visible_us, std::uint64_t /*id*/, std::int64_t target_us) {
  if (target_us > 0) {
    latencies_.push_back(visible_us - target_us);
    if (latencies_.size() > kWindow) latencies_.pop_front();
  }
  // Sans période annoncée, aucun pas sûr pour numéroter les vblanks : pas de grille (C12-08).
  if (!(nominal_us_ > 0.0)) return;
  if (!have_last_) {
    restart(visible_us);
    return;
  }
  if (visible_us <= last_us_) return;  // hors d'ordre ou doublon : sans effet sur la grille
  if (visible_us - last_us_ > kMaxGapUs) {
    restart(visible_us);
    return;
  }
  // Le pas : la période annoncée, affinée par la régression tant qu'elle en reste proche (164,78 Hz pour 165
  // annoncés ; jamais la cadence du contenu, à des multiples de là).
  const double period = fit_ ? fit_->period_us : nominal_us_;
  const double ratio = static_cast<double>(visible_us - last_us_) / period;
  const std::int64_t steps = std::max<std::int64_t>(1, std::llround(ratio));
  if (std::abs(ratio - static_cast<double>(steps)) > kOffGrid) {
    if (++rejects_ >= kMaxRejects) restart(visible_us);
    return;
  }
  push(visible_us, steps);
}

// La régression sur les dates gardées, refaite à chaque date retenue et non à chaque lecture de grid() (trois
// ou quatre fois par tour de main).
void VblankClock::refit() {
  fit_.reset();
  if (times_.size() < kMinSamples || !(nominal_us_ > 0.0)) return;
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
  if (denom <= 0.0) return;
  const double period = (n * skt - sk * st) / denom;
  if (period <= 0.0) return;
  // Loin de la période annoncée : des numéros faux (dates hors grille), pas une grille.
  if (std::abs(period / nominal_us_ - 1.0) > kNominalTolerance) return;
  const double phase = (st - period * sk) / n;
  fit_ = VblankGrid{.phase_us = static_cast<double>(t0) + phase - period * static_cast<double>(k0),
                    .period_us = period};
}

double VblankClock::hz() const {
  return fit_ ? 1e6 / fit_->period_us : 0.0;
}

std::int64_t VblankClock::output_latency_us() const {
  if (latencies_.empty()) return 0;
  std::vector<std::int64_t> sorted(latencies_.begin(), latencies_.end());
  std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2),
                   sorted.end());
  return sorted[sorted.size() / 2];
}

}  // namespace ajp::core
