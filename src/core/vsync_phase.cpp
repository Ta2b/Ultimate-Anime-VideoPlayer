#include "core/vsync_phase.h"

#include <cmath>

namespace ajp::core {

void VsyncPhase::set_period(std::int64_t period_us) {
  if (period_us == period_us_) return;
  period_us_ = period_us;
  known_ = false;
  phase_us_ = 0.0;
}

void VsyncPhase::note(std::int64_t host_us) {
  if (period_us_ <= 0) return;
  const auto period = static_cast<double>(period_us_);
  const double seen = std::fmod(static_cast<double>(host_us), period);
  if (!known_) {
    phase_us_ = seen;
    known_ = true;
    return;
  }
  double delta = seen - phase_us_;
  delta -= period * std::round(delta / period);  // l'écart le plus court, dans ±période/2
  phase_us_ = std::fmod(phase_us_ + kSmoothing * delta + period, period);
}

std::int64_t VsyncPhase::near(std::int64_t host_us) const {
  if (!known_ || period_us_ <= 0) return host_us;
  const auto period = static_cast<double>(period_us_);
  const double k = std::round((static_cast<double>(host_us) - phase_us_) / period);
  return static_cast<std::int64_t>(std::llround(phase_us_ + k * period));
}

}  // namespace ajp::core
