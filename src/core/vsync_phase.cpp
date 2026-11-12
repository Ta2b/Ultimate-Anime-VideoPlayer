#include "core/vsync_phase.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace ajp::core {

void VsyncPhase::set_period(std::int64_t period_us) {
  if (period_us == period_us_) return;
  period_us_ = period_us;
  cos_ = 0.0;
  sin_ = 0.0;
  weight_ = 0.0;
  notes_ = 0;
}

void VsyncPhase::note(std::int64_t host_us) {
  if (period_us_ <= 0) return;
  const auto period = static_cast<double>(period_us_);
  const double angle = 2.0 * std::numbers::pi * std::fmod(static_cast<double>(host_us), period) / period;
  cos_ += kSmoothing * (std::cos(angle) - cos_);
  sin_ += kSmoothing * (std::sin(angle) - sin_);
  weight_ += kSmoothing * (1.0 - weight_);  // la moyenne part de zéro : son poids la ramène à l'échelle
  notes_ = std::min(notes_ + 1, kMinNotes);
  last_note_us_ = host_us;
}

bool VsyncPhase::known() const {
  if (period_us_ <= 0 || notes_ < kMinNotes || weight_ <= 0.0) return false;
  return std::hypot(cos_, sin_) / weight_ >= kAgreement;
}

double VsyncPhase::phase_us() const {
  const auto period = static_cast<double>(period_us_);
  double turn = std::atan2(sin_, cos_) / (2.0 * std::numbers::pi);  // (−0,5 ; 0,5]
  if (turn < 0.0) turn += 1.0;
  return turn * period;
}

std::int64_t VsyncPhase::nearest(std::int64_t host_us) const {
  if (!known()) return host_us;
  const auto period = static_cast<double>(period_us_);
  const double phase = phase_us();
  const double k = std::round((static_cast<double>(host_us) - phase) / period);
  return static_cast<std::int64_t>(std::llround(phase + k * period));
}

}  // namespace ajp::core
