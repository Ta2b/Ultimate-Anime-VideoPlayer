#include "audio/drift.h"

#include <algorithm>
#include <cmath>

#include "audio/pcm.h"

namespace ajp::audio {

double DriftRegulator::update(std::int64_t error_us, std::int64_t elapsed_us) {
  const double error_s = static_cast<double>(error_us) * 1e-6;
  const double dt_s = static_cast<double>(std::max<std::int64_t>(elapsed_us, 0)) * 1e-6;
  if (!primed_) {
    filtered_s_ = error_s;
    primed_ = true;
  } else {
    filtered_s_ += (error_s - filtered_s_) * std::min(1.0, dt_s / gains_.smoothing_s);
  }

  // L'intégrale n'avance que si la correction qui en résulte reste dans la borne.
  constexpr double kMax = limits::kMaxDriftCorrection;
  const double integral = integral_ + filtered_s_ * dt_s;
  if (std::abs(gains_.proportional * filtered_s_ + gains_.integral * integral) < kMax) integral_ = integral;

  // L'audio en avance ralentit : rapport inférieur à 1.
  const double correction = gains_.proportional * filtered_s_ + gains_.integral * integral_;
  ratio_ = 1.0 - std::clamp(correction, -kMax, kMax);
  return ratio_;
}

void DriftRegulator::reset() {
  filtered_s_ = 0.0;
  integral_ = 0.0;
  ratio_ = 1.0;
  primed_ = false;
}

}  // namespace ajp::audio
