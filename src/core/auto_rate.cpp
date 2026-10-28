#include "core/auto_rate.h"

#include <algorithm>

namespace ajp::core {

int auto_ceiling(double display_hz, double source_fps) {
  if (display_hz <= 0.0 || source_fps <= 0.0) return 1;
  int best = 1;
  for (int step = 2; step <= kAutoMaxStep; ++step) {
    if (source_fps * step <= display_hz * 1.01) best = step;
  }
  return best;
}

AutoRate::AutoRate(int ceiling)
    : ceiling_(std::clamp(ceiling, 1, kAutoMaxStep)), step_(std::min(2, ceiling_)) {}

void AutoRate::set_ceiling(int ceiling) {
  ceiling_ = std::clamp(ceiling, 1, kAutoMaxStep);
  if (step_ > ceiling_) {
    step_ = ceiling_;
    clean_ = 0;
    grace_ = kGracePairs;
  }
}

int AutoRate::pair(bool late) {
  if (grace_ > 0) {
    --grace_;
    return step_;
  }
  if (late) {
    if (step_ > 1) {
      --step_;
      grace_ = kGracePairs;
      // Un palier tenté qui échoue : on le re-tentera plus tard.
      if (probing_) backoff_ = std::min(backoff_ + 1, kMaxBackoff);
    }
    probing_ = false;
    clean_ = 0;
    return step_;
  }
  ++clean_;
  const std::uint32_t needed = kRisePairs << backoff_;
  if (probing_ && clean_ >= kRisePairs * 2) {  // le palier tenté tient : l'hystérésis se relâche
    probing_ = false;
    backoff_ = std::max(backoff_ - 1, 0);
  }
  if (clean_ >= needed && step_ < ceiling_) {
    ++step_;
    probing_ = true;
    clean_ = 0;
  }
  return step_;
}

}  // namespace ajp::core
