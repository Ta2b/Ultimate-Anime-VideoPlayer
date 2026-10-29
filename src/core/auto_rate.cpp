#include "core/auto_rate.h"

#include <algorithm>
#include <bit>

namespace ajp::core {

int auto_ceiling(double display_hz, double source_fps, int granularity) {
  granularity = std::clamp(granularity, 1, 2);
  // Hors de [5, 240] i/s : une cadence fausse (VFR, base de temps) ; l'interpolation reste à ×1.
  if (display_hz <= 0.0 || source_fps < 5.0 || source_fps > 240.0) return granularity;
  int best = granularity;
  for (int step = 2 * granularity; step <= kAutoMaxStep * granularity; ++step) {
    if (source_fps * step / granularity <= display_hz * 1.01) best = step;
  }
  return best;
}

AutoRate::AutoRate(int ceiling, int granularity)
    : granularity_(std::clamp(granularity, 1, 2)),
      ceiling_(std::clamp(ceiling, granularity_, kAutoMaxStep * granularity_)),
      step_(std::min(2 * granularity_, ceiling_)) {}

void AutoRate::set_ceiling(int ceiling, int granularity) {
  granularity = std::clamp(granularity, 1, 2);
  if (granularity != granularity_) {
    // De demi-paliers à paliers entiers, l'arrondi est vers le bas : jamais un palier plus haut que celui
    // qui tenait. Les échecs comptés ne se transportent pas d'une unité à l'autre.
    const auto convert = [&](int value) {
      return granularity > granularity_ ? value * 2 : std::max(value / 2, value > 0 ? 1 : 0);
    };
    step_ = std::max(convert(step_), granularity);
    cap_ = convert(cap_);
    fails_.fill(0);
    granularity_ = granularity;
    clean_ = 0;
    probing_ = false;
    grace_ = kGracePairs;
  }
  set_ceiling(ceiling);
}

void AutoRate::set_ceiling(int ceiling) {
  ceiling_ = std::clamp(ceiling, granularity_, kAutoMaxStep * granularity_);
  if (cap_ > ceiling_) cap_ = 0;
  if (step_ > ceiling_) {
    step_ = ceiling_;
    clean_ = 0;
    grace_ = kGracePairs;
  }
}

void AutoRate::descend(bool strong) {
  if (step_ > granularity_) {
    if (probing_) {
      // Une tentative qui échoue : on la re-tentera plus tard ; la deuxième fait du palier d'en dessous
      // le plafond appris.
      fails_[static_cast<std::size_t>(step_)] += strong ? kFailsToCap : 1;
      if (fails_[static_cast<std::size_t>(step_)] >= kFailsToCap) {
        cap_ = step_ - 1;
        fails_[static_cast<std::size_t>(step_)] = 0;
      }
      backoff_ = std::min(backoff_ + 1, kMaxBackoff);
    }
    --step_;
    grace_ = kGracePairs;
  }
  probing_ = false;
  clean_ = 0;
  lates_ = 0;
}

int AutoRate::overloaded() {
  if (grace_ == 0) descend(true);
  return step_;
}

int AutoRate::pair(bool late) {
  if (grace_ > 0) {
    --grace_;
    return step_;
  }
  lates_ = (lates_ << 1U) | (late ? 1U : 0U);  // kLateWindow = 64 : tout le mot
  if (late) {
    clean_ = 0;
    if (std::popcount(lates_) < kLateTolerance) return step_;  // un retard isolé : le palier reste
    descend();
    return step_;
  }
  ++clean_;
  const bool at_cap = cap_ > 0 && step_ >= cap_;
  const std::uint32_t needed = at_cap ? kCapPairs : kRisePairs << backoff_;
  if (probing_ && clean_ >= kRisePairs * 2) {  // le palier tenté tient : l'hystérésis se relâche
    probing_ = false;
    backoff_ = std::max(backoff_ - 1, 0);
    fails_[static_cast<std::size_t>(step_)] = 0;
    if (cap_ > 0 && step_ > cap_) cap_ = 0;  // le palier au-dessus du plafond tient : il n'y en a plus
  }
  if (clean_ >= needed && step_ < ceiling_) {
    ++step_;
    probing_ = true;
    clean_ = 0;
  }
  return step_;
}

}  // namespace ajp::core
