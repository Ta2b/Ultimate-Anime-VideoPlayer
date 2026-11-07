// Détection des flashs selon la recommandation UIT-R BT.1702-2 (P3.51) : un flash est une paire de variations
// opposées de luminance d'au moins 20 cd/m², l'image la plus sombre sous 160 cd/m². Plus de trois flashs dans
// une seconde rendent la séquence dangereuse pour les personnes photosensibles. Le lecteur mesure la
// luminance moyenne de chaque image source (FrameStats::apl_nits) : une variation de 20 cd/m² de la moyenne
// vaut celle de 80 cd/m² sur le quart de l'écran, la surface que la recommandation retient.
#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>

namespace ajp::media {

inline constexpr double kFlashStepNits = 20.0;
inline constexpr double kFlashDarkNits = 160.0;
inline constexpr int kFlashesPerSecond = 3;  // au-delà, la séquence est dangereuse
// Le plafond tient une seconde après le dernier instant dangereux.
inline constexpr std::int64_t kFlashHoldUs = 1'000'000;

class FlashGuard {
 public:
  // L'image source à pts_us, de luminance moyenne apl_nits. Vrai tant que le plafond doit tenir.
  bool push(std::int64_t pts_us, double apl_nits) {
    if (pts_us <= last_pts_) reset();
    last_pts_ = pts_us;
    if (!started_) {
      started_ = true;
      low_ = high_ = apl_nits;
      return false;
    }
    // Suit l'extrême du sens courant ; une variation opposée d'au moins kFlashStepNits fait une transition.
    if (rising_) {
      high_ = std::max(high_, apl_nits);
      if (high_ - apl_nits >= kFlashStepNits && apl_nits < kFlashDarkNits) turn(pts_us, apl_nits);
    } else {
      low_ = std::min(low_, apl_nits);
      if (apl_nits - low_ >= kFlashStepNits && low_ < kFlashDarkNits) turn(pts_us, apl_nits);
    }
    while (!turns_.empty() && turns_.front() <= pts_us - 1'000'000) turns_.pop_front();
    // Deux transitions font un flash : plus de trois flashs, au moins sept transitions dans la seconde.
    if (std::ssize(turns_) > 2 * kFlashesPerSecond) limited_until_ = pts_us + kFlashHoldUs;
    return pts_us < limited_until_;
  }

  void reset() { *this = FlashGuard{}; }

 private:
  void turn(std::int64_t pts_us, double apl_nits) {
    turns_.push_back(pts_us);
    rising_ = !rising_;
    low_ = high_ = apl_nits;
  }

  std::deque<std::int64_t> turns_;
  std::int64_t last_pts_ = INT64_MIN;
  std::int64_t limited_until_ = INT64_MIN;
  double low_ = 0.0, high_ = 0.0;
  bool rising_ = true;
  bool started_ = false;
};

}  // namespace ajp::media
