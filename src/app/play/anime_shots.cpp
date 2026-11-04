#include "app/play/anime_shots.h"

#include <algorithm>
#include <cmath>
#include <iterator>

#include "render/curve.h"

namespace ajp::app::play {

double anime_exposure(double shot_apl_nits, double series_apl_nits) {
  if (!(shot_apl_nits > 0.0) || !(series_apl_nits > 0.0)) return 0.0;
  const double e = -render::kCurveSlope * std::log2(shot_apl_nits / series_apl_nits);
  return std::clamp(e, -render::kCurveMaxE, render::kCurveMaxE);
}

double series_apl_of(const std::vector<media::Shot>& shots) {
  std::vector<double> apl;
  apl.reserve(shots.size());
  for (const media::Shot& s : shots)
    if (s.meta.apl_mean_nits > 0.0) apl.push_back(s.meta.apl_mean_nits);
  if (apl.empty()) return 0.0;
  const auto mid = apl.begin() + static_cast<std::ptrdiff_t>(apl.size() / 2);
  std::nth_element(apl.begin(), mid, apl.end());
  return *mid;
}

void AnimeGlide::aim(const AnimeShotLook& target, std::int64_t now_us, bool cut) {
  if (!known_ || !cut) {
    // Hors coupe : le plan est le même, sa valeur suit aussitôt (le glissé en cours finit vers elle).
    if (known_ && gliding(now_us)) {
      to_ = target;
      return;
    }
    from_ = target;
    to_ = target;
    start_us_ = now_us - kAnimeGlideUs;
    known_ = true;
    return;
  }
  from_ = at(now_us);
  to_ = target;
  start_us_ = now_us;
}

AnimeShotLook AnimeGlide::at(std::int64_t now_us) const {
  if (!gliding(now_us)) return to_;
  const double x =
      std::clamp(static_cast<double>(now_us - start_us_) / static_cast<double>(kAnimeGlideUs), 0.0, 1.0);
  const double t = x * x * (3.0 - 2.0 * x);
  return AnimeShotLook{.e = std::lerp(from_.e, to_.e, t),
                       .light = {.apl_nits = std::lerp(from_.light.apl_nits, to_.light.apl_nits, t),
                                 .lit = std::lerp(from_.light.lit, to_.light.lit, t)}};
}

bool AnimeGlide::gliding(std::int64_t now_us) const {
  return known_ && now_us - start_us_ < kAnimeGlideUs;
}

}  // namespace ajp::app::play
