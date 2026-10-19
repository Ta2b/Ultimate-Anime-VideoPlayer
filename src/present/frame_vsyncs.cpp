#include "present/frame_vsyncs.h"

#include <algorithm>

namespace ajp::present {

VsyncTick FrameVsyncs::next(const VsyncEstimator& estimator, std::int64_t acquired_us, int latency) {
  const auto estimate = estimator.estimate();
  const auto now = estimator.index_at(acquired_us);
  if (!estimate || !now) {
    have_raw_ = false;
    last_index_ += 1;
    return {.host_us = acquired_us, .index = last_index_};
  }
  const std::int64_t raw = *now + latency;
  if (!have_raw_ || raw < last_raw_) {
    // Première estimation, ou estimation recommencée (ses rangs repartent de zéro) : nos rangs
    // continuent ceux déjà donnés.
    offset_ = last_index_ + 1 - raw;
  }
  last_raw_ = raw;
  have_raw_ = true;
  last_index_ = std::max(raw + offset_, last_index_ + 1);
  return {
      .host_us = estimator.host_at(last_index_ - offset_).value_or(acquired_us),
      .index = last_index_,
      .period_us = estimate->period_us,
  };
}

}  // namespace ajp::present
