#include "export/progress.h"

#include <algorithm>
#include <cmath>

namespace ajp::exporter {

ExportProgress ProgressMeter::frame_done(ExportClock::time_point now) {
  if (!start_) start_ = now;
  ++done_;
  return at(now);
}

ExportProgress ProgressMeter::at(ExportClock::time_point now) const {
  ExportProgress progress{.frames_done = done_, .frames_total = total_};
  if (!start_) return progress;
  const auto elapsed = std::max(now - *start_, ExportClock::duration::zero());
  progress.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
  const double seconds = std::chrono::duration<double>(elapsed).count();
  if (seconds <= 0.0 || done_ == 0) return progress;
  progress.frames_per_second = static_cast<double>(done_) / seconds;
  if (total_) {
    const auto left = std::max<std::int64_t>(*total_ - done_, 0);
    const double left_seconds = static_cast<double>(left) / progress.frames_per_second;
    progress.remaining = std::chrono::milliseconds(std::llround(left_seconds * 1000.0));
  }
  return progress;
}

}  // namespace ajp::exporter
