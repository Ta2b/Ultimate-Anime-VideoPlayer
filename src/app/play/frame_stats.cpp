#include "app/play/frame_stats.h"

#include <algorithm>
#include <cmath>
#include <format>

#include "core/frame.h"

namespace ajp::app::play {

void Window::add(std::int64_t us) {
  values_[next_] = us;
  next_ = (next_ + 1) % kSize;
  count_ = std::min(count_ + 1, kSize);
}

std::int64_t Window::percentile(int percent) const {
  if (count_ == 0) return 0;
  std::array<std::int64_t, kSize> sorted = values_;
  const auto begin = sorted.begin();
  const auto end = begin + static_cast<std::ptrdiff_t>(count_);
  const auto rank =
      static_cast<std::ptrdiff_t>((count_ - 1) * static_cast<std::size_t>(std::clamp(percent, 0, 100)) / 100);
  std::nth_element(begin, begin + rank, end);
  return *(begin + rank);
}

std::int64_t Window::median() const {
  return percentile(50);
}

double Window::stddev() const {
  if (count_ < 2) return 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < count_; ++i) sum += static_cast<double>(values_[i]);
  const double mean = sum / static_cast<double>(count_);
  double squares = 0.0;
  for (std::size_t i = 0; i < count_; ++i) {
    const double d = static_cast<double>(values_[i]) - mean;
    squares += d * d;
  }
  return std::sqrt(squares / static_cast<double>(count_ - 1));
}

void FrameStats::presented(std::int64_t render_us, std::int64_t present_us, bool fresh) {
  ++presents;
  render.add(render_us);
  if (last_present_us != 0) present_interval.add(present_us - last_present_us);
  last_present_us = present_us;
  if (!fresh) return;
  ++new_frames;
  if (last_new_present_us != 0) frame_interval.add(present_us - last_new_present_us);
  last_new_present_us = present_us;
}

std::string format_ms(std::int64_t us) {
  std::string text = std::format("{:.1f} ms", static_cast<double>(us) / 1000.0);
  std::ranges::replace(text, '.', ',');
  return text;
}

std::string format_position(std::int64_t us) {
  if (us == kNoPts) return "--:--:--.---";
  const std::int64_t ms = std::max<std::int64_t>(us, 0) / 1000;
  return std::format("{:02}:{:02}:{:02}.{:03}", ms / 3'600'000, ms / 60'000 % 60, ms / 1000 % 60, ms % 1000);
}

}  // namespace ajp::app::play
