#include "audio/timeline.h"

#include <algorithm>
#include <iterator>

extern "C" {
#include <libavutil/common.h>
}

#include "audio/pcm.h"

namespace kirari::audio {

void Timeline::append(std::int64_t start, std::int64_t frames, std::optional<std::int64_t> pts_us) {
  if (frames <= 0) return;
  segments_.push_back({.start = start, .frames = frames, .pts_us = pts_us});
}

std::optional<std::int64_t> Timeline::pts_at(std::int64_t index) const {
  if (segments_.empty() || index < segments_.front().start) return std::nullopt;
  // Le dernier segment qui commence avant index ; au-delà de sa fin, on reste à sa fin.
  const auto after = std::ranges::upper_bound(segments_, index, {}, &Segment::start);
  const Segment& segment = *std::prev(after);
  if (!segment.pts_us) return std::nullopt;
  const std::int64_t offset = std::min(av_sat_sub64(index, segment.start), segment.frames);
  return av_sat_add64(*segment.pts_us, frames_to_us(offset, sample_rate_));
}

void Timeline::forget_before(std::int64_t index) {
  while (segments_.size() > 1 && segments_[1].start <= index) segments_.pop_front();
}

}  // namespace kirari::audio
