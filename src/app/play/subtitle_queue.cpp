#include "app/play/subtitle_queue.h"

#include <algorithm>
#include <utility>

#include "core/frame.h"

namespace ajp::app::play {

void SubtitleQueue::push(subs::SubtitleFrame&& frame, std::int64_t shown_pts) {
  // subs répond dans l'ordre des demandes ; une réponse en retard se range quand même à sa place.
  const auto at =
      std::upper_bound(frames_.begin(), frames_.end(), frame.pts_us,
                       [](std::int64_t pts, const subs::SubtitleFrame& f) { return pts < f.pts_us; });
  frames_.insert(at, std::move(frame));
  if (shown_pts != kNoPts) forget_before(shown_pts);
  while (frames_.size() > capacity_) frames_.pop_back();
}

const subs::SubtitleFrame* SubtitleQueue::for_pts(std::int64_t pts) {
  if (pts == kNoPts) return nullptr;
  forget_before(pts);
  if (frames_.empty() || frames_.front().pts_us > pts) return nullptr;
  return &frames_.front();
}

void SubtitleQueue::forget_before(std::int64_t pts) {
  // Garde la dernière réponse de pts <= pts (celle de l'image) et tout ce qui la suit.
  while (frames_.size() > 1 && frames_[1].pts_us <= pts) frames_.pop_front();
}

}  // namespace ajp::app::play
