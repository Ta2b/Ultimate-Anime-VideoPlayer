#include "app/play/frame_queue.h"

#include <algorithm>
#include <utility>

namespace ajp::app::play {

std::vector<RingFrame> FrameQueue::take_stale(std::uint32_t epoch) {
  std::vector<RingFrame> stale;
  const auto old =
      std::ranges::stable_partition(frames_, [epoch](const RingFrame& f) { return f.epoch >= epoch; });
  stale.assign(old.begin(), old.end());
  frames_.erase(old.begin(), old.end());
  return stale;
}

std::vector<RingFrame> FrameQueue::take_all() {
  std::vector<RingFrame> all(frames_.begin(), frames_.end());
  frames_.clear();
  return all;
}

const RingFrame* FrameQueue::first_of(std::uint32_t epoch) const {
  const auto it = std::ranges::find_if(frames_, [epoch](const RingFrame& f) { return f.epoch == epoch; });
  return it == frames_.end() ? nullptr : &*it;
}

std::optional<RingFrame> FrameQueue::pop_front() {
  if (frames_.empty()) return std::nullopt;
  RingFrame front = frames_.front();
  frames_.erase(frames_.begin());
  return front;
}

}  // namespace ajp::app::play
