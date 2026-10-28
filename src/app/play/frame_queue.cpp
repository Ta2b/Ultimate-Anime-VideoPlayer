#include "app/play/frame_queue.h"

#include <algorithm>
#include <utility>

#include "core/frame.h"

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

std::optional<RingFrame> FrameQueue::pick(std::int64_t media_now_us, std::int64_t lead_us,
                                          std::vector<RingFrame>& late,
                                          const std::function<bool(const RingFrame&)>& ready) {
  std::optional<RingFrame> chosen;
  while (!frames_.empty()) {
    // Lu avant pop_front, qui détruit l'élément (et parfois le bloc de la deque qui le porte).
    const bool untimed = frames_.front().pts_us == kNoPts;
    const bool due = untimed || frames_.front().pts_us <= media_now_us + lead_us;
    if (!due || (ready && !ready(frames_.front()))) break;
    if (chosen) late.push_back(*chosen);
    chosen = frames_.front();
    frames_.pop_front();
    if (untimed) break;  // sans horodatage : montrée seule, à son tour
  }
  return chosen;
}

std::optional<RingFrame> FrameQueue::pop_front() {
  if (frames_.empty()) return std::nullopt;
  RingFrame front = frames_.front();
  frames_.pop_front();
  return front;
}

}  // namespace ajp::app::play
