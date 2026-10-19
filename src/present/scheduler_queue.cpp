#include "present/scheduler_queue.h"

#include <algorithm>

namespace ajp::present {

void FrameQueue::push_back(const RingFrame& frame) {
  frames_[(head_ + count_) % kCapacity] = frame;
  ++count_;
}

RingFrame FrameQueue::pop_front() {
  const RingFrame frame = frames_[head_];
  head_ = (head_ + 1) % kCapacity;
  --count_;
  return frame;
}

void ReleaseList::add(const RingFrame& frame, bool displayed) {
  items_[count_++] = FrameRelease{.free = release_of(frame), .displayed = displayed};
}

void ReleaseList::forget_delivered() {
  const auto first = items_.begin();
  std::move(first + static_cast<std::ptrdiff_t>(delivered_), first + static_cast<std::ptrdiff_t>(count_),
            first);
  count_ -= delivered_;
  delivered_ = 0;
}

std::span<const FrameRelease> ReleaseList::deliver() {
  delivered_ = count_;
  return {items_.data(), count_};
}

}  // namespace ajp::present
