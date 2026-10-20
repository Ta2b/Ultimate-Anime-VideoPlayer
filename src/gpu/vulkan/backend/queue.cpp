#include "gpu/vulkan/backend/queue.h"

#include <utility>

namespace ajp::gpu::vk {

std::expected<BackendQueue, Refusal> BackendQueue::create(VkContext& context) {
  auto timeline = TimelineSemaphore::create(context, 0);
  if (!timeline) return std::unexpected(timeline.error());
  return BackendQueue(context, std::move(*timeline));
}

Done BackendQueue::submit(Recording& recording, std::span<const TimelinePoint> extra_waits,
                          std::span<const TimelinePoint> extra_signals) {
  waits_.assign(pending_.begin(), pending_.end());
  if (value_ > 0) waits_.push_back(timeline_.at(value_));
  waits_.insert(waits_.end(), extra_waits.begin(), extra_waits.end());
  signals_.assign(1, timeline_.at(value_ + 1));
  signals_.insert(signals_.end(), extra_signals.begin(), extra_signals.end());
  if (auto ok = context_->submit(recording, waits_, signals_); !ok) return ok;
  ++value_;
  pending_.clear();
  return {};
}

}  // namespace ajp::gpu::vk
