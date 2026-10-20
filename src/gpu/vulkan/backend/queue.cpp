#include "gpu/vulkan/backend/queue.h"

#include <format>
#include <utility>

namespace ajp::gpu::vk {

std::expected<BackendQueue, Refusal> BackendQueue::create(VkContext& context,
                                                          VkExternalSemaphoreHandleTypeFlags exports) {
  auto timeline = TimelineSemaphore::create(context, 0, exports);
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
  if (auto ok = context_->enqueue(recording, waits_, signals_); !ok) return ok;
  ++value_;
  pending_.clear();
  // Un sémaphore d'ailleurs (slot du ring, image décodée) est attendu hors du lot : il part.
  return extra_signals.empty() ? Done{} : context_->flush();
}

std::optional<BackendQueue::Step> BackendQueue::external_step() {
  if (!pending_.empty()) return std::nullopt;
  return host_step();
}

BackendQueue::Step BackendQueue::host_step() {
  const Step step{.wait = value_, .signal = value_ + 1};
  ++value_;
  return step;
}

Done BackendQueue::wait_elsewhere(std::uint64_t value, std::chrono::nanoseconds timeout) const {
  if (value == 0) return {};
  const VkSemaphore semaphore = timeline_.handle();
  const VkSemaphoreWaitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                 .semaphoreCount = 1,
                                 .pSemaphores = &semaphore,
                                 .pValues = &value};
  const auto nanoseconds = static_cast<std::uint64_t>(timeout.count() < 0 ? 0 : timeout.count());
  return check(context_->fn().vkWaitSemaphores(context_->device(), &wait, nanoseconds),
               std::format("vkWaitSemaphores(valeur {}, autre fil)", value));
}

}  // namespace ajp::gpu::vk
