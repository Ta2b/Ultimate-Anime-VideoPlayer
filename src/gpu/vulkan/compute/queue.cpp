// Soumission sur la file compute (par lots), attentes hôte et command buffers jetables.
#include <chrono>
#include <cstddef>
#include <format>
#include <optional>
#include <span>
#include <vector>

#include <spdlog/logger.h>

#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/recording.h"

namespace ajp::gpu::vk {
namespace {

// Au-delà, le lot part sans attendre le prochain flush : borne la latence et la mémoire.
constexpr std::size_t kMaxBatch = 64;

// Rend le command buffer jetable à son pool quand la soumission est finie ou n'a pas eu lieu.
class OnceBuffer {
 public:
  OnceBuffer(const DeviceFunctions& fn, VkDevice device, VkCommandPool pool)
      : fn_(fn), device_(device), pool_(pool) {}
  OnceBuffer(const OnceBuffer&) = delete;
  OnceBuffer& operator=(const OnceBuffer&) = delete;
  OnceBuffer(OnceBuffer&&) = delete;
  OnceBuffer& operator=(OnceBuffer&&) = delete;
  ~OnceBuffer() {
    if (buffer != VK_NULL_HANDLE && !pending) fn_.vkFreeCommandBuffers(device_, pool_, 1, &buffer);
  }

  VkCommandBuffer buffer = VK_NULL_HANDLE;
  bool pending = false;  // soumis et pas encore attendu : le libérer serait invalide

 private:
  const DeviceFunctions& fn_;
  VkDevice device_;
  VkCommandPool pool_;
};

}  // namespace

Done VkContext::queue_submit(std::span<const VkSubmitInfo> infos) {
  if (lock_queue_) lock_queue_();
  const VkResult result = device_fn_.vkQueueSubmit(queue_, static_cast<std::uint32_t>(infos.size()),
                                                   infos.data(), VK_NULL_HANDLE);
  if (unlock_queue_) unlock_queue_();
  ++queue_submits_;
  if (result == VK_SUCCESS && !infos.empty() && infos.back().signalSemaphoreCount > 0) {
    const VkSubmitInfo& info = infos.back();
    const auto* timeline = static_cast<const VkTimelineSemaphoreSubmitInfo*>(info.pNext);
    last_submitted_ = {.semaphore = info.pSignalSemaphores[0], .value = timeline->pSignalSemaphoreValues[0]};
  }
  return check(result, "vkQueueSubmit");
}

bool VkContext::settle(std::optional<TimelinePoint> point) {
  // Le lot part d'abord, même sous un GPU bloqué : il ne garde pas un enregistrement détruit, et un
  // travail encore en attente lirait la ressource après sa fin de vie.
  if (auto flushed = flush(); !flushed) log().error("fin de vie : {}", flushed.error().detail);
  if (stalled_) return false;
  const TimelinePoint target = point.value_or(last_submitted_);
  if (target.semaphore == VK_NULL_HANDLE || target.value == 0) return true;
  const VkSemaphoreWaitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                 .semaphoreCount = 1,
                                 .pSemaphores = &target.semaphore,
                                 .pValues = &target.value};
  const auto timeout = std::chrono::duration_cast<std::chrono::nanoseconds>(kSettleTimeout);
  const VkResult result =
      device_fn_.vkWaitSemaphores(device_, &wait, static_cast<std::uint64_t>(timeout.count()));
  if (result == VK_SUCCESS) return true;
  stalled_ = true;
  log().error(
      "fin de vie : le GPU n'a pas atteint la valeur {} en {} s ({}) ; ses ressources restent en place",
      target.value, kSettleTimeout.count(), result == VK_TIMEOUT ? "délai dépassé" : "appareil perdu");
  return false;
}

Done VkContext::submit(Recording& recording, std::span<const TimelinePoint> waits,
                       std::span<const TimelinePoint> signals) {
  if (auto ok = enqueue(recording, waits, signals); !ok) return ok;
  return flush();
}

Done VkContext::enqueue(Recording& recording, std::span<const TimelinePoint> waits,
                        std::span<const TimelinePoint> signals) {
  if (&recording.context_ != this) {
    return refuse(RefusalCode::InputMismatch, "enregistrement d'un autre contexte Vulkan");
  }
  if (!recording.finished_) return refuse(RefusalCode::InputMismatch, "enregistrement non terminé");
  for (const TimelinePoint& point : waits) {
    if (point.semaphore == VK_NULL_HANDLE)
      return refuse(RefusalCode::InputMismatch, "attente sur un sémaphore nul");
  }
  for (const TimelinePoint& point : signals) {
    if (point.semaphore == VK_NULL_HANDLE)
      return refuse(RefusalCode::InputMismatch, "signal d'un sémaphore nul");
  }
  if (batched_ == kMaxBatch) {
    if (auto ok = flush(); !ok) return ok;
  }
  if (batched_ == batch_.size()) batch_.emplace_back();
  Batched& entry = batch_[batched_];
  entry.recording = &recording;
  entry.wait_semaphores.clear();
  entry.wait_values.clear();
  // L'exécution précédente du même enregistrement, attendue côté GPU (dans le même lot au
  // besoin : un sémaphore timeline peut attendre une valeur signalée plus tôt dans l'appel).
  if (recording.submitted_ > 0) {
    entry.wait_semaphores.push_back(recording.runs_.handle());
    entry.wait_values.push_back(recording.submitted_);
  }
  for (const TimelinePoint& point : waits) {
    entry.wait_semaphores.push_back(point.semaphore);
    entry.wait_values.push_back(point.value);
  }
  entry.stages.assign(entry.wait_semaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  entry.signal_semaphores.assign(1, recording.runs_.handle());
  entry.signal_values.assign(1, recording.submitted_ + 1);
  for (const TimelinePoint& point : signals) {
    entry.signal_semaphores.push_back(point.semaphore);
    entry.signal_values.push_back(point.value);
  }
  ++batched_;
  ++recording.submitted_;
  return {};
}

Done VkContext::flush() {
  if (batched_ == 0) return {};
  batch_timelines_.resize(batched_);
  batch_infos_.resize(batched_);
  batch_buffers_.resize(batched_);
  for (std::size_t k = 0; k < batched_; ++k) {
    const Batched& entry = batch_[k];
    batch_buffers_[k] = entry.recording->command_buffer_;
    batch_timelines_[k] = {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = static_cast<std::uint32_t>(entry.wait_values.size()),
        .pWaitSemaphoreValues = entry.wait_values.data(),
        .signalSemaphoreValueCount = static_cast<std::uint32_t>(entry.signal_values.size()),
        .pSignalSemaphoreValues = entry.signal_values.data()};
    batch_infos_[k] = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .pNext = &batch_timelines_[k],
                       .waitSemaphoreCount = static_cast<std::uint32_t>(entry.wait_semaphores.size()),
                       .pWaitSemaphores = entry.wait_semaphores.data(),
                       .pWaitDstStageMask = entry.stages.data(),
                       .commandBufferCount = 1,
                       .pCommandBuffers = &batch_buffers_[k],
                       .signalSemaphoreCount = static_cast<std::uint32_t>(entry.signal_semaphores.size()),
                       .pSignalSemaphores = entry.signal_semaphores.data()};
  }
  const std::size_t count = batched_;
  batched_ = 0;
  auto ok = queue_submit(std::span(batch_infos_).first(count));
  if (!ok) {
    // Rien n'est parti : les enregistrements n'attendront pas des exécutions jamais soumises.
    for (std::size_t k = 0; k < count; ++k) --batch_[k].recording->submitted_;
  }
  return ok;
}

Done VkContext::wait_on_host(const TimelinePoint& point, std::chrono::nanoseconds timeout) {
  if (point.semaphore == VK_NULL_HANDLE)
    return refuse(RefusalCode::InputMismatch, "attente d'un sémaphore nul");
  if (auto ok = flush(); !ok) return ok;
  ++host_waits_;
  const VkSemaphoreWaitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                 .semaphoreCount = 1,
                                 .pSemaphores = &point.semaphore,
                                 .pValues = &point.value};
  const auto nanoseconds = static_cast<std::uint64_t>(timeout.count() < 0 ? 0 : timeout.count());
  return check(device_fn_.vkWaitSemaphores(device_, &wait, nanoseconds),
               std::format("vkWaitSemaphores(valeur {})", point.value));
}

Done VkContext::run_once(const std::function<void(VkCommandBuffer)>& record) {
  if (auto ok = flush(); !ok) return ok;  // les copies voient tout ce qui a été soumis avant
  OnceBuffer once(device_fn_, device_, once_pool_);
  const VkCommandBufferAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                             .commandPool = once_pool_,
                                             .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                             .commandBufferCount = 1};
  if (auto ok = check(device_fn_.vkAllocateCommandBuffers(device_, &allocate, &once.buffer),
                      "vkAllocateCommandBuffers");
      !ok) {
    return ok;
  }
  const VkCommandBufferBeginInfo begin{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  if (auto ok = check(device_fn_.vkBeginCommandBuffer(once.buffer, &begin), "vkBeginCommandBuffer"); !ok)
    return ok;
  record(once.buffer);
  if (auto ok = check(device_fn_.vkEndCommandBuffer(once.buffer), "vkEndCommandBuffer"); !ok) return ok;

  const std::uint64_t value = ++once_count_;
  const VkTimelineSemaphoreSubmitInfo timeline{.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                               .signalSemaphoreValueCount = 1,
                                               .pSignalSemaphoreValues = &value};
  const VkSubmitInfo info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .pNext = &timeline,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &once.buffer,
                          .signalSemaphoreCount = 1,
                          .pSignalSemaphores = &once_done_};
  if (auto ok = queue_submit(std::span(&info, 1)); !ok) return ok;
  once.pending = true;
  if (auto ok = wait_on_host({.semaphore = once_done_, .value = value}); !ok) return ok;
  once.pending = false;
  return {};
}

}  // namespace ajp::gpu::vk
