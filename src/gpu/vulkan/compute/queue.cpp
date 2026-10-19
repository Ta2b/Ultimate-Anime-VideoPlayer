// Soumission sur la file compute, attentes hôte et command buffers jetables.
#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <span>

#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/recording.h"

namespace ajp::gpu::vk {
namespace {

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

// Sémaphores et valeurs d'un côté d'une soumission, sur la pile : soumettre chaque image
// n'alloue rien sur l'hôte. Une place de plus que kMaxSubmitPoints pour le sémaphore runs.
class PointList {
 public:
  void push(VkSemaphore semaphore, std::uint64_t value) {
    semaphores_[count_] = semaphore;
    values_[count_] = value;
    ++count_;
  }
  std::span<const VkSemaphore> semaphores() const { return std::span(semaphores_).first(count_); }
  const std::uint64_t* values() const { return values_.data(); }
  std::uint32_t size() const { return static_cast<std::uint32_t>(count_); }

 private:
  static constexpr std::size_t kCapacity = VkContext::kMaxSubmitPoints + 1;
  std::array<VkSemaphore, kCapacity> semaphores_{};
  std::array<std::uint64_t, kCapacity> values_{};
  std::size_t count_ = 0;
};

// Toutes les attentes d'une soumission bloquent tout le travail qui suit.
constexpr std::array<VkPipelineStageFlags, VkContext::kMaxSubmitPoints + 1> kWaitStages = [] {
  std::array<VkPipelineStageFlags, VkContext::kMaxSubmitPoints + 1> stages{};
  stages.fill(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  return stages;
}();

}  // namespace

Done VkContext::queue_submit(const VkSubmitInfo& info) {
  if (lock_queue_) lock_queue_();
  const VkResult result = device_fn_.vkQueueSubmit(queue_, 1, &info, VK_NULL_HANDLE);
  if (unlock_queue_) unlock_queue_();
  return check(result, "vkQueueSubmit");
}

// Un signal externe valide : sémaphore non nul, absent du lot (runs compris), valeur au-delà de
// la valeur courante. Une valeur qu'une autre soumission en attente signale reste à l'appelant.
Done VkContext::check_signal(const TimelinePoint& point, std::span<const VkSemaphore> batch) const {
  if (point.semaphore == VK_NULL_HANDLE)
    return refuse(RefusalCode::InputMismatch, "signal d'un sémaphore nul");
  if (std::ranges::find(batch, point.semaphore) != batch.end()) {
    return refuse(
        RefusalCode::InputMismatch,
        "sémaphore signalé deux fois par la même soumission (ou sémaphore runs de l'enregistrement)");
  }
  std::uint64_t current = 0;
  if (auto ok = check(device_fn_.vkGetSemaphoreCounterValue(device_, point.semaphore, &current),
                      "vkGetSemaphoreCounterValue");
      !ok) {
    return ok;
  }
  if (point.value <= current) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("signal de la valeur {}, le sémaphore est déjà à {}", point.value, current));
  }
  return {};
}

Done VkContext::submit(Recording& recording, std::span<const TimelinePoint> waits,
                       std::span<const TimelinePoint> signals) {
  if (&recording.context_ != this) {
    return refuse(RefusalCode::InputMismatch, "enregistrement d'un autre contexte Vulkan");
  }
  if (!recording.finished_) return refuse(RefusalCode::InputMismatch, "enregistrement non terminé");
  if (waits.size() > kMaxSubmitPoints || signals.size() > kMaxSubmitPoints) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("{} attentes et {} signaux, {} au plus de chaque",
                                                            waits.size(), signals.size(), kMaxSubmitPoints));
  }
  PointList wait_points;
  // L'exécution précédente du même enregistrement, attendue côté GPU.
  if (recording.submitted_ > 0) wait_points.push(recording.runs_.handle(), recording.submitted_);
  PointList signal_points;
  signal_points.push(recording.runs_.handle(), recording.submitted_ + 1);
  for (const TimelinePoint& point : waits) {
    if (point.semaphore == VK_NULL_HANDLE)
      return refuse(RefusalCode::InputMismatch, "attente sur un sémaphore nul");
    // L'enchaînement des exécutions est déjà attendu ; une autre valeur bloquerait la file.
    if (point.semaphore == recording.runs_.handle()) {
      return refuse(RefusalCode::InputMismatch, "attente sur le sémaphore runs de l'enregistrement");
    }
    wait_points.push(point.semaphore, point.value);
  }
  for (const TimelinePoint& point : signals) {
    if (auto ok = check_signal(point, signal_points.semaphores()); !ok) return ok;
    signal_points.push(point.semaphore, point.value);
  }
  const VkTimelineSemaphoreSubmitInfo timeline{.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                               .waitSemaphoreValueCount = wait_points.size(),
                                               .pWaitSemaphoreValues = wait_points.values(),
                                               .signalSemaphoreValueCount = signal_points.size(),
                                               .pSignalSemaphoreValues = signal_points.values()};
  const VkCommandBuffer cb = recording.command_buffer_;
  const VkSubmitInfo info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .pNext = &timeline,
                          .waitSemaphoreCount = wait_points.size(),
                          .pWaitSemaphores = wait_points.semaphores().data(),
                          .pWaitDstStageMask = kWaitStages.data(),
                          .commandBufferCount = 1,
                          .pCommandBuffers = &cb,
                          .signalSemaphoreCount = signal_points.size(),
                          .pSignalSemaphores = signal_points.semaphores().data()};
  if (auto ok = queue_submit(info); !ok) return ok;
  ++recording.submitted_;
  return {};
}

Done VkContext::wait_on_host(const TimelinePoint& point, std::chrono::nanoseconds timeout) {
  if (point.semaphore == VK_NULL_HANDLE)
    return refuse(RefusalCode::InputMismatch, "attente d'un sémaphore nul");
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

  // once_count_ ne compte que les soumissions acceptées : la destruction attend la dernière.
  const std::uint64_t value = once_count_ + 1;
  const VkTimelineSemaphoreSubmitInfo timeline{.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                               .signalSemaphoreValueCount = 1,
                                               .pSignalSemaphoreValues = &value};
  const VkSubmitInfo info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .pNext = &timeline,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &once.buffer,
                          .signalSemaphoreCount = 1,
                          .pSignalSemaphores = &once_done_};
  if (auto ok = queue_submit(info); !ok) return ok;
  once_count_ = value;
  once.pending = true;
  if (auto ok = wait_on_host({.semaphore = once_done_, .value = value}); !ok) return ok;
  once.pending = false;
  return {};
}

}  // namespace ajp::gpu::vk
