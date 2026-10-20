// Sémaphores timeline : chaque soumission attend et signale des valeurs, côté GPU. L'hôte ne
// lit la valeur courante que pour constater un état ; il n'attend que dans les tests et les
// envois ou lectures hôte (VkContext::wait_on_host).
#pragma once

#include <cstdint>
#include <expected>

#include "core/types.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/functions.h"
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

// Un VkSemaphore de type timeline, de valeur initiale initial. Le propriétaire le détruit.
std::expected<VkSemaphore, Refusal> create_timeline(const DeviceFunctions& fn, VkDevice device,
                                                    std::uint64_t initial);

class TimelineSemaphore {
 public:
  static std::expected<TimelineSemaphore, Refusal> create(const VkContext& context,
                                                          std::uint64_t initial = 0);

  TimelineSemaphore(TimelineSemaphore&& other) noexcept;
  TimelineSemaphore& operator=(TimelineSemaphore&& other) noexcept;
  TimelineSemaphore(const TimelineSemaphore&) = delete;
  TimelineSemaphore& operator=(const TimelineSemaphore&) = delete;
  ~TimelineSemaphore();

  VkSemaphore handle() const { return semaphore_; }
  TimelinePoint at(std::uint64_t value) const { return {.semaphore = semaphore_, .value = value}; }

  // Valeur atteinte à cet instant (vkGetSemaphoreCounterValue) : un constat, pas une attente.
  std::expected<std::uint64_t, Refusal> value() const;

  // Signal depuis l'hôte (vkSignalSemaphore) : value doit dépasser la valeur courante.
  Done signal(std::uint64_t value);

 private:
  TimelineSemaphore(const DeviceFunctions* fn, VkDevice device, VkSemaphore semaphore)
      : fn_(fn), device_(device), semaphore_(semaphore) {}
  void destroy();

  const DeviceFunctions* fn_ = nullptr;
  VkDevice device_ = VK_NULL_HANDLE;
  VkSemaphore semaphore_ = VK_NULL_HANDLE;
};

}  // namespace ajp::gpu::vk
