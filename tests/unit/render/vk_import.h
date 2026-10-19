// Côté écrivain, pour les tests : importe un slot exporté (mémoire et sémaphore par fd) dans
// une nouvelle VkImage du même appareil, l'enveloppe en pl_tex, et y écrit comme gpu le ferait.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>

#include <libplacebo/vulkan.h>

#include "core/gpu_device.h"
#include "core/ring.h"
#include "render/output_ring.h"
#include "render/render_device.h"

namespace ajp::render::test {

// Les quelques fonctions Vulkan dont les tests ont besoin, chargées par vkGetInstanceProcAddr.
struct VulkanCalls {
  explicit VulkanCalls(const GpuDevice& device);

  std::uint64_t counter(std::uint64_t semaphore) const;
  void signal(std::uint64_t semaphore, std::uint64_t value) const;

  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  PFN_vkGetPhysicalDeviceMemoryProperties memory_properties = nullptr;
  PFN_vkCreateImage create_image = nullptr;
  PFN_vkDestroyImage destroy_image = nullptr;
  PFN_vkGetImageMemoryRequirements image_requirements = nullptr;
  PFN_vkAllocateMemory allocate = nullptr;
  PFN_vkFreeMemory free = nullptr;
  PFN_vkBindImageMemory bind = nullptr;
  PFN_vkCreateSemaphore create_semaphore = nullptr;
  PFN_vkDestroySemaphore destroy_semaphore = nullptr;
  PFN_vkImportSemaphoreFdKHR import_semaphore = nullptr;
  PFN_vkGetSemaphoreCounterValue counter_value = nullptr;
  PFN_vkSignalSemaphore signal_semaphore = nullptr;
};

class ImportedSlot {
 public:
  ImportedSlot(const VulkanCalls& vk, pl_gpu gpu) : vk_(vk), gpu_(gpu) {}
  ImportedSlot(const ImportedSlot&) = delete;
  ImportedSlot& operator=(const ImportedSlot&) = delete;
  ImportedSlot(ImportedSlot&&) = delete;
  ImportedSlot& operator=(ImportedSlot&&) = delete;
  ~ImportedSlot();

  // Attend wait_value, remplit l'image de color, signale signal_value : un tour d'écrivain.
  void write(pl_gpu gpu, std::uint64_t wait_value, const std::array<float, 4>& color,
             std::uint64_t signal_value);

  const VulkanCalls& vk_;
  pl_gpu gpu_;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkSemaphore semaphore = VK_NULL_HANDLE;
  pl_tex tex = nullptr;

 private:
  bool first_ = true;
};

std::expected<std::unique_ptr<ImportedSlot>, std::string> import_slot(const VulkanCalls& vk,
                                                                      const RenderDevice& device,
                                                                      const OutputRing& ring,
                                                                      std::uint32_t slot);

}  // namespace ajp::render::test
