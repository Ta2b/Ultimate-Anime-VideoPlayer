// Banc Vulkan des tests d'interop : l'appareil Vulkan de même UUID que l'appareil CUDA, une
// file, et de quoi exporter une image RGBA16, un tampon et un sémaphore timeline comme render
// exportera les slots du ring. Le chargeur Vulkan s'ouvre à l'exécution (libvulkan.so.1) : sans
// lui, ou sans appareil de cet UUID (lavapipe seul), les tests se SKIP avec la raison.
#pragma once

#define VK_NO_PROTOTYPES
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "gpu/cuda/compute/device.h"

namespace ajp::gpu::cuda::test {

struct VulkanFunctions;

// Une image ou un tampon exporté : sa mémoire dédiée et le descripteur de fichier de celle-ci.
struct Exported {
  VkImage image = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;  // allocationSize
  int fd = -1;
};

class VulkanRig {
 public:
  // Le banc sur l'appareil Vulkan d'UUID donné ; nullptr et la raison sinon.
  static std::unique_ptr<VulkanRig> open(const Uuid& uuid, std::string& reason);
  ~VulkanRig();
  VulkanRig(const VulkanRig&) = delete;
  VulkanRig& operator=(const VulkanRig&) = delete;

  Exported export_image(int width, int height);
  Exported export_buffer(VkDeviceSize bytes);
  // Sémaphore timeline exporté ; fd reçoit son descripteur.
  VkSemaphore export_timeline(int& fd);

  // Image : UNDEFINED → GENERAL, puis cédée à VK_QUEUE_FAMILY_EXTERNAL (CUDA y écrira).
  void release_to_external(VkImage image);
  // Attend la valeur wait du sémaphore, reprend l'image de VK_QUEUE_FAMILY_EXTERNAL, la copie
  // dans un tampon hôte, puis signale signal. Rend les octets lus.
  std::vector<std::byte> read_image(VkImage image, int width, int height, VkSemaphore semaphore,
                                    std::uint64_t wait, std::uint64_t signal);
  // Remplit le tampon de value (vkCmdFillBuffer), le cède à VK_QUEUE_FAMILY_EXTERNAL et signale
  // signal ; n'attend pas : finish() attend la soumission en cours.
  void fill_buffer(VkBuffer buffer, VkDeviceSize bytes, std::uint32_t value, VkSemaphore semaphore,
                   std::uint64_t signal);
  void finish();

  void destroy(Exported& e);
  void destroy(VkSemaphore semaphore);

 private:
  VulkanRig() = default;
  std::uint32_t memory_type(std::uint32_t bits, VkMemoryPropertyFlags flags) const;
  VkCommandBuffer begin();
  void submit(VkCommandBuffer cmd, VkSemaphore semaphore, std::uint64_t wait, std::uint64_t signal,
              bool block);

  void* library_ = nullptr;
  std::unique_ptr<VulkanFunctions> fn_;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  std::uint32_t family_ = 0;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  VkCommandBuffer pending_ = VK_NULL_HANDLE;  // soumis, pas encore attendu
};

}  // namespace ajp::gpu::cuda::test
