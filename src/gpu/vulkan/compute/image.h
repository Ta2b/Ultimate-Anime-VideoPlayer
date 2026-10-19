// Images de stockage RGBA16 UNORM, la sortie du ring. L'image est créée à format mutable : les
// noyaux l'écrivent par une vue R16G16B16A16_UINT (codes unorm16 exacts, sans la conversion
// flottante du matériel), les consommateurs la lisent par une vue UNORM. Elle reste en
// VK_IMAGE_LAYOUT_GENERAL pendant toute sa vie.
#pragma once

#include <cstdint>
#include <expected>
#include <vector>

#include "core/types.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

class StorageImage {
 public:
  static constexpr VkFormat kFormat = VK_FORMAT_R16G16B16A16_UNORM;
  static constexpr VkFormat kStorageFormat = VK_FORMAT_R16G16B16A16_UINT;

  // Refus : InputMismatch (dimensions hors de [1, maxImageDimension2D], allocateur d'un autre
  // contexte), FeatureUnavailable (format sans stockage ou sans échantillonnage), ceux de
  // l'allocateur, VulkanCallFailed.
  static std::expected<StorageImage, Refusal> create(VkContext& context, MemoryAllocator& memory,
                                                     std::uint32_t width, std::uint32_t height);

  StorageImage(StorageImage&& other) noexcept;
  StorageImage& operator=(StorageImage&& other) noexcept;
  StorageImage(const StorageImage&) = delete;
  StorageImage& operator=(const StorageImage&) = delete;
  ~StorageImage();

  const VkContext* context() const { return context_; }
  VkImage image() const { return image_; }
  VkImageView storage_view() const { return storage_view_; }  // R16G16B16A16_UINT, pour les noyaux
  VkImageView sampled_view() const { return sampled_view_; }  // R16G16B16A16_UNORM, pour la lecture
  std::uint32_t width() const { return width_; }
  std::uint32_t height() const { return height_; }

 private:
  explicit StorageImage(const VkContext* context) : context_(context) {}
  void destroy();

  const VkContext* context_ = nullptr;
  VkImage image_ = VK_NULL_HANDLE;
  Allocation allocation_;
  VkImageView storage_view_ = VK_NULL_HANDLE;
  VkImageView sampled_view_ = VK_NULL_HANDLE;
  std::uint32_t width_ = 0;
  std::uint32_t height_ = 0;
};

// Les codes RGBA16 de l'image, quatre par pixel, ligne par ligne : lecture hôte par un tampon de
// transit et une soumission attendue (tests, outils). Les écritures doivent être achevées.
// Refus InputMismatch : image vide ou d'un autre contexte.
std::expected<std::vector<std::uint16_t>, Refusal> read_rgba16(VkContext& context, MemoryAllocator& memory,
                                                               const StorageImage& image);

}  // namespace ajp::gpu::vk
