// Ensembles de descripteurs d'un enregistrement : un par dispatch, tirés d'un pool borné à sa
// création. Une demande au-delà des bornes est refusée (GpuMemoryExhausted) avant d'atteindre le
// pilote ; rien n'est jamais réécrit après l'enregistrement, qui peut être rejoué tel quel.
#pragma once

#include <cstdint>
#include <expected>
#include <span>

#include "core/types.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/pipeline.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

struct DescriptorLimits {
  std::uint32_t sets = 64;      // dispatchs par enregistrement
  std::uint32_t buffers = 512;  // tampons de stockage, tous ensembles confondus
  std::uint32_t images = 16;    // images, de stockage ou échantillonnées
};

class DescriptorArena {
 public:
  // Refus : FieldOutOfRange (borne nulle), VulkanCallFailed.
  static std::expected<DescriptorArena, Refusal> create(const VkContext& context, DescriptorLimits limits);

  DescriptorArena(DescriptorArena&& other) noexcept;
  DescriptorArena& operator=(DescriptorArena&& other) noexcept;
  DescriptorArena(const DescriptorArena&) = delete;
  DescriptorArena& operator=(const DescriptorArena&) = delete;
  ~DescriptorArena();

  // Un ensemble pour ce pipeline, ses tampons aux bindings 0..n−1 puis ses images (vues en
  // VK_IMAGE_LAYOUT_GENERAL) : celles de stockage (pipeline.images()), puis les échantillonnées
  // (pipeline.sampled(), avec l'échantillonneur immuable du pipeline). Refus : GpuMemoryExhausted
  // (bornes atteintes).
  std::expected<VkDescriptorSet, Refusal> bind(const Pipeline& pipeline, std::span<const BufferView> buffers,
                                               std::span<const VkImageView> images);

  std::uint32_t sets_used() const { return sets_; }

 private:
  DescriptorArena(const VkContext* context, VkDescriptorPool pool, DescriptorLimits limits)
      : context_(context), pool_(pool), limits_(limits) {}
  void destroy();

  const VkContext* context_ = nullptr;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  DescriptorLimits limits_;
  std::uint32_t sets_ = 0;
  std::uint32_t buffers_ = 0;
  std::uint32_t images_ = 0;
};

}  // namespace ajp::gpu::vk
