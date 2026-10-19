// Tampons de stockage : lus et écrits par les noyaux, source et cible de transferts. Envoi et
// lecture hôte : directs par la projection pour les usages Upload et Readback, par un tampon de
// transit et une soumission attendue (VkContext::run_once) pour l'usage Device.
#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <utility>

#include "core/types.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

// Une plage d'un tampon, telle qu'un noyau la lie. Le tampon doit vivre plus longtemps que
// tout enregistrement qui la lie.
struct BufferView {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  VkDeviceSize size = 0;
};

class Buffer {
 public:
  // Refus : FieldOutOfRange (taille nulle), InputMismatch (allocateur d'un autre contexte), ceux de
  // MemoryAllocator::allocate, VulkanCallFailed.
  static std::expected<Buffer, Refusal> create(const VkContext& context, MemoryAllocator& memory,
                                               VkDeviceSize size, MemoryUse use);

  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer();

  const VkContext* context() const { return context_; }
  VkBuffer handle() const { return buffer_; }
  VkDeviceSize size() const { return size_; }
  MemoryUse use() const { return use_; }
  // Projection hôte des usages Upload et Readback ; nullptr pour Device.
  std::byte* mapped() const { return use_ == MemoryUse::Device ? nullptr : allocation_.mapped(); }

  BufferView view() const { return {.buffer = buffer_, .offset = 0, .size = size_}; }
  // Refus InputMismatch : plage vide ou hors du tampon.
  std::expected<BufferView, Refusal> view(VkDeviceSize offset, VkDeviceSize size) const;

 private:
  Buffer(const VkContext* context, VkBuffer buffer, Allocation allocation, VkDeviceSize size, MemoryUse use)
      : context_(context), buffer_(buffer), allocation_(std::move(allocation)), size_(size), use_(use) {}
  void destroy();

  const VkContext* context_ = nullptr;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  Allocation allocation_;
  VkDeviceSize size_ = 0;
  MemoryUse use_ = MemoryUse::Device;
};

// Écrit bytes à offset dans le tampon. Le GPU ne doit pas lire ni écrire cette plage pendant
// l'envoi. Usage Device : une soumission attendue sur l'hôte. Refus InputMismatch : hors du tampon,
// tampon ou allocateur d'un autre contexte (de même pour download).
Done upload(VkContext& context, MemoryAllocator& memory, const Buffer& buffer, VkDeviceSize offset,
            std::span<const std::byte> bytes);

// Lit out.size() octets depuis offset. Les écritures du GPU doivent être achevées (attente
// hôte de leur soumission). Usage Device : une soumission attendue sur l'hôte.
Done download(VkContext& context, MemoryAllocator& memory, const Buffer& buffer, VkDeviceSize offset,
              std::span<std::byte> out);

}  // namespace ajp::gpu::vk
