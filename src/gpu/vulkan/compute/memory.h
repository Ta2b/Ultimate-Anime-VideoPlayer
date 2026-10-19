// Sous-allocateur de mémoire d'appareil, borné : des blocs par type de mémoire (et par
// disposition, linéaire pour les tampons, optimale pour les images, ce qui évite la granularité
// tampon-image), découpés au premier emplacement libre et recousus à la libération. Une demande
// qui dépasse le budget, ou que l'appareil refuse, est refusée par son nom (GpuMemoryExhausted),
// jamais réduite. Les blocs visibles par l'hôte restent projetés pendant toute leur vie.
// Comme le contexte, l'allocateur appartient au thread qui l'utilise.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <vector>

#include "core/types.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

enum class MemoryUse : std::uint8_t {
  Device,    // lu et écrit par le GPU seul ; DEVICE_LOCAL de préférence
  Upload,    // écrit par l'hôte, lu par le GPU : HOST_VISIBLE et HOST_COHERENT, DEVICE_LOCAL de préférence
  Readback,  // écrit par le GPU, lu par l'hôte : HOST_VISIBLE et HOST_COHERENT, HOST_CACHED de préférence
};

enum class Tiling : std::uint8_t { Linear, Optimal };

struct MemoryLimits {
  VkDeviceSize block_size = VkDeviceSize{64} << 20U;  // une demande plus grande a son propre bloc
  VkDeviceSize budget = VkDeviceSize{2} << 30U;       // total des blocs réservés à l'appareil
};

class MemoryAllocator;

// Une portion d'un bloc, rendue à l'allocateur à sa destruction. L'allocateur doit lui survivre.
class Allocation {
 public:
  Allocation() = default;
  Allocation(Allocation&& other) noexcept;
  Allocation& operator=(Allocation&& other) noexcept;
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;
  ~Allocation();

  VkDeviceMemory memory() const { return memory_; }
  VkDeviceSize offset() const { return offset_; }
  VkDeviceSize size() const { return size_; }
  // Début de la portion dans la projection du bloc ; nullptr si le bloc n'est pas visible.
  std::byte* mapped() const { return mapped_; }

 private:
  friend class MemoryAllocator;
  void release();

  MemoryAllocator* owner_ = nullptr;
  std::uint64_t block_ = 0;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkDeviceSize offset_ = 0;
  VkDeviceSize size_ = 0;
  std::byte* mapped_ = nullptr;
};

class MemoryAllocator {
 public:
  // Refus : FieldOutOfRange (bloc nul, budget plus petit qu'un bloc).
  static std::expected<std::unique_ptr<MemoryAllocator>, Refusal> create(const VkContext& context,
                                                                         MemoryLimits limits = {});

  MemoryAllocator(const MemoryAllocator&) = delete;
  MemoryAllocator& operator=(const MemoryAllocator&) = delete;
  MemoryAllocator(MemoryAllocator&&) = delete;
  MemoryAllocator& operator=(MemoryAllocator&&) = delete;
  ~MemoryAllocator();

  // Refus : FeatureUnavailable (aucun type de mémoire ne convient), GpuMemoryExhausted (budget
  // atteint, ou vkAllocateMemory à court), VulkanCallFailed.
  std::expected<Allocation, Refusal> allocate(const VkMemoryRequirements& requirements, MemoryUse use,
                                              Tiling tiling);

  const VkContext& context() const { return context_; }
  const MemoryLimits& limits() const { return limits_; }
  VkDeviceSize reserved() const { return reserved_; }  // octets des blocs alloués à l'appareil
  VkDeviceSize used() const { return used_; }          // octets des portions vivantes
  std::size_t block_count() const { return blocks_.size(); }

 private:
  struct Range {
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
  };
  struct Block {
    std::uint64_t id = 0;
    std::uint32_t type = 0;
    Tiling tiling = Tiling::Linear;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    std::byte* mapped = nullptr;
    std::vector<Range> free;  // trié par offset, sans deux portions contiguës
    std::size_t live = 0;     // portions vivantes
  };

  MemoryAllocator(const VkContext& context, MemoryLimits limits);
  std::expected<std::uint32_t, Refusal> choose_type(std::uint32_t allowed, MemoryUse use) const;
  std::expected<Block*, Refusal> new_block(std::uint32_t type, Tiling tiling, VkDeviceSize size);
  static bool carve(Block& block, VkDeviceSize size, VkDeviceSize alignment, VkDeviceSize& offset);
  void release(std::uint64_t block, VkDeviceSize offset, VkDeviceSize size);

  const VkContext& context_;
  MemoryLimits limits_;
  VkPhysicalDeviceMemoryProperties properties_{};
  std::vector<std::unique_ptr<Block>> blocks_;
  std::uint64_t next_id_ = 1;
  VkDeviceSize reserved_ = 0;
  VkDeviceSize used_ = 0;

  friend class Allocation;
};

}  // namespace ajp::gpu::vk
