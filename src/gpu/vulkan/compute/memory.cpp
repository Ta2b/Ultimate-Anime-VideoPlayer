#include "gpu/vulkan/compute/memory.h"

#include <algorithm>
#include <format>
#include <optional>
#include <utility>

#include "gpu/vulkan/compute/refuse.h"

namespace ajp::gpu::vk {
namespace {

constexpr VkMemoryPropertyFlags kHostAccess =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
// Types que le socle n'utilise jamais : mémoire paresseuse (pièces jointes de rendu) et protégée.
constexpr VkMemoryPropertyFlags kExcluded =
    VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT | VK_MEMORY_PROPERTY_PROTECTED_BIT;

VkMemoryPropertyFlags required_flags(MemoryUse use) {
  return use == MemoryUse::Device ? 0 : kHostAccess;
}

VkMemoryPropertyFlags preferred_flags(MemoryUse use) {
  return use == MemoryUse::Readback ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT
                                    : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
}

const char* use_name(MemoryUse use) {
  switch (use) {
    case MemoryUse::Device:
      return "appareil";
    case MemoryUse::Upload:
      return "envoi";
    case MemoryUse::Readback:
      return "lecture";
  }
  return "?";
}

VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment) {
  return alignment <= 1 ? value : (value + alignment - 1) / alignment * alignment;
}

}  // namespace

MemoryLimits device_limits(const VkContext& context) {
  VkPhysicalDeviceMemoryProperties properties{};
  context.instance_functions().vkGetPhysicalDeviceMemoryProperties(context.physical_device(), &properties);
  VkDeviceSize largest = 0;
  for (std::uint32_t i = 0; i < properties.memoryHeapCount; ++i) {
    if ((properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
      largest = std::max(largest, properties.memoryHeaps[i].size);
  }
  MemoryLimits limits;
  const VkDeviceSize reserve = VkDeviceSize{512} << 20U;
  const VkDeviceSize sized = largest / 4 * 3;
  if (sized > reserve) limits.budget = std::max(limits.budget, sized - reserve);
  return limits;
}

Allocation::Allocation(Allocation&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      block_(other.block_),
      memory_(other.memory_),
      offset_(other.offset_),
      size_(other.size_),
      mapped_(other.mapped_) {}

Allocation& Allocation::operator=(Allocation&& other) noexcept {
  if (this != &other) {
    release();
    owner_ = std::exchange(other.owner_, nullptr);
    block_ = other.block_;
    memory_ = other.memory_;
    offset_ = other.offset_;
    size_ = other.size_;
    mapped_ = other.mapped_;
  }
  return *this;
}

Allocation::~Allocation() {
  release();
}

void Allocation::release() {
  if (owner_ != nullptr) owner_->release(block_, offset_, size_);
  owner_ = nullptr;
}

MemoryAllocator::MemoryAllocator(const VkContext& context, MemoryLimits limits)
    : context_(context), limits_(limits) {
  context_.instance_functions().vkGetPhysicalDeviceMemoryProperties(context_.physical_device(), &properties_);
}

std::expected<std::unique_ptr<MemoryAllocator>, Refusal> MemoryAllocator::create(const VkContext& context,
                                                                                 MemoryLimits limits) {
  if (limits.block_size == 0 || limits.budget < limits.block_size) {
    return refuse(
        RefusalCode::FieldOutOfRange,
        std::format("allocateur : bloc de {} o pour un budget de {} o", limits.block_size, limits.budget));
  }
  return std::unique_ptr<MemoryAllocator>(new MemoryAllocator(context, limits));
}

MemoryAllocator::~MemoryAllocator() {
  // Les portions encore vivantes ne doivent plus être utilisées : leurs blocs disparaissent ici,
  // sauf sous un GPU bloqué (VkContext::settle), qui les lit peut-être encore.
  if (context_.stalled()) return;
  for (const auto& block : blocks_) context_.fn().vkFreeMemory(context_.device(), block->memory, nullptr);
}

std::expected<std::uint32_t, Refusal> MemoryAllocator::choose_type(std::uint32_t allowed,
                                                                   MemoryUse use) const {
  const VkMemoryPropertyFlags required = required_flags(use);
  const VkMemoryPropertyFlags preferred = preferred_flags(use);
  std::optional<std::uint32_t> best;
  int best_score = -1;
  for (std::uint32_t i = 0; i < properties_.memoryTypeCount; ++i) {
    const VkMemoryPropertyFlags flags = properties_.memoryTypes[i].propertyFlags;
    if ((allowed & (1U << i)) == 0 || (flags & required) != required || (flags & kExcluded) != 0) continue;
    // Le type préféré d'abord ; pour l'appareil seul, un type invisible de l'hôte ensuite.
    int score = (flags & preferred) == preferred ? 2 : 0;
    if (use == MemoryUse::Device && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) score += 1;
    if (score > best_score) {
      best = i;
      best_score = score;
    }
  }
  if (!best) {
    return refuse(
        RefusalCode::FeatureUnavailable,
        std::format("aucun type de mémoire pour l'usage {} (types permis {:#x})", use_name(use), allowed));
  }
  return *best;
}

std::expected<MemoryAllocator::Block*, Refusal> MemoryAllocator::new_block(std::uint32_t type, Tiling tiling,
                                                                           VkDeviceSize size) {
  if (size > limits_.budget - reserved_) {
    return refuse(RefusalCode::GpuMemoryExhausted,
                  std::format("budget mémoire de {} o atteint : {} o réservés, bloc de {} o demandé",
                              limits_.budget, reserved_, size));
  }
  const VkMemoryAllocateInfo info{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = size, .memoryTypeIndex = type};
  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (auto ok = check(context_.fn().vkAllocateMemory(context_.device(), &info, nullptr, &memory),
                      std::format("vkAllocateMemory({} o, type {})", size, type));
      !ok) {
    return std::unexpected(ok.error());
  }
  void* mapped = nullptr;
  if ((properties_.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
    if (auto ok = check(context_.fn().vkMapMemory(context_.device(), memory, 0, VK_WHOLE_SIZE, 0, &mapped),
                        "vkMapMemory");
        !ok) {
      context_.fn().vkFreeMemory(context_.device(), memory, nullptr);
      return std::unexpected(ok.error());
    }
  }
  auto block = std::make_unique<Block>(Block{.id = next_id_++,
                                             .type = type,
                                             .tiling = tiling,
                                             .memory = memory,
                                             .size = size,
                                             .mapped = static_cast<std::byte*>(mapped),
                                             .free = {Range{.offset = 0, .size = size}},
                                             .live = 0});
  reserved_ += size;
  blocks_.push_back(std::move(block));
  return blocks_.back().get();
}

// Premier emplacement libre qui tient, une fois aligné ; ce qui reste de part et d'autre reste libre.
bool MemoryAllocator::carve(Block& block, VkDeviceSize size, VkDeviceSize alignment, VkDeviceSize& offset) {
  for (std::size_t i = 0; i < block.free.size(); ++i) {
    const Range range = block.free[i];
    const VkDeviceSize start = align_up(range.offset, alignment);
    const VkDeviceSize end = range.offset + range.size;
    if (start > end || end - start < size) continue;
    std::vector<Range> pieces;
    if (start > range.offset) pieces.push_back({.offset = range.offset, .size = start - range.offset});
    if (end > start + size) pieces.push_back({.offset = start + size, .size = end - start - size});
    const auto at = block.free.erase(block.free.begin() + static_cast<std::ptrdiff_t>(i));
    block.free.insert(at, pieces.begin(), pieces.end());
    offset = start;
    return true;
  }
  return false;
}

std::expected<Allocation, Refusal> MemoryAllocator::allocate(const VkMemoryRequirements& requirements,
                                                             MemoryUse use, Tiling tiling) {
  if (requirements.size == 0) return refuse(RefusalCode::FieldOutOfRange, "allocation de 0 o");
  auto type = choose_type(requirements.memoryTypeBits, use);
  if (!type) return std::unexpected(type.error());

  Block* chosen = nullptr;
  VkDeviceSize offset = 0;
  for (const auto& block : blocks_) {
    if (block->type == *type && block->tiling == tiling &&
        carve(*block, requirements.size, requirements.alignment, offset)) {
      chosen = block.get();
      break;
    }
  }
  if (chosen == nullptr) {
    // Une demande plus grande qu'un bloc a son bloc propre, à sa taille exacte.
    auto block = new_block(*type, tiling, std::max(requirements.size, limits_.block_size));
    if (!block) return std::unexpected(block.error());
    chosen = *block;
    carve(*chosen, requirements.size, requirements.alignment, offset);
  }
  ++chosen->live;
  used_ += requirements.size;
  Allocation allocation;
  allocation.owner_ = this;
  allocation.block_ = chosen->id;
  allocation.memory_ = chosen->memory;
  allocation.offset_ = offset;
  allocation.size_ = requirements.size;
  allocation.mapped_ = chosen->mapped == nullptr ? nullptr : chosen->mapped + offset;
  return allocation;
}

void MemoryAllocator::release(std::uint64_t id, VkDeviceSize offset, VkDeviceSize size) {
  const auto found = std::ranges::find_if(blocks_, [id](const auto& block) { return block->id == id; });
  if (found == blocks_.end()) return;
  Block& block = **found;
  used_ -= size;
  if (--block.live == 0 && !context_.stalled()) {
    // Un bloc vide retourne aussitôt à l'appareil : la mémoire réservée suit l'usage.
    context_.fn().vkFreeMemory(context_.device(), block.memory, nullptr);
    reserved_ -= block.size;
    blocks_.erase(found);
    return;
  }
  const auto at = std::ranges::lower_bound(block.free, offset, {}, &Range::offset);
  const auto inserted = block.free.insert(at, Range{.offset = offset, .size = size});
  // Recoud avec les voisines contiguës : la suivante, puis la précédente.
  auto next = inserted + 1;
  if (next != block.free.end() && inserted->offset + inserted->size == next->offset) {
    inserted->size += next->size;
    block.free.erase(next);
  }
  if (inserted != block.free.begin()) {
    auto previous = inserted - 1;
    if (previous->offset + previous->size == inserted->offset) {
      previous->size += inserted->size;
      block.free.erase(inserted);
    }
  }
}

}  // namespace ajp::gpu::vk
