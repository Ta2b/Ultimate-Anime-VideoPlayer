#include "gpu/vulkan/compute/memory.h"

#include <algorithm>
#include <format>
#include <functional>
#include <utility>
#include <vector>

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

std::vector<std::uint32_t> memory_candidates(const VkPhysicalDeviceMemoryProperties& properties,
                                             std::uint32_t allowed, MemoryUse use) {
  const VkMemoryPropertyFlags required = required_flags(use);
  const VkMemoryPropertyFlags preferred = preferred_flags(use);
  std::vector<std::pair<int, std::uint32_t>> scored;
  for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    const VkMemoryPropertyFlags flags = properties.memoryTypes[i].propertyFlags;
    if ((allowed & (1U << i)) == 0 || (flags & required) != required || (flags & kExcluded) != 0) continue;
    // Le type préféré d'abord ; pour l'appareil seul, un type invisible de l'hôte ensuite.
    int score = (flags & preferred) == preferred ? 2 : 0;
    if (use == MemoryUse::Device && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) score += 1;
    scored.emplace_back(score, i);
  }
  std::ranges::stable_sort(scored, std::greater{}, &std::pair<int, std::uint32_t>::first);
  // Pour l'appareil seul, pas de repli hors de DEVICE_LOCAL quand un tel type est permis.
  if (use == MemoryUse::Device && !scored.empty() && scored.front().first >= 2) {
    std::erase_if(scored, [](const auto& entry) { return entry.first < 2; });
  }
  std::vector<std::uint32_t> types;
  types.reserve(scored.size());
  for (const auto& [score, type] : scored) types.push_back(type);
  return types;
}

MemoryLimits device_limits(const VkContext& context) {
  const bool budget_known =
      std::ranges::find(context.caps().extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) !=
      context.caps().extensions.end();
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
                                               .pNext = budget_known ? &budget : nullptr};
  context.instance_functions().vkGetPhysicalDeviceMemoryProperties2(context.physical_device(), &properties);
  VkDeviceSize largest = 0;
  VkDeviceSize available = 0;  // sur le plus grand tas, ce que le pilote dit libre pour le processus
  for (std::uint32_t i = 0; i < properties.memoryProperties.memoryHeapCount; ++i) {
    const VkMemoryHeap& heap = properties.memoryProperties.memoryHeaps[i];
    if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0 || heap.size <= largest) continue;
    largest = heap.size;
    available = budget.heapBudget[i] > budget.heapUsage[i] ? budget.heapBudget[i] - budget.heapUsage[i] : 0;
  }
  MemoryLimits limits;
  const VkDeviceSize reserve = VkDeviceSize{512} << 20U;
  const VkDeviceSize sized = largest / 4 * 3;
  if (sized > reserve) limits.budget = std::max(limits.budget, sized - reserve);
  if (budget_known && largest > 0) {
    const VkDeviceSize floor = VkDeviceSize{1} << 30U;
    const VkDeviceSize real = available > reserve ? available - reserve : 0;
    limits.budget = std::max(floor, std::min(limits.budget, real));
  }
  limits.large_spares = std::min(limits.budget / 8, VkDeviceSize{1} << 30U);
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

std::size_t MemoryAllocator::spare_count() const {
  return static_cast<std::size_t>(
      std::ranges::count_if(blocks_, [](const auto& block) { return block->live == 0; }));
}

void MemoryAllocator::free_block(std::vector<std::unique_ptr<Block>>::iterator at) {
  context_.fn().vkFreeMemory(context_.device(), (*at)->memory, nullptr);
  reserved_ -= (*at)->size;
  blocks_.erase(at);
}

void MemoryAllocator::evict_spares(VkDeviceSize needed) {
  for (auto it = blocks_.begin(); it != blocks_.end() && needed > limits_.budget - reserved_;) {
    if ((*it)->live == 0) {
      free_block(it);
      it = blocks_.begin();
    } else {
      ++it;
    }
  }
}

std::expected<MemoryAllocator::Block*, Refusal> MemoryAllocator::new_block(std::uint32_t type, Tiling tiling,
                                                                           VkDeviceSize size,
                                                                           bool& heap_full) {
  heap_full = false;
  // La réserve d'abord : un bloc vide d'un autre type ou tiling ne sert pas cette demande.
  if (size > limits_.budget - reserved_ && !context_.stalled()) evict_spares(size);
  if (size > limits_.budget - reserved_) {
    return refuse(RefusalCode::GpuMemoryExhausted,
                  std::format("budget mémoire de {} o atteint : {} o réservés, bloc de {} o demandé",
                              limits_.budget, reserved_, size));
  }
  const VkMemoryAllocateInfo info{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = size, .memoryTypeIndex = type};
  VkDeviceMemory memory = VK_NULL_HANDLE;
  const VkResult allocated = context_.fn().vkAllocateMemory(context_.device(), &info, nullptr, &memory);
  if (auto ok = check(allocated, std::format("vkAllocateMemory({} o, type {})", size, type)); !ok) {
    heap_full = allocated == VK_ERROR_OUT_OF_DEVICE_MEMORY;
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
  const std::vector<std::uint32_t> types = memory_candidates(properties_, requirements.memoryTypeBits, use);
  if (types.empty()) {
    return refuse(RefusalCode::FeatureUnavailable,
                  std::format("aucun type de mémoire pour l'usage {} (types permis {:#x})", use_name(use),
                              requirements.memoryTypeBits));
  }

  Block* chosen = nullptr;
  VkDeviceSize offset = 0;
  Refusal full{.code = RefusalCode::GpuMemoryExhausted, .detail = {}};  // le dernier tas à court
  for (const std::uint32_t type : types) {
    for (const auto& block : blocks_) {
      if (block->type == type && block->tiling == tiling &&
          carve(*block, requirements.size, requirements.alignment, offset)) {
        chosen = block.get();
        break;
      }
    }
    if (chosen != nullptr) break;
    // Une demande plus grande qu'un bloc a son bloc propre, à sa taille exacte.
    bool heap_full = false;
    auto block = new_block(type, tiling, std::max(requirements.size, limits_.block_size), heap_full);
    if (block) {
      chosen = *block;
      carve(*chosen, requirements.size, requirements.alignment, offset);
      break;
    }
    if (!heap_full) return std::unexpected(std::move(block.error()));
    full = std::move(block.error());  // ce tas est plein : le type suivant
  }
  if (chosen == nullptr) return std::unexpected(std::move(full));
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
    // Un bloc vide de la taille courante reste en réserve, un par type et tiling : une chaîne
    // qui change (nouveau moteur, seek) relâche et redemande les mêmes tailles, et chaque
    // vkAllocateMemory coûte des millisecondes. Un bloc propre aussi, tant que les blocs propres
    // vides tiennent dans large_spares (les plus anciens partent d'abord). Les autres blocs vides
    // retournent à l'appareil : la mémoire réservée suit l'usage.
    block.free = {Range{.offset = 0, .size = block.size}};
    if (block.size != limits_.block_size) {
      if (block.size > limits_.large_spares) {
        free_block(found);
        return;
      }
      VkDeviceSize kept = block.size;
      for (const auto& other : blocks_) {
        if (other->id != id && other->live == 0 && other->size != limits_.block_size) kept += other->size;
      }
      for (auto it = blocks_.begin(); it != blocks_.end() && kept > limits_.large_spares;) {
        if ((*it)->id != id && (*it)->live == 0 && (*it)->size != limits_.block_size) {
          kept -= (*it)->size;
          free_block(it);
          it = blocks_.begin();
        } else {
          ++it;
        }
      }
      return;
    }
    const bool spare_present = std::ranges::any_of(blocks_, [&](const auto& other) {
      return other->id != id && other->live == 0 && other->size == limits_.block_size &&
             other->type == block.type && other->tiling == block.tiling;
    });
    if (spare_present) free_block(found);
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
