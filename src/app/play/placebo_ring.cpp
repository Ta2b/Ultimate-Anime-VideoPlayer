// Le ring de PlaceboScreen : textures RGBA16 exportées vers CUDA (descripteurs, HANDLE sous Windows), passage
// des slots en disposition GENERAL par VK_QUEUE_FAMILY_EXTERNAL (core/ring.h : kRingLayout).
#include <format>
#include <utility>

#include "app/play/placebo_impl.h"
#include "core/log.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ajp::app::play {
namespace {

constexpr VkImageLayout kLayout = VK_IMAGE_LAYOUT_GENERAL;

// Les poignées que le backend TensorRT importe dans CUDA : descripteurs sous Linux, HANDLE NT
// sous Windows.
#ifdef _WIN32
constexpr HandleKind kExportKind = HandleKind::OpaqueWin32;
std::uintptr_t raw_handle(const pl_handle& handle) {
  return reinterpret_cast<std::uintptr_t>(handle.handle);
}
void close_handle(std::uintptr_t handle) {
  CloseHandle(reinterpret_cast<HANDLE>(handle));
}
#else
constexpr HandleKind kExportKind = HandleKind::OpaqueFd;
std::uintptr_t raw_handle(const pl_handle& handle) {
  return static_cast<std::uintptr_t>(handle.fd);
}
void close_handle(std::uintptr_t handle) {
  close(static_cast<int>(handle));
}
#endif
static_assert(static_cast<std::uint32_t>(kLayout) == kRingLayout);

// libplacebo dédie l'allocation d'une image (VkMemoryDedicatedAllocateInfo) quand le pilote le préfère
// (vulkan/gpu_tex.c, prefersDedicatedAllocation) ; sinon l'image est une page d'un bloc partagé, même à
// un décalage nul. CUDA doit le savoir pour mapper l'image avec la bonne disposition (ring_slot.cpp).
bool dedicated_image(PFN_vkGetImageMemoryRequirements2 requirements, VkDevice device, VkImage image) {
  if (requirements == nullptr || device == VK_NULL_HANDLE || image == VK_NULL_HANDLE) return false;
  VkMemoryDedicatedRequirements dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
  VkMemoryRequirements2 reqs{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated};
  const VkImageMemoryRequirementsInfo2 info{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                                            .image = image};
  requirements(device, &info, &reqs);
  return dedicated.prefersDedicatedAllocation == VK_TRUE;
}

// Cède tex à CUDA : la transition et le transfert de file sont faits quand sem atteint value.
// qf : VK_QUEUE_FAMILY_EXTERNAL pour CUDA, VK_QUEUE_FAMILY_IGNORED sur le même VkDevice (le backend
// Vulkan écrit le slot sans acquisition : aucun transfert de propriété, core/ring.h).
bool hand_to_cuda(pl_gpu gpu, pl_tex tex, VkSemaphore sem, std::uint64_t value, std::uint32_t qf) {
  pl_vulkan_hold_params hold{};
  hold.tex = tex;
  hold.layout = kLayout;
  hold.qf = qf;
  hold.semaphore = {.sem = sem, .value = value};
  return pl_vulkan_hold_ex(gpu, &hold);
}

// Reprend tex de CUDA : libplacebo attend que sem atteigne value avant de la lire.
void take_from_cuda(pl_gpu gpu, pl_tex tex, VkSemaphore sem, std::uint64_t value, std::uint32_t qf) {
  pl_vulkan_release_params release{};
  release.tex = tex;
  release.layout = kLayout;
  release.qf = qf;
  release.semaphore = {.sem = sem, .value = value};
  pl_vulkan_release_ex(gpu, &release);
}

}  // namespace

std::uint32_t PlaceboRing::family() const {
  return external_ ? VK_QUEUE_FAMILY_EXTERNAL : VK_QUEUE_FAMILY_IGNORED;
}

std::expected<std::vector<RingSlotDesc>, Refusal> PlaceboRing::create(pl_gpu gpu, int width, int height,
                                                                      std::uint32_t count, bool external,
                                                                      RingFormat wanted) {
  clear(gpu);
  external_ = external;
  ++generation_;
  // RGB10A2 (étape 5.5a) : inscriptible et échantillonnable, sinon RGBA16 ; le format Vulkan est contrôlé
  // sur le premier emplacement (l'ordre des bits doit être celui de store_ring).
  constexpr auto kRingCaps = static_cast<pl_fmt_caps>(PL_FMT_CAP_STORABLE | PL_FMT_CAP_SAMPLEABLE);
  if (wanted == RingFormat::Rgb10a2Unorm) {
    const pl_fmt ten = pl_find_named_fmt(gpu, "rgb10a2");
    if (ten == nullptr || (ten->caps & kRingCaps) != kRingCaps) wanted = RingFormat::Rgba16Unorm;
  }
  const bool ten_bits = wanted == RingFormat::Rgb10a2Unorm;
  const pl_fmt format = pl_find_named_fmt(gpu, ten_bits ? "rgb10a2" : "rgba16");
  if (format == nullptr || (format->caps & PL_FMT_CAP_STORABLE) == 0) {
    return refuse(RefusalCode::FormatUnsupported, "rgba16 n'est pas inscriptible sur cet appareil");
  }
  // Lecture des sémaphores sans attente (ready) : Vulkan 1.2, ou VK_KHR_timeline_semaphore.
  const pl_vulkan vk = pl_vulkan_get(gpu);
  device_ = VK_NULL_HANDLE;
  counter_ = nullptr;
  PFN_vkGetImageMemoryRequirements2 requirements = nullptr;
  if (vk != nullptr) {
    const auto device_proc =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(vk->get_proc_addr(vk->instance, "vkGetDeviceProcAddr"));
    if (device_proc != nullptr) {
      requirements = reinterpret_cast<PFN_vkGetImageMemoryRequirements2>(
          device_proc(vk->device, "vkGetImageMemoryRequirements2"));
      if (requirements == nullptr) {
        requirements = reinterpret_cast<PFN_vkGetImageMemoryRequirements2>(
            device_proc(vk->device, "vkGetImageMemoryRequirements2KHR"));
      }
      counter_ = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
          device_proc(vk->device, "vkGetSemaphoreCounterValue"));
      if (counter_ == nullptr) {
        counter_ = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
            device_proc(vk->device, "vkGetSemaphoreCounterValueKHR"));
      }
    }
    device_ = vk->device;
  }
  // Sémaphore de la première cession, attendu sur l'hôte une fois pour tout le ring : le slot
  // doit être en GENERAL côté CUDA avant sa première écriture, qui n'attend que 0.
  pl_vulkan_sem_params temp_params{};
  temp_params.type = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphore handover = pl_vulkan_sem_create(gpu, &temp_params);
  if (handover == VK_NULL_HANDLE) return refuse(RefusalCode::VendorCallFailed, "pl_vulkan_sem_create");

  // Exporter la mémoire et les sémaphores ne sert qu'au backend TensorRT (import dans CUDA) ; le
  // backend Vulkan écrit sur l'appareil même du rendu. Un appareil qui n'exporte pas (lavapipe)
  // garde un ring sans poignées.
  const bool exported =
      external && (gpu->export_caps.tex & kRingHandle) != 0 && (gpu->export_caps.sync & kRingHandle) != 0;
  const pl_handle_type handles = exported ? kRingHandle : pl_handle_type{};
  const HandleKind kind = exported ? kExportKind : HandleKind::None;
  std::vector<RingSlotDesc> descs;
  for (std::uint32_t i = 0; i < count; ++i) {
    Slot slot;
    pl_tex_params params{};
    params.w = width;
    params.h = height;
    params.format = format;
    params.sampleable = true;
    params.storable = true;
    params.export_handle = handles;
    slot.tex = pl_tex_create(gpu, &params);
    union pl_handle handle{};
    pl_vulkan_sem_params sem_params{};
    sem_params.type = VK_SEMAPHORE_TYPE_TIMELINE;
    sem_params.export_handle = handles;
    sem_params.out_handle = exported ? &handle : nullptr;
    slot.semaphore = pl_vulkan_sem_create(gpu, &sem_params);
    if (exported && slot.semaphore != VK_NULL_HANDLE) slot.semaphore_export = raw_handle(handle);
    const bool ok = slot.tex != nullptr && slot.semaphore != VK_NULL_HANDLE &&
                    hand_to_cuda(gpu, slot.tex, handover, std::uint64_t{i} + 1, family());
    slots_.push_back(slot);
    if (!ok) {
      pl_gpu_finish(gpu);
      pl_vulkan_sem_destroy(gpu, &handover);
      clear(gpu);
      return refuse(RefusalCode::VendorCallFailed, std::format("slot {} : création ou cession refusée", i));
    }
    VkFormat vk_format{};
    VkImageUsageFlags usage{};
    const VkImage image = pl_vulkan_unwrap(gpu, slot.tex, &vk_format, &usage);
    if (ten_bits && vk_format != VK_FORMAT_A2B10G10R10_UNORM_PACK32) {
      // Un autre ordre des bits que celui de store_ring : tout est refait en RGBA16.
      pl_gpu_finish(gpu);
      pl_vulkan_sem_destroy(gpu, &handover);
      clear(gpu);
      return create(gpu, width, height, count, external, RingFormat::Rgba16Unorm);
    }
    descs.push_back(RingSlotDesc{
        .index = i,
        .generation = generation_,
        .width = width,
        .height = height,
        .format = wanted,
        .image = reinterpret_cast<std::uint64_t>(image),
        .memory = {.kind = kind, .value = exported ? raw_handle(slot.tex->shared_mem.handle) : 0},
        .memory_size = slot.tex->shared_mem.size,
        .memory_offset = slot.tex->shared_mem.offset,
        .dedicated = exported && vk != nullptr && dedicated_image(requirements, vk->device, image),
        .semaphore = reinterpret_cast<std::uint64_t>(slot.semaphore),
        .semaphore_handle = {.kind = kind, .value = slot.semaphore_export.value_or(0)},
    });
  }
  pl_gpu_finish(gpu);
  pl_vulkan_sem_destroy(gpu, &handover);
  return descs;
}

std::expected<pl_tex, Refusal> PlaceboRing::slot_of(const RingFrame& frame) {
  if (frame.generation != generation_ || frame.slot >= slots_.size()) {
    return refuse(RefusalCode::SequenceOrder,
                  std::format("RingFrame du slot {} génération {} : ring courant {}", frame.slot,
                              frame.generation, generation_));
  }
  return slots_[frame.slot].tex;
}

std::expected<pl_tex, Refusal> PlaceboRing::acquire(pl_gpu gpu, const RingFrame& frame) {
  auto tex = slot_of(frame);
  if (!tex) return tex;
  Slot& slot = slots_[frame.slot];
  if (slot.placebo_value != frame.ready_value) {
    take_from_cuda(gpu, slot.tex, slot.semaphore, frame.ready_value, family());
    slot.placebo_value = frame.ready_value;
  }
  return slot.tex;
}

bool PlaceboRing::ready(const RingFrame& frame) const {
  // Hors du ring courant, ou sans moyen de lire : vrai, acquire ou retire en décident.
  if (counter_ == nullptr || frame.generation != generation_ || frame.slot >= slots_.size()) return true;
  std::uint64_t value = 0;
  if (counter_(device_, slots_[frame.slot].semaphore, &value) != VK_SUCCESS) return true;
  return value >= frame.ready_value;
}

SlotFree PlaceboRing::retire(pl_gpu gpu, const RingFrame& frame) {
  if (slot_of(frame)) {
    Slot& slot = slots_[frame.slot];
    // Jamais montrée : libplacebo attend tout de même ready_value avant de la rendre, pour que
    // les valeurs du sémaphore restent croissantes (core/ring.h).
    if (slot.placebo_value != frame.ready_value)
      take_from_cuda(gpu, slot.tex, slot.semaphore, frame.ready_value, family());
    // Cession refusée (mémoire de commandes, appareil en échec) : le sémaphore n'atteindra jamais la valeur
    // que l'écriture suivante attend. Le slot n'est pas rendu au producteur, et le journal le dit (A10-09).
    if (!hand_to_cuda(gpu, slot.tex, slot.semaphore, frame.ready_value + 1, family())) {
      log_channel("app")->error("anneau : le slot {} n'a pas pu être rendu au moteur (pl_vulkan_hold_ex)",
                                frame.slot);
      slot.placebo_value = 0;
      return SlotFree{.slot = frame.slot, .semaphore_value = 0, .generation = UINT32_MAX};
    }
    slot.placebo_value = 0;
  }
  return release_of(frame);
}

VkSemaphore PlaceboRing::semaphore_of(const RingFrame& frame) const {
  if (frame.generation != generation_ || frame.slot >= slots_.size()) return VK_NULL_HANDLE;
  return slots_[frame.slot].semaphore;
}

void PlaceboRing::clear(pl_gpu gpu) {
  if (watcher_ != nullptr) watcher_->forget_all();  // ses sémaphores vont être détruits
  if (gpu != nullptr && !slots_.empty()) pl_gpu_finish(gpu);
  for (Slot& slot : slots_) {
    if (gpu != nullptr) {
      pl_tex_destroy(gpu, &slot.tex);
      pl_vulkan_sem_destroy(gpu, &slot.semaphore);
    }
    if (slot.semaphore_export) close_handle(*slot.semaphore_export);
  }
  slots_.clear();
}

}  // namespace ajp::app::play
