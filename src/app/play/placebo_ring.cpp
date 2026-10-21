// Le ring de PlaceboScreen : textures RGBA16 exportées vers CUDA (descripteurs, HANDLE sous Windows), passage
// des slots en disposition GENERAL par VK_QUEUE_FAMILY_EXTERNAL (core/ring.h : kRingLayout).
#include <format>
#include <utility>

#include "app/play/placebo_impl.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

constexpr VkImageLayout kLayout = VK_IMAGE_LAYOUT_GENERAL;

// Les poignées que le backend TensorRT importe dans CUDA : descripteurs sous Linux, HANDLE NT
// sous Windows.
#ifdef _WIN32
constexpr pl_handle_type kExport = PL_HANDLE_WIN32;
constexpr HandleKind kExportKind = HandleKind::OpaqueWin32;
std::uintptr_t raw_handle(const pl_handle& handle) {
  return reinterpret_cast<std::uintptr_t>(handle.handle);
}
void close_handle(std::uintptr_t handle) {
  CloseHandle(reinterpret_cast<HANDLE>(handle));
}
#else
constexpr pl_handle_type kExport = PL_HANDLE_FD;
constexpr HandleKind kExportKind = HandleKind::OpaqueFd;
std::uintptr_t raw_handle(const pl_handle& handle) {
  return static_cast<std::uintptr_t>(handle.fd);
}
void close_handle(std::uintptr_t handle) {
  close(static_cast<int>(handle));
}
#endif
static_assert(static_cast<std::uint32_t>(kLayout) == kRingLayout);

// Cède tex à CUDA : la transition et le transfert de file sont faits quand sem atteint value.
bool hand_to_cuda(pl_gpu gpu, pl_tex tex, VkSemaphore sem, std::uint64_t value) {
  pl_vulkan_hold_params hold{};
  hold.tex = tex;
  hold.layout = kLayout;
  hold.qf = VK_QUEUE_FAMILY_EXTERNAL;
  hold.semaphore = {.sem = sem, .value = value};
  return pl_vulkan_hold_ex(gpu, &hold);
}

// Reprend tex de CUDA : libplacebo attend que sem atteigne value avant de la lire.
void take_from_cuda(pl_gpu gpu, pl_tex tex, VkSemaphore sem, std::uint64_t value) {
  pl_vulkan_release_params release{};
  release.tex = tex;
  release.layout = kLayout;
  release.qf = VK_QUEUE_FAMILY_EXTERNAL;
  release.semaphore = {.sem = sem, .value = value};
  pl_vulkan_release_ex(gpu, &release);
}

}  // namespace

std::expected<std::vector<RingSlotDesc>, Refusal> PlaceboRing::create(pl_gpu gpu, int width, int height,
                                                                      std::uint32_t count) {
  clear(gpu);
  ++generation_;
  const pl_fmt format = pl_find_named_fmt(gpu, "rgba16");
  if (format == nullptr || (format->caps & PL_FMT_CAP_STORABLE) == 0) {
    return refuse(RefusalCode::FormatUnsupported, "rgba16 n'est pas inscriptible sur cet appareil");
  }
  // Lecture des sémaphores sans attente (ready) : Vulkan 1.2, ou VK_KHR_timeline_semaphore.
  const pl_vulkan vk = pl_vulkan_get(gpu);
  device_ = VK_NULL_HANDLE;
  counter_ = nullptr;
  if (vk != nullptr) {
    const auto device_proc =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(vk->get_proc_addr(vk->instance, "vkGetDeviceProcAddr"));
    if (device_proc != nullptr) {
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
  const bool exported = (gpu->export_caps.tex & kExport) != 0 && (gpu->export_caps.sync & kExport) != 0;
  const pl_handle_type handles = exported ? kExport : pl_handle_type{};
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
                    hand_to_cuda(gpu, slot.tex, handover, std::uint64_t{i} + 1);
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
    descs.push_back(RingSlotDesc{
        .index = i,
        .generation = generation_,
        .width = width,
        .height = height,
        .format = RingFormat::Rgba16Unorm,
        .image = reinterpret_cast<std::uint64_t>(image),
        .memory = {.kind = kind, .value = exported ? raw_handle(slot.tex->shared_mem.handle) : 0},
        .memory_size = slot.tex->shared_mem.size,
        .memory_offset = slot.tex->shared_mem.offset,
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
    take_from_cuda(gpu, slot.tex, slot.semaphore, frame.ready_value);
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
      take_from_cuda(gpu, slot.tex, slot.semaphore, frame.ready_value);
    hand_to_cuda(gpu, slot.tex, slot.semaphore, frame.ready_value + 1);
    slot.placebo_value = 0;
  }
  return release_of(frame);
}

void PlaceboRing::clear(pl_gpu gpu) {
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
