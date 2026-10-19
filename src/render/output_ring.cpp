#include "render/output_ring.h"

#include <array>
#include <format>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <libplacebo/gpu.h>
#include <libplacebo/vulkan.h>

#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>
#endif

#include "render/c_structs.h"
#include "render/render_device.h"
#include "render/vk_util.h"

namespace ajp::render {
namespace {

// Le slot est cédé à l'écrivain dans cette disposition, et lui revient dans la même.
constexpr VkImageLayout kWriterLayout = VK_IMAGE_LAYOUT_GENERAL;

// Le type de handle de libplacebo ; aucun sans export.
std::optional<pl_handle_type> to_placebo(HandleKind kind) {
  switch (kind) {
    case HandleKind::OpaqueFd:
      return PL_HANDLE_FD;
    case HandleKind::OpaqueWin32:
      return PL_HANDLE_WIN32;
    case HandleKind::None:
      break;
  }
  return std::nullopt;
}

std::uintptr_t handle_value(HandleKind kind, const pl_handle& handle) {
  if (kind == HandleKind::OpaqueFd) return static_cast<std::uintptr_t>(handle.fd);
  return reinterpret_cast<std::uintptr_t>(handle.handle);
}

void close_handle(const ExportedHandle& handle) {
#ifndef _WIN32
  if (handle.kind == HandleKind::OpaqueFd) close(static_cast<int>(handle.value));
#else
  if (handle.kind == HandleKind::OpaqueWin32) CloseHandle(reinterpret_cast<HANDLE>(handle.value));
#endif
}

Refusal out_of_range(std::string detail) {
  return Refusal{RefusalCode::FieldOutOfRange, std::move(detail)};
}

}  // namespace

struct OutputRing::Impl {
  enum class State : std::uint8_t { Writer, Render };
  struct Slot {
    pl_tex tex = nullptr;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    State state = State::Writer;
    std::uint64_t released_value = 0;  // dernière valeur cédée à l'écrivain
    bool dedicated = false;
  };

  pl_gpu gpu = nullptr;
  std::uint32_t generation = 0;
  std::uint32_t queue_family = VK_QUEUE_FAMILY_IGNORED;
  VkImageUsageFlags usage = 0;
  std::vector<Slot> slots;
  std::vector<RingSlotDesc> descs;

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  ~Impl() {
    if (gpu == nullptr) return;
    pl_gpu_finish(gpu);
    for (std::size_t i = 0; i < slots.size(); ++i) {
      if (i < descs.size()) close_handle(descs[i].semaphore_handle);
      pl_vulkan_sem_destroy(gpu, &slots[i].semaphore);
      pl_tex_destroy(gpu, &slots[i].tex);
    }
  }

  std::expected<void, Refusal> add_slot(const RenderDevice& device, const OutputRingOptions& options,
                                        pl_fmt format) {
    const auto index = static_cast<std::uint32_t>(slots.size());
    const std::optional<pl_handle_type> handle_type = to_placebo(options.export_kind);
    Slot& slot = slots.emplace_back();
    auto tex_params = detail::zeroed<pl_tex_params>();
    tex_params.w = options.width;
    tex_params.h = options.height;
    tex_params.format = format;
    tex_params.sampleable = true;
    tex_params.storable = true;
    tex_params.blit_dst = detail::has_all(format->caps, detail::flag_bits(PL_FMT_CAP_BLITTABLE));
    if (handle_type) tex_params.export_handle = *handle_type;
    tex_params.debug_tag = "ajp ring slot";
    slot.tex = pl_tex_create(gpu, &tex_params);
    if (slot.tex == nullptr)
      return std::unexpected(Refusal{RefusalCode::RenderFailed, "pl_tex_create (slot du ring)"});

    pl_handle sem_handle{};
    auto sem_params = detail::zeroed<pl_vulkan_sem_params>();
    sem_params.type = VK_SEMAPHORE_TYPE_TIMELINE;
    sem_params.initial_value = 0;
    if (handle_type) {
      sem_params.export_handle = *handle_type;
      sem_params.out_handle = &sem_handle;
    }
    sem_params.debug_tag = "ajp ring slot";
    slot.semaphore = pl_vulkan_sem_create(gpu, &sem_params);
    if (slot.semaphore == VK_NULL_HANDLE) {
      return std::unexpected(Refusal{RefusalCode::RenderFailed, "pl_vulkan_sem_create (slot du ring)"});
    }

    RingSlotDesc desc{.index = index,
                      .generation = generation,
                      .width = options.width,
                      .height = options.height,
                      .format = options.format};
    desc.semaphore = detail::handle_to_u64(slot.semaphore);
    if (handle_type) {
      desc.memory = {.kind = options.export_kind,
                     .value = handle_value(options.export_kind, slot.tex->shared_mem.handle)};
      desc.memory_size = slot.tex->shared_mem.size;
      desc.memory_offset = slot.tex->shared_mem.offset;
      desc.semaphore_handle = {.kind = options.export_kind,
                               .value = handle_value(options.export_kind, sem_handle)};
    }
    VkFormat vk_format = VK_FORMAT_UNDEFINED;
    const VkImage image = pl_vulkan_unwrap(gpu, slot.tex, &vk_format, &usage);
    desc.image = detail::handle_to_u64(image);
    descs.push_back(desc);
    slot.dedicated = prefers_dedicated(device, image);
    return {};
  }

  // libplacebo prend une allocation dédiée quand le pilote la préfère (gpu_tex.c) : la même
  // question au pilote donne la même réponse.
  static bool prefers_dedicated(const RenderDevice& device, VkImage image) {
    const pl_vulkan vk = device.vulkan();
    const VkDevice vk_device = vk->device;
    const auto gdpa = detail::instance_function<PFN_vkGetDeviceProcAddr>(vk->get_proc_addr, vk->instance,
                                                                         "vkGetDeviceProcAddr");
    const auto get_reqs =
        reinterpret_cast<PFN_vkGetImageMemoryRequirements2>(gdpa(vk_device, "vkGetImageMemoryRequirements2"));
    VkMemoryDedicatedRequirements dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 reqs{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated};
    const VkImageMemoryRequirementsInfo2 info{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                                              .image = image};
    get_reqs(vk_device, &info, &reqs);
    return dedicated.prefersDedicatedAllocation == VK_TRUE ||
           dedicated.requiresDedicatedAllocation == VK_TRUE;
  }

  // Efface chaque slot, puis le cède à l'écrivain. Le hold signale un sémaphore jetable, attendu
  // ici une fois pour toutes : le sémaphore du slot reste à 0, comme le protocole le veut.
  std::expected<void, Refusal> hand_over_initial() {
    auto temp_params = detail::zeroed<pl_vulkan_sem_params>();
    temp_params.type = VK_SEMAPHORE_TYPE_TIMELINE;
    temp_params.debug_tag = "ajp ring hold initial";
    VkSemaphore temp = pl_vulkan_sem_create(gpu, &temp_params);
    if (temp == VK_NULL_HANDLE)
      return std::unexpected(Refusal{RefusalCode::RenderFailed, "pl_vulkan_sem_create"});
    bool ok = true;
    std::uint64_t value = 0;
    for (const Slot& slot : slots) {
      if (slot.tex->params.blit_dst) pl_tex_clear(gpu, slot.tex, std::array<float, 4>{}.data());
      const pl_vulkan_hold_params hold{.tex = slot.tex,
                                       .layout = kWriterLayout,
                                       .qf = queue_family,
                                       .semaphore = {.sem = temp, .value = ++value}};
      ok = ok && pl_vulkan_hold_ex(gpu, &hold);
    }
    pl_gpu_finish(gpu);
    pl_vulkan_sem_destroy(gpu, &temp);
    if (!ok)
      return std::unexpected(Refusal{RefusalCode::RenderFailed, "pl_vulkan_hold_ex (cession initiale)"});
    return {};
  }

  std::expected<Slot*, Refusal> find(std::uint32_t index) {
    if (index >= slots.size()) {
      return std::unexpected(out_of_range(std::format("slot {} hors du ring de {}", index, slots.size())));
    }
    return &slots[index];
  }
};

std::expected<OutputRing, Refusal> OutputRing::create(const RenderDevice& device,
                                                      const OutputRingOptions& options) {
  const pl_gpu gpu = device.gpu();
  if (options.slots == 0 || options.slots > ring_limits::kMaxSlots) {
    return std::unexpected(
        out_of_range(std::format("ring.slots = {} (1 à {})", options.slots, ring_limits::kMaxSlots)));
  }
  if (options.width <= 0 || options.height <= 0 ||
      static_cast<std::uint32_t>(options.width) > gpu->limits.max_tex_2d_dim ||
      static_cast<std::uint32_t>(options.height) > gpu->limits.max_tex_2d_dim) {
    return std::unexpected(out_of_range(std::format("ring {}x{} (1 à {} par côté)", options.width,
                                                    options.height, gpu->limits.max_tex_2d_dim)));
  }
  if (options.format != RingFormat::Rgba16Unorm) {
    return std::unexpected(
        Refusal{RefusalCode::FormatUnsupported, "ring Yuv444p16 : pas encore composé par render"});
  }
  const pl_fmt format = pl_find_named_fmt(gpu, "rgba16");
  if (format == nullptr ||
      !detail::has_all(format->caps, detail::flag_bits(PL_FMT_CAP_SAMPLEABLE, PL_FMT_CAP_STORABLE))) {
    return std::unexpected(Refusal{RefusalCode::FeatureUnavailable, "rgba16 échantillonnable et stockable"});
  }
  const std::optional<pl_handle_type> handle_type = to_placebo(options.export_kind);
  const auto caps = handle_type ? static_cast<pl_handle_caps>(*handle_type) : pl_handle_caps{0};
  if (caps != 0 && ((gpu->export_caps.tex & caps) == 0 || (gpu->export_caps.sync & caps) == 0)) {
    return std::unexpected(
        Refusal{RefusalCode::FeatureUnavailable, "export de la mémoire ou des sémaphores du ring"});
  }

  auto impl = std::make_unique<Impl>();
  impl->gpu = gpu;
  impl->generation = options.generation;
  impl->queue_family =
      options.writer == SlotWriter::ExternalApi ? VK_QUEUE_FAMILY_EXTERNAL : VK_QUEUE_FAMILY_IGNORED;
  impl->slots.reserve(options.slots);
  for (std::uint32_t i = 0; i < options.slots; ++i) {
    if (auto ok = impl->add_slot(device, options, format); !ok) return std::unexpected(ok.error());
  }
  if (auto ok = impl->hand_over_initial(); !ok) return std::unexpected(ok.error());
  return OutputRing(std::move(impl));
}

OutputRing::OutputRing(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
OutputRing::OutputRing(OutputRing&&) noexcept = default;
OutputRing& OutputRing::operator=(OutputRing&&) noexcept = default;
OutputRing::~OutputRing() = default;

std::span<const RingSlotDesc> OutputRing::slots() const {
  return impl_->descs;
}

std::uint32_t OutputRing::generation() const {
  return impl_->generation;
}

bool OutputRing::dedicated(std::uint32_t slot) const {
  return slot < impl_->slots.size() && impl_->slots[slot].dedicated;
}

std::uint32_t OutputRing::image_usage() const {
  return impl_->usage;
}

std::expected<const pl_tex_t*, Refusal> OutputRing::acquire(const RingFrame& frame) {
  if (frame.generation != impl_->generation) {
    return std::unexpected(
        Refusal{RefusalCode::SequenceOrder,
                std::format("RingFrame de la génération {} (ring {})", frame.generation, impl_->generation)});
  }
  auto found = impl_->find(frame.slot);
  if (!found) return std::unexpected(found.error());
  Impl::Slot& slot = **found;
  if (slot.state != Impl::State::Writer) {
    return std::unexpected(
        Refusal{RefusalCode::SequenceOrder, std::format("slot {} déjà en lecture", frame.slot)});
  }
  if (frame.ready_value % 2 == 0 || frame.ready_value <= slot.released_value) {
    return std::unexpected(
        out_of_range(std::format("slot {} : ready_value {} après {} (impaire et croissante)", frame.slot,
                                 frame.ready_value, slot.released_value)));
  }
  const pl_vulkan_release_params release{.tex = slot.tex,
                                         .layout = kWriterLayout,
                                         .qf = impl_->queue_family,
                                         .semaphore = {.sem = slot.semaphore, .value = frame.ready_value}};
  pl_vulkan_release_ex(impl_->gpu, &release);
  slot.state = Impl::State::Render;
  slot.released_value = frame.ready_value;
  return slot.tex;
}

std::expected<SlotFree, Refusal> OutputRing::release(std::uint32_t index) {
  auto found = impl_->find(index);
  if (!found) return std::unexpected(found.error());
  Impl::Slot& slot = **found;
  if (slot.state != Impl::State::Render) {
    return std::unexpected(Refusal{RefusalCode::SequenceOrder, std::format("slot {} pas en lecture", index)});
  }
  const std::uint64_t value = slot.released_value + 1;
  const pl_vulkan_hold_params hold{.tex = slot.tex,
                                   .layout = kWriterLayout,
                                   .qf = impl_->queue_family,
                                   .semaphore = {.sem = slot.semaphore, .value = value}};
  if (!pl_vulkan_hold_ex(impl_->gpu, &hold)) {
    return std::unexpected(
        Refusal{RefusalCode::RenderFailed, std::format("pl_vulkan_hold_ex (slot {})", index)});
  }
  slot.state = Impl::State::Writer;
  slot.released_value = value;
  return SlotFree{.slot = index, .semaphore_value = value, .generation = impl_->generation};
}

}  // namespace ajp::render
