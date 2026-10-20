// Écriture d'un slot du ring (core/ring.h) : la VkImage du slot, sur le VkDevice partagé avec
// render, en VK_IMAGE_LAYOUT_GENERAL (kRingLayout), écrite par ring_store.comp entre l'attente
// de la valeur de libération et le signal de ready_value sur le sémaphore timeline du slot. Une
// vue par slot et par génération ; la première écriture d'une génération plus récente retire
// les vues des anciennes, détruites une fois leurs écritures achevées côté GPU.
#include <format>

#include "ajp_vk_backend_spirv.h"
#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/compute/refuse.h"

namespace ajp::gpu::vk {
namespace {

constexpr VkFormat kSlotFormat = VK_FORMAT_R16G16B16A16_UNORM;

std::unexpected<Refusal> refuse_ring(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

VkImage slot_image(std::uint64_t handle) {
  return detail::to_image(static_cast<std::uintptr_t>(handle));
}

}  // namespace

std::expected<VkImageView, Refusal> VulkanKernelOps::ring_view(const RingSlotDesc& slot) {
  if (slot.generation < ring_generation_) {
    return refuse_ring(RefusalCode::FieldOutOfRange,
                       std::format("ring : génération {} après la {}", slot.generation, ring_generation_));
  }
  ring_generation_ = slot.generation;
  for (const RingView& v : ring_views_) {
    if (v.generation == slot.generation && v.index == slot.index && v.image == slot.image) return v.view;
  }
  VkFormatProperties properties{};
  context_.instance_functions().vkGetPhysicalDeviceFormatProperties(context_.physical_device(), kSlotFormat,
                                                                    &properties);
  if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) == 0) {
    return refuse_ring(RefusalCode::FeatureUnavailable, "ring : R16G16B16A16_UNORM sans écriture en stockage");
  }
  const VkImageViewCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = slot_image(slot.image),
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = kSlotFormat,
      .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1}};
  VkImageView view = VK_NULL_HANDLE;
  if (auto ok = check(context_.fn().vkCreateImageView(context_.device(), &info, nullptr, &view),
                      "vkCreateImageView(slot du ring)");
      !ok) {
    return std::unexpected(ok.error());
  }
  ring_views_.push_back({.generation = slot.generation, .index = slot.index, .image = slot.image, .view = view});
  return view;
}

std::expected<void, Refusal> VulkanKernelOps::write_ring(const Tensor& rgb, const RingSlotDesc& slot,
                                                         const RingWrite& sync) {
  auto src = detail::tensor_view(rgb, 3, "ring : tenseur RGB");
  if (!src) return std::unexpected(src.error());
  if (slot.format != RingFormat::Rgba16Unorm) {
    return refuse_ring(RefusalCode::FormatUnsupported, "ring : slot Yuv444p16, seul Rgba16Unorm s'écrit");
  }
  if (std::cmp_not_equal(slot.width, src->width) || std::cmp_not_equal(slot.height, src->height)) {
    return refuse_ring(RefusalCode::ShapeInvalid, std::format("ring : slot {}x{}, tenseur {}x{}", slot.width,
                                                              slot.height, src->width, src->height));
  }
  if (slot.image == 0 || slot.semaphore == 0) {
    return refuse_ring(RefusalCode::FeatureUnavailable,
                       "ring : slot sans VkImage ni sémaphore de l'appareil partagé (import par handle exporté non pris "
                       "en charge)");
  }
  if (sync.signal_value % 2 != 1 || sync.signal_value <= sync.wait_value) {
    return refuse_ring(RefusalCode::FieldOutOfRange,
                       std::format("ring : signal {} après {}, impair et croissant attendu", sync.signal_value,
                                   sync.wait_value));
  }
  auto view = ring_view(slot);
  if (!view) return std::unexpected(view.error());
  const std::string_view shader = src->precision == Precision::F16 ? "ring_store_f16" : "ring_store";
  const std::array<std::uint32_t, 1> constants{src->precision == Precision::F16 ? 1U : 0U};
  auto pipeline = pipelines_.get({.shader = shader,
                                  .constants = constants,
                                  .buffers = 1,
                                  .images = 1,
                                  .push_bytes = 8,
                                  .spirv = net::spirv::words(shader)});
  if (!pipeline) return std::unexpected(pipeline.error());
  OpKey key;
  key.add(40);
  detail::add_view(key, *src);
  key.add(detail::to_bits(*view));
  auto rec = recording(key, [&](Recorder& r) -> Done {
    const std::array buffers{src->buffer};
    const std::array images{*view};
    const std::array<std::uint32_t, 2> push{src->width, src->height};
    return r.dispatch(**pipeline, buffers, images, std::as_bytes(std::span(push)),
                      {(src->width + 15) / 16, (src->height + 15) / 16, 1});
  });
  if (!rec) return std::unexpected(rec.error());
  const VkSemaphore semaphore = detail::to_semaphore(static_cast<std::uintptr_t>(slot.semaphore));
  const std::array<TimelinePoint, 1> wait{TimelinePoint{.semaphore = semaphore, .value = sync.wait_value}};
  const std::array<TimelinePoint, 1> signal{TimelinePoint{.semaphore = semaphore, .value = sync.signal_value}};
  const std::span<const TimelinePoint> waits = sync.wait_value > 0 ? std::span<const TimelinePoint>(wait)
                                                                    : std::span<const TimelinePoint>();
  if (auto ok = submit(**rec, waits, signal); !ok) return ok;
  for (RingView& v : ring_views_) {
    if (v.view == *view) v.last_write = queue_.last().value;
  }
  return {};
}

}  // namespace ajp::gpu::vk
