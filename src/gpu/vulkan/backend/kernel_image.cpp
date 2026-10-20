// Images décodées en mémoire Vulkan : la VkImage multiplan de FFmpeg lue en place par les
// noyaux (vues d'un plan chacune, quand l'image est échantillonnable à format mutable), ou
// copiée par le GPU dans un tampon de disposition serrée ; dans les deux cas, d'abord son passage
// en GENERAL sur notre file.
#include <array>
#include <initializer_list>

#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"

namespace ajp::gpu::vk {

namespace detail {

const PlaneFormats* plane_formats(kernels::Format format) {
  static constexpr PlaneFormats kNv12{.image = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                                      .luma = VK_FORMAT_R8_UNORM,
                                      .chroma = VK_FORMAT_R8G8_UNORM};
  static constexpr PlaneFormats kP010{.image = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
                                      .luma = VK_FORMAT_R10X6_UNORM_PACK16,
                                      .chroma = VK_FORMAT_R10X6G10X6_UNORM_2PACK16};
  static constexpr PlaneFormats kP016{.image = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM,
                                      .luma = VK_FORMAT_R16_UNORM,
                                      .chroma = VK_FORMAT_R16G16_UNORM};
  switch (format) {
    case kernels::Format::Nv12:
      return &kNv12;
    case kernels::Format::P010:
      return &kP010;
    case kernels::Format::P016:
      return &kP016;
    default:
      return nullptr;
  }
}

}  // namespace detail

namespace {

using detail::image_bits;
using detail::plane_copies;
using detail::PlaneCopy;

std::unexpected<Refusal> refuse_image(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::move(text)});
}

}  // namespace

bool VulkanKernelOps::reads_in_place(const VulkanImageAccess& access, kernels::Format format) {
  if ((access.usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0 ||
      (access.flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) == 0)
    return false;
  const detail::PlaneFormats* planes = detail::plane_formats(format);
  if (planes == nullptr || access.format != static_cast<std::uint32_t>(planes->image)) return false;
  for (const auto& [known, verdict] : plane_formats_) {
    if (known == access.format) return verdict;
  }
  bool ok = true;
  for (const VkFormat plane : {planes->luma, planes->chroma}) {
    VkFormatProperties properties{};
    context_.instance_functions().vkGetPhysicalDeviceFormatProperties(context_.physical_device(), plane,
                                                                      &properties);
    ok = ok && (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
  }
  plane_formats_.emplace_back(access.format, ok);
  return ok;
}

std::expected<YuvPlanes, Refusal> VulkanKernelOps::plane_views_for(VkImage image, kernels::Format format,
                                                                   std::uint32_t width,
                                                                   std::uint32_t height) {
  YuvPlanes planes{.format = format, .width = width, .height = height};
  const std::uint64_t bits = image_bits(image);
  for (const PlaneViews& v : plane_views_) {
    if (v.image == bits && v.pool == pool_generation_) {
      planes.views = v.views;
      return planes;
    }
  }
  const detail::PlaneFormats* formats = detail::plane_formats(format);
  if (formats == nullptr)
    return refuse_image(RefusalCode::FormatUnsupported, "image décodée : plans sans format");
  PlaneViews made{.image = bits, .pool = pool_generation_};
  const std::array<VkImageAspectFlags, 2> aspects{VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT};
  const std::array<VkFormat, 2> plane_formats{formats->luma, formats->chroma};
  for (std::size_t p = 0; p < 2; ++p) {
    const VkImageViewUsageCreateInfo usage{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                           .usage = VK_IMAGE_USAGE_SAMPLED_BIT};
    const VkImageViewCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = &usage,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = plane_formats[p],
        .subresourceRange = {.aspectMask = aspects[p], .levelCount = 1, .layerCount = 1}};
    if (auto ok = check(context_.fn().vkCreateImageView(context_.device(), &info, nullptr, &made.views[p]),
                        "vkCreateImageView(plan d'une image décodée)");
        !ok) {
      if (p == 1) context_.fn().vkDestroyImageView(context_.device(), made.views[0], nullptr);
      return std::unexpected(ok.error());
    }
  }
  plane_views_.push_back(made);
  planes.views = made.views;
  return planes;
}

void VulkanKernelOps::release_plane_views() {
  if (plane_views_.empty()) return;
  // Un GPU bloqué garde les vues : elles fuient plutôt que d'être rendues sous une lecture.
  if (!context_.settle(queue_.last())) {
    plane_views_.clear();
    return;
  }
  for (const PlaneViews& v : plane_views_) {
    for (const VkImageView view : v.views) context_.fn().vkDestroyImageView(context_.device(), view, nullptr);
  }
  plane_views_.clear();
}

void VulkanKernelOps::enter_image(VkCommandBuffer cb, const Source& src) const {
  if (!src.access) return;  // image hôte ou mémo : rien à préparer
  const VulkanImageState& state = src.access->state;
  const bool owned =
      state.queue_family != VK_QUEUE_FAMILY_IGNORED && state.queue_family != context_.queue_family();
  const VkImageMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = static_cast<VkImageLayout>(state.layout),
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = owned ? state.queue_family : VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = owned ? context_.queue_family() : VK_QUEUE_FAMILY_IGNORED,
      .image = src.image,
      .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1}};
  context_.fn().vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     0, nullptr, 0, nullptr, 1, &barrier);
}

void VulkanKernelOps::copy_image(VkCommandBuffer cb, const Source& src) const {
  if (!src.access) return;  // image hôte : rien à copier
  enter_image(cb, src);
  std::array<PlaneCopy, 3> planes{};
  const std::uint32_t count = plane_copies(src.view, planes);
  constexpr std::array<VkImageAspectFlagBits, 3> kAspects{
      VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT, VK_IMAGE_ASPECT_PLANE_2_BIT};
  std::array<VkBufferImageCopy, 3> regions{};
  for (std::uint32_t p = 0; p < count; ++p) {
    regions[p] = {
        .bufferOffset = src.view.buffer.offset + planes[p].dst.offset,
        .bufferRowLength = planes[p].dst.pitch / planes[p].texel,
        .imageSubresource = {.aspectMask = static_cast<VkImageAspectFlags>(kAspects[p]), .layerCount = 1},
        .imageExtent = {.width = planes[p].width, .height = planes[p].height, .depth = 1}};
  }
  context_.fn().vkCmdCopyImageToBuffer(cb, src.image, VK_IMAGE_LAYOUT_GENERAL, src.view.buffer.buffer, count,
                                       regions.data());
}

Done VulkanKernelOps::prepare_image(Recorder& r, const Source& src) const {
  if (!src.access) return {};
  return r.commands([&](VkCommandBuffer cb) {
    if (src.planes) {
      enter_image(cb, src);
    } else {
      copy_image(cb, src);
    }
  });
}

}  // namespace ajp::gpu::vk
