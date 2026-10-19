#include "gpu/vulkan/compute/image.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <format>
#include <utility>

#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/refuse.h"

namespace ajp::gpu::vk {
namespace {

constexpr VkImageSubresourceRange kColor{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                         .baseMipLevel = 0,
                                         .levelCount = 1,
                                         .baseArrayLayer = 0,
                                         .layerCount = 1};

Done check_format(const VkContext& context, VkFormat format, VkFormatFeatureFlags needed, const char* what) {
  VkFormatProperties properties{};
  context.instance_functions().vkGetPhysicalDeviceFormatProperties(context.physical_device(), format,
                                                                   &properties);
  if ((properties.optimalTilingFeatures & needed) != needed) {
    return refuse(RefusalCode::FeatureUnavailable,
                  std::format("{} : {} sans les fonctions {:#x}", context.caps().device_name, what, needed));
  }
  return {};
}

std::expected<VkImageView, Refusal> make_view(const VkContext& context, VkImage image, VkFormat format,
                                              VkImageUsageFlags usage) {
  const VkImageViewUsageCreateInfo restricted{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                              .usage = usage};
  const VkImageViewCreateInfo info{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                   .pNext = &restricted,
                                   .image = image,
                                   .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                   .format = format,
                                   .subresourceRange = kColor};
  VkImageView view = VK_NULL_HANDLE;
  if (auto ok =
          check(context.fn().vkCreateImageView(context.device(), &info, nullptr, &view), "vkCreateImageView");
      !ok) {
    return std::unexpected(ok.error());
  }
  return view;
}

}  // namespace

std::expected<StorageImage, Refusal> StorageImage::create(VkContext& context, MemoryAllocator& memory,
                                                          std::uint32_t width, std::uint32_t height) {
  if (&memory.context() != &context) {
    return refuse(RefusalCode::InputMismatch, "image RGBA16 : allocateur d'un autre contexte Vulkan");
  }
  const std::uint32_t limit = context.caps().max_image_dimension_2d;
  if (width == 0 || height == 0 || width > limit || height > limit) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("image RGBA16 {}x{} hors de [1, {}]", width, height, limit));
  }
  if (auto ok =
          check_format(context, kStorageFormat, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, "R16G16B16A16_UINT");
      !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = check_format(context, kFormat, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT, "R16G16B16A16_UNORM");
      !ok) {
    return std::unexpected(ok.error());
  }
  const DeviceFunctions& fn = context.fn();
  const VkDevice device = context.device();
  StorageImage result(&context);
  result.width_ = width;
  result.height_ = height;

  // Le stockage n'est demandé qu'au format UINT : EXTENDED_USAGE permet un usage que le format
  // de création (UNORM) n'a pas, tant qu'une vue compatible l'a.
  const std::array formats{kFormat, kStorageFormat};
  const VkImageFormatListCreateInfo list{.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
                                         .viewFormatCount = static_cast<std::uint32_t>(formats.size()),
                                         .pViewFormats = formats.data()};
  const VkImageCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &list,
      .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = kFormat,
      .extent = {.width = width, .height = height, .depth = 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
  if (auto ok = check(fn.vkCreateImage(device, &info, nullptr, &result.image_), "vkCreateImage"); !ok) {
    return std::unexpected(ok.error());
  }
  VkMemoryRequirements requirements{};
  fn.vkGetImageMemoryRequirements(device, result.image_, &requirements);
  auto allocation = memory.allocate(requirements, MemoryUse::Device, Tiling::Optimal);
  if (!allocation) return std::unexpected(allocation.error());
  result.allocation_ = std::move(*allocation);
  if (auto ok = check(fn.vkBindImageMemory(device, result.image_, result.allocation_.memory(),
                                           result.allocation_.offset()),
                      "vkBindImageMemory");
      !ok) {
    return std::unexpected(ok.error());
  }
  auto storage = make_view(context, result.image_, kStorageFormat, VK_IMAGE_USAGE_STORAGE_BIT);
  if (!storage) return std::unexpected(storage.error());
  result.storage_view_ = *storage;
  auto sampled = make_view(context, result.image_, kFormat, VK_IMAGE_USAGE_SAMPLED_BIT);
  if (!sampled) return std::unexpected(sampled.error());
  result.sampled_view_ = *sampled;

  // Passage en GENERAL une fois pour toutes ; le contenu initial est indéfini.
  const VkImage image = result.image_;
  auto ready = context.run_once([&fn, image](VkCommandBuffer cb) {
    const VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = kColor};
    fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                            nullptr, 0, nullptr, 1, &barrier);
  });
  if (!ready) return std::unexpected(ready.error());
  return result;
}

StorageImage::StorageImage(StorageImage&& other) noexcept
    : context_(other.context_),
      image_(std::exchange(other.image_, VK_NULL_HANDLE)),
      allocation_(std::move(other.allocation_)),
      storage_view_(std::exchange(other.storage_view_, VK_NULL_HANDLE)),
      sampled_view_(std::exchange(other.sampled_view_, VK_NULL_HANDLE)),
      width_(other.width_),
      height_(other.height_) {}

StorageImage& StorageImage::operator=(StorageImage&& other) noexcept {
  if (this != &other) {
    destroy();
    context_ = other.context_;
    image_ = std::exchange(other.image_, VK_NULL_HANDLE);
    allocation_ = std::move(other.allocation_);
    storage_view_ = std::exchange(other.storage_view_, VK_NULL_HANDLE);
    sampled_view_ = std::exchange(other.sampled_view_, VK_NULL_HANDLE);
    width_ = other.width_;
    height_ = other.height_;
  }
  return *this;
}

StorageImage::~StorageImage() {
  destroy();
}

void StorageImage::destroy() {
  if (context_ == nullptr) return;
  const DeviceFunctions& fn = context_->fn();
  if (storage_view_ != VK_NULL_HANDLE) fn.vkDestroyImageView(context_->device(), storage_view_, nullptr);
  if (sampled_view_ != VK_NULL_HANDLE) fn.vkDestroyImageView(context_->device(), sampled_view_, nullptr);
  if (image_ != VK_NULL_HANDLE) fn.vkDestroyImage(context_->device(), image_, nullptr);
  storage_view_ = VK_NULL_HANDLE;
  sampled_view_ = VK_NULL_HANDLE;
  image_ = VK_NULL_HANDLE;
  allocation_ = Allocation();
}

std::expected<std::vector<std::uint16_t>, Refusal> read_rgba16(VkContext& context, MemoryAllocator& memory,
                                                               const StorageImage& image) {
  if (image.context() != &context || image.image() == VK_NULL_HANDLE) {
    return refuse(RefusalCode::InputMismatch, "lecture RGBA16 : image vide ou d'un autre contexte Vulkan");
  }
  const std::size_t samples = std::size_t{image.width()} * image.height() * 4;
  auto staging = Buffer::create(context, memory, samples * sizeof(std::uint16_t), MemoryUse::Readback);
  if (!staging) return std::unexpected(staging.error());
  const DeviceFunctions& fn = context.fn();
  auto copied = context.run_once([&](VkCommandBuffer cb) {
    const VkMemoryBarrier before{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                 .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                                 .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
    fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                            &before, 0, nullptr, 0, nullptr);
    const VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                             .mipLevel = 0,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
        .imageOffset = {.x = 0, .y = 0, .z = 0},
        .imageExtent = {.width = image.width(), .height = image.height(), .depth = 1}};
    fn.vkCmdCopyImageToBuffer(cb, image.image(), VK_IMAGE_LAYOUT_GENERAL, staging->handle(), 1, &region);
    const VkMemoryBarrier after{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0,
                            nullptr, 0, nullptr);
  });
  if (!copied) return std::unexpected(copied.error());
  std::vector<std::uint16_t> codes(samples);
  std::memcpy(codes.data(), staging->mapped(), samples * sizeof(std::uint16_t));
  return codes;
}

}  // namespace ajp::gpu::vk
