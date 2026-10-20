#include "gpu/vulkan/compute/descriptors.h"

#include <array>
#include <cstddef>
#include <format>
#include <utility>
#include <vector>

#include "gpu/vulkan/compute/refuse.h"

namespace ajp::gpu::vk {

std::expected<DescriptorArena, Refusal> DescriptorArena::create(const VkContext& context,
                                                                DescriptorLimits limits) {
  if (limits.sets == 0 || limits.buffers == 0 || limits.images == 0) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("pool de descripteurs de {} ensembles, {} tampons et {} images", limits.sets,
                              limits.buffers, limits.images));
  }
  const std::array sizes{
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = limits.buffers},
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = limits.images}};
  const VkDescriptorPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = limits.sets,
                                        .poolSizeCount = static_cast<std::uint32_t>(sizes.size()),
                                        .pPoolSizes = sizes.data()};
  VkDescriptorPool pool = VK_NULL_HANDLE;
  if (auto ok = check(context.fn().vkCreateDescriptorPool(context.device(), &info, nullptr, &pool),
                      "vkCreateDescriptorPool");
      !ok) {
    return std::unexpected(ok.error());
  }
  return DescriptorArena(&context, pool, limits);
}

DescriptorArena::DescriptorArena(DescriptorArena&& other) noexcept
    : context_(other.context_),
      pool_(std::exchange(other.pool_, VK_NULL_HANDLE)),
      limits_(other.limits_),
      sets_(other.sets_),
      buffers_(other.buffers_),
      images_(other.images_) {}

DescriptorArena& DescriptorArena::operator=(DescriptorArena&& other) noexcept {
  if (this != &other) {
    destroy();
    context_ = other.context_;
    pool_ = std::exchange(other.pool_, VK_NULL_HANDLE);
    limits_ = other.limits_;
    sets_ = other.sets_;
    buffers_ = other.buffers_;
    images_ = other.images_;
  }
  return *this;
}

DescriptorArena::~DescriptorArena() {
  destroy();
}

void DescriptorArena::destroy() {
  // Les ensembles disparaissent avec leur pool.
  if (pool_ != VK_NULL_HANDLE) context_->fn().vkDestroyDescriptorPool(context_->device(), pool_, nullptr);
  pool_ = VK_NULL_HANDLE;
}

std::expected<VkDescriptorSet, Refusal> DescriptorArena::bind(const Pipeline& pipeline,
                                                              std::span<const BufferView> buffers,
                                                              std::span<const VkImageView> images) {
  const auto buffer_count = static_cast<std::uint32_t>(buffers.size());
  const auto image_count = static_cast<std::uint32_t>(images.size());
  if (sets_ + 1 > limits_.sets || buffers_ + buffer_count > limits_.buffers ||
      images_ + image_count > limits_.images) {
    return refuse(
        RefusalCode::GpuMemoryExhausted,
        std::format("{} : pool de descripteurs plein ({} ensembles sur {}, {} tampons sur {}, {} images "
                    "sur {})",
                    pipeline.name(), sets_, limits_.sets, buffers_, limits_.buffers, images_,
                    limits_.images));
  }
  const VkDescriptorSetLayout layout = pipeline.set_layout();
  const VkDescriptorSetAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                             .descriptorPool = pool_,
                                             .descriptorSetCount = 1,
                                             .pSetLayouts = &layout};
  VkDescriptorSet set = VK_NULL_HANDLE;
  if (auto ok = check(context_->fn().vkAllocateDescriptorSets(context_->device(), &allocate, &set),
                      std::format("vkAllocateDescriptorSets({})", pipeline.name()));
      !ok) {
    return std::unexpected(ok.error());
  }
  ++sets_;
  buffers_ += buffer_count;
  images_ += image_count;

  std::vector<VkDescriptorBufferInfo> buffer_infos;
  buffer_infos.reserve(buffers.size());
  for (const BufferView& view : buffers) {
    buffer_infos.push_back({.buffer = view.buffer, .offset = view.offset, .range = view.size});
  }
  std::vector<VkDescriptorImageInfo> image_infos;
  image_infos.reserve(images.size());
  for (const VkImageView view : images) {
    image_infos.push_back(
        {.sampler = VK_NULL_HANDLE, .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL});
  }
  std::vector<VkWriteDescriptorSet> writes;
  writes.reserve(std::size_t{buffer_count} + image_count);
  for (std::uint32_t i = 0; i < buffer_count; ++i) {
    writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                      .dstSet = set,
                      .dstBinding = i,
                      .descriptorCount = 1,
                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                      .pBufferInfo = &buffer_infos[i]});
  }
  for (std::uint32_t i = 0; i < image_count; ++i) {
    writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                      .dstSet = set,
                      .dstBinding = buffer_count + i,
                      .descriptorCount = 1,
                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                      .pImageInfo = &image_infos[i]});
  }
  context_->fn().vkUpdateDescriptorSets(context_->device(), static_cast<std::uint32_t>(writes.size()),
                                        writes.data(), 0, nullptr);
  return set;
}

}  // namespace ajp::gpu::vk
