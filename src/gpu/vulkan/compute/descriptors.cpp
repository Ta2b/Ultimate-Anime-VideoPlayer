#include "gpu/vulkan/compute/descriptors.h"

#include <algorithm>
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
  DescriptorArena arena(&context, limits);
  if (auto ok = arena.open_pool(0, 0); !ok) return std::unexpected(ok.error());
  return arena;
}

Done DescriptorArena::open_pool(std::uint32_t buffers, std::uint32_t images) {
  const DescriptorLimits limits{.sets = limits_.sets,
                                .buffers = std::max(limits_.buffers, buffers),
                                .images = std::max(limits_.images, images)};
  const std::array sizes{
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = limits.buffers},
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = limits.images},
      VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                           .descriptorCount = limits.images}};
  const VkDescriptorPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = limits.sets,
                                        .poolSizeCount = static_cast<std::uint32_t>(sizes.size()),
                                        .pPoolSizes = sizes.data()};
  VkDescriptorPool pool = VK_NULL_HANDLE;
  if (auto ok = check(context_->fn().vkCreateDescriptorPool(context_->device(), &info, nullptr, &pool),
                      "vkCreateDescriptorPool");
      !ok) {
    return ok;
  }
  pools_.push_back(pool);
  current_ = limits;
  sets_ = 0;
  buffers_ = 0;
  images_ = 0;
  return {};
}

DescriptorArena::DescriptorArena(DescriptorArena&& other) noexcept
    : context_(other.context_),
      pools_(std::exchange(other.pools_, {})),
      limits_(other.limits_),
      current_(other.current_),
      sets_(other.sets_),
      buffers_(other.buffers_),
      images_(other.images_),
      total_sets_(other.total_sets_) {}

DescriptorArena& DescriptorArena::operator=(DescriptorArena&& other) noexcept {
  if (this != &other) {
    destroy();
    context_ = other.context_;
    pools_ = std::exchange(other.pools_, {});
    limits_ = other.limits_;
    current_ = other.current_;
    sets_ = other.sets_;
    buffers_ = other.buffers_;
    images_ = other.images_;
    total_sets_ = other.total_sets_;
  }
  return *this;
}

DescriptorArena::~DescriptorArena() {
  destroy();
}

void DescriptorArena::destroy() {
  // Les ensembles disparaissent avec leurs pools.
  for (const VkDescriptorPool pool : pools_)
    context_->fn().vkDestroyDescriptorPool(context_->device(), pool, nullptr);
  pools_.clear();
}

std::expected<VkDescriptorSet, Refusal> DescriptorArena::bind(const Pipeline& pipeline,
                                                              std::span<const BufferView> buffers,
                                                              std::span<const VkImageView> images) {
  const auto buffer_count = static_cast<std::uint32_t>(buffers.size());
  const auto image_count = static_cast<std::uint32_t>(images.size());
  if (pools_.empty() || sets_ + 1 > current_.sets || buffers_ + buffer_count > current_.buffers ||
      images_ + image_count > current_.images) {
    if (pools_.size() >= kMaxPools) {
      return refuse(RefusalCode::GpuMemoryExhausted,
                    std::format("{} : {} pools de descripteurs pleins ({} ensembles)", pipeline.name(),
                                pools_.size(), total_sets_));
    }
    if (auto ok = open_pool(buffer_count, image_count); !ok) return std::unexpected(ok.error());
  }
  const VkDescriptorSetLayout layout = pipeline.set_layout();
  const VkDescriptorSetAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                             .descriptorPool = pools_.back(),
                                             .descriptorSetCount = 1,
                                             .pSetLayouts = &layout};
  VkDescriptorSet set = VK_NULL_HANDLE;
  if (auto ok = check(context_->fn().vkAllocateDescriptorSets(context_->device(), &allocate, &set),
                      std::format("vkAllocateDescriptorSets({})", pipeline.name()));
      !ok) {
    return std::unexpected(ok.error());
  }
  ++sets_;
  ++total_sets_;
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
                      .descriptorType = i < pipeline.images() ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                              : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      .pImageInfo = &image_infos[i]});
  }
  context_->fn().vkUpdateDescriptorSets(context_->device(), static_cast<std::uint32_t>(writes.size()),
                                        writes.data(), 0, nullptr);
  return set;
}

}  // namespace ajp::gpu::vk
