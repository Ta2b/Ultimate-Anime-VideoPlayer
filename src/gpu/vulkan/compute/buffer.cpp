#include "gpu/vulkan/compute/buffer.h"

#include <cstring>
#include <format>
#include <string_view>
#include <utility>

namespace ajp::gpu::vk {
namespace {

constexpr VkBufferUsageFlags kUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

Done check_range(const Buffer& buffer, VkDeviceSize offset, VkDeviceSize size, std::string_view what) {
  if (size == 0 || offset > buffer.size() || size > buffer.size() - offset) {
    return refuse(RefusalCode::InputMismatch, std::format("{} : {} o à l'offset {} d'un tampon de {} o", what,
                                                          size, offset, buffer.size()));
  }
  return {};
}

// Barrière globale : tout ce qui précède, dans l'ordre de soumission, avant la copie ; la copie
// avant ce qui suit (les noyaux, ou l'hôte).
void barrier(const DeviceFunctions& fn, VkCommandBuffer cb, VkPipelineStageFlags src,
             VkAccessFlags src_access, VkPipelineStageFlags dst, VkAccessFlags dst_access) {
  const VkMemoryBarrier memory{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = src_access, .dstAccessMask = dst_access};
  fn.vkCmdPipelineBarrier(cb, src, dst, 0, 1, &memory, 0, nullptr, 0, nullptr);
}

}  // namespace

std::expected<Buffer, Refusal> Buffer::create(const VkContext& context, MemoryAllocator& memory,
                                              VkDeviceSize size, MemoryUse use) {
  if (size == 0) return refuse(RefusalCode::FieldOutOfRange, "tampon de 0 o");
  const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = size,
                                .usage = kUsage,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  const DeviceFunctions& fn = context.fn();
  VkBuffer buffer = VK_NULL_HANDLE;
  if (auto ok = check(fn.vkCreateBuffer(context.device(), &info, nullptr, &buffer), "vkCreateBuffer"); !ok) {
    return std::unexpected(ok.error());
  }
  VkMemoryRequirements requirements{};
  fn.vkGetBufferMemoryRequirements(context.device(), buffer, &requirements);
  auto allocation = memory.allocate(requirements, use, Tiling::Linear);
  if (!allocation) {
    fn.vkDestroyBuffer(context.device(), buffer, nullptr);
    return std::unexpected(allocation.error());
  }
  if (auto ok =
          check(fn.vkBindBufferMemory(context.device(), buffer, allocation->memory(), allocation->offset()),
                "vkBindBufferMemory");
      !ok) {
    fn.vkDestroyBuffer(context.device(), buffer, nullptr);
    return std::unexpected(ok.error());
  }
  return Buffer(&context, buffer, std::move(*allocation), size, use);
}

Buffer::Buffer(Buffer&& other) noexcept
    : context_(other.context_),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      allocation_(std::move(other.allocation_)),
      size_(other.size_),
      use_(other.use_) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    destroy();
    context_ = other.context_;
    buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
    allocation_ = std::move(other.allocation_);
    size_ = other.size_;
    use_ = other.use_;
  }
  return *this;
}

Buffer::~Buffer() {
  destroy();
}

void Buffer::destroy() {
  if (buffer_ != VK_NULL_HANDLE) context_->fn().vkDestroyBuffer(context_->device(), buffer_, nullptr);
  buffer_ = VK_NULL_HANDLE;
  allocation_ = Allocation();
}

std::expected<BufferView, Refusal> Buffer::view(VkDeviceSize offset, VkDeviceSize size) const {
  if (auto ok = check_range(*this, offset, size, "vue de tampon"); !ok) return std::unexpected(ok.error());
  return BufferView{.buffer = buffer_, .offset = offset, .size = size};
}

Done upload(VkContext& context, MemoryAllocator& memory, const Buffer& buffer, VkDeviceSize offset,
            std::span<const std::byte> bytes) {
  if (auto ok = check_range(buffer, offset, bytes.size(), "envoi hôte"); !ok) return ok;
  if (buffer.mapped() != nullptr) {
    std::memcpy(buffer.mapped() + offset, bytes.data(), bytes.size());
    return {};
  }
  auto staging = Buffer::create(context, memory, bytes.size(), MemoryUse::Upload);
  if (!staging) return std::unexpected(staging.error());
  std::memcpy(staging->mapped(), bytes.data(), bytes.size());
  const DeviceFunctions& fn = context.fn();
  return context.run_once([&](VkCommandBuffer cb) {
    barrier(fn, cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT);
    const VkBufferCopy region{.srcOffset = 0, .dstOffset = offset, .size = bytes.size()};
    fn.vkCmdCopyBuffer(cb, staging->handle(), buffer.handle(), 1, &region);
    barrier(fn, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  });
}

Done download(VkContext& context, MemoryAllocator& memory, const Buffer& buffer, VkDeviceSize offset,
              std::span<std::byte> out) {
  if (auto ok = check_range(buffer, offset, out.size(), "lecture hôte"); !ok) return ok;
  if (buffer.mapped() != nullptr) {
    std::memcpy(out.data(), buffer.mapped() + offset, out.size());
    return {};
  }
  auto staging = Buffer::create(context, memory, out.size(), MemoryUse::Readback);
  if (!staging) return std::unexpected(staging.error());
  const DeviceFunctions& fn = context.fn();
  auto copied = context.run_once([&](VkCommandBuffer cb) {
    barrier(fn, cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    const VkBufferCopy region{.srcOffset = offset, .dstOffset = 0, .size = out.size()};
    fn.vkCmdCopyBuffer(cb, buffer.handle(), staging->handle(), 1, &region);
    barrier(fn, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_HOST_READ_BIT);
  });
  if (!copied) return copied;
  std::memcpy(out.data(), staging->mapped(), out.size());
  return {};
}

}  // namespace ajp::gpu::vk
