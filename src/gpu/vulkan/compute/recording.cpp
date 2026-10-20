#include "gpu/vulkan/compute/recording.h"

#include <cstring>
#include <format>
#include <limits>
#include <string_view>
#include <utility>

#include <spdlog/logger.h>

namespace ajp::gpu::vk {
namespace {

// Écritures des noyaux et des transferts, avant les lectures et écritures de la commande suivante.
constexpr VkPipelineStageFlags kWorkStages =
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
constexpr VkAccessFlags kWrites = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
constexpr VkAccessFlags kReadsAndWrites = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                          VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

}  // namespace

Recording::~Recording() {
  const DeviceFunctions& fn = context_.fn();
  // Une exécution encore dans le lot de VkContext::enqueue part avant d'être attendue.
  if (auto flushed = context_.flush(); !flushed)
    context_.log().error("fin d'enregistrement : {}", flushed.error().detail);
  if (submitted_ > 0) {
    // Fin de vie : ce n'est pas une attente du flux (host_waits ne la compte pas).
    const VkSemaphore semaphore = runs_.handle();
    const VkSemaphoreWaitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                   .semaphoreCount = 1,
                                   .pSemaphores = &semaphore,
                                   .pValues = &submitted_};
    if (fn.vkWaitSemaphores(context_.device(), &wait, std::numeric_limits<std::uint64_t>::max()) !=
        VK_SUCCESS) {
      context_.log().error("fin d'enregistrement : l'attente de l'exécution {} a échoué", submitted_);
    }
  }
  if (pool_ != VK_NULL_HANDLE) fn.vkDestroyCommandPool(context_.device(), pool_, nullptr);
}

std::expected<Recorder, Refusal> Recorder::begin(VkContext& context, MemoryAllocator& memory,
                                                 DescriptorLimits limits) {
  auto runs = TimelineSemaphore::create(context, 0);
  if (!runs) return std::unexpected(runs.error());
  auto arena = DescriptorArena::create(context, limits);
  if (!arena) return std::unexpected(arena.error());
  std::unique_ptr<Recording> recording(new Recording(context, std::move(*runs), std::move(*arena)));

  const DeviceFunctions& fn = context.fn();
  const VkCommandPoolCreateInfo pool{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                     .queueFamilyIndex = context.queue_family()};
  if (auto ok = check(fn.vkCreateCommandPool(context.device(), &pool, nullptr, &recording->pool_),
                      "vkCreateCommandPool");
      !ok) {
    return std::unexpected(ok.error());
  }
  const VkCommandBufferAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                             .commandPool = recording->pool_,
                                             .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                             .commandBufferCount = 1};
  if (auto ok = check(fn.vkAllocateCommandBuffers(context.device(), &allocate, &recording->command_buffer_),
                      "vkAllocateCommandBuffers");
      !ok) {
    return std::unexpected(ok.error());
  }
  // Rejouable sans réenregistrement, et soumissible alors que l'exécution précédente est en
  // attente : la suivante attend la précédente sur le sémaphore runs.
  const VkCommandBufferBeginInfo info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                      .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT};
  if (auto ok = check(fn.vkBeginCommandBuffer(recording->command_buffer_, &info), "vkBeginCommandBuffer");
      !ok) {
    return std::unexpected(ok.error());
  }
  return Recorder(context, memory, std::move(recording));
}

Done Recorder::open() const {
  if (recording_ == nullptr) return refuse(RefusalCode::InputMismatch, "enregistrement déjà terminé");
  return {};
}

void Recorder::barrier() {
  if (!pending_) return;
  const VkMemoryBarrier memory{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = kWrites, .dstAccessMask = kReadsAndWrites};
  context_->fn().vkCmdPipelineBarrier(recording_->command_buffer_, kWorkStages, kWorkStages, 0, 1, &memory, 0,
                                      nullptr, 0, nullptr);
  pending_ = false;
}

Done Recorder::check_view(const BufferView& view, std::string_view what) const {
  const DeviceCaps& caps = context_->caps();
  if (view.buffer == VK_NULL_HANDLE || view.size == 0) {
    return refuse(RefusalCode::InputMismatch, std::format("{} : vue de tampon vide", what));
  }
  if (caps.min_storage_buffer_alignment > 1 && view.offset % caps.min_storage_buffer_alignment != 0) {
    return refuse(RefusalCode::InputMismatch, std::format("{} : offset {} non multiple de {}", what,
                                                          view.offset, caps.min_storage_buffer_alignment));
  }
  if (view.size > caps.max_storage_buffer_range) {
    return refuse(RefusalCode::InputMismatch, std::format("{} : vue de {} o, l'appareil en lie {} au plus",
                                                          what, view.size, caps.max_storage_buffer_range));
  }
  return {};
}

std::expected<BufferView, Refusal> Recorder::scratch(VkDeviceSize bytes, MemoryUse use) {
  if (auto ok = open(); !ok) return std::unexpected(ok.error());
  auto buffer = Buffer::create(*context_, *memory_, bytes, use);
  if (!buffer) return std::unexpected(buffer.error());
  const BufferView view = buffer->view();
  recording_->owned_.push_back(std::move(*buffer));
  return view;
}

std::expected<BufferView, Refusal> Recorder::constants(std::span<const std::byte> bytes) {
  auto view = scratch(bytes.size(), MemoryUse::Upload);
  if (!view) return view;
  std::memcpy(recording_->owned_.back().mapped(), bytes.data(), bytes.size());
  return view;
}

Done Recorder::dispatch(const Pipeline& pipeline, std::span<const BufferView> buffers,
                        std::span<const VkImageView> images, std::span<const std::byte> push,
                        std::array<std::uint32_t, 3> groups) {
  if (auto ok = open(); !ok) return ok;
  if (buffers.size() != pipeline.buffers() || images.size() != pipeline.images() ||
      push.size() != pipeline.push_bytes()) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : {} tampons, {} images et {} o de push, {}, {} et {} attendus",
                              pipeline.name(), buffers.size(), images.size(), push.size(), pipeline.buffers(),
                              pipeline.images(), pipeline.push_bytes()));
  }
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    if (auto ok = check_view(buffers[i], std::format("{} binding {}", pipeline.name(), i)); !ok) return ok;
  }
  const DeviceCaps& caps = context_->caps();
  for (std::size_t axis = 0; axis < groups.size(); ++axis) {
    if (groups[axis] == 0 || groups[axis] > caps.max_workgroup_count[axis]) {
      return refuse(RefusalCode::FeatureUnavailable,
                    std::format("{} : {} groupes sur l'axe {}, de 1 à {} permis", pipeline.name(),
                                groups[axis], axis, caps.max_workgroup_count[axis]));
    }
  }
  auto set = recording_->arena_.bind(pipeline, buffers, images);
  if (!set) return std::unexpected(set.error());

  barrier();
  const DeviceFunctions& fn = context_->fn();
  const VkCommandBuffer cb = recording_->command_buffer_;
  fn.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
  fn.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1, &*set, 0, nullptr);
  if (!push.empty()) {
    fn.vkCmdPushConstants(cb, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                          static_cast<std::uint32_t>(push.size()), push.data());
  }
  fn.vkCmdDispatch(cb, groups[0], groups[1], groups[2]);
  ++recording_->dispatches_;
  pending_ = true;
  return {};
}

Done Recorder::fill(const BufferView& view, std::uint32_t value) {
  if (auto ok = open(); !ok) return ok;
  if (view.buffer == VK_NULL_HANDLE || view.size == 0 || view.offset % 4 != 0 || view.size % 4 != 0) {
    return refuse(
        RefusalCode::InputMismatch,
        std::format("remplissage de {} o à l'offset {} : multiples de 4 attendus", view.size, view.offset));
  }
  barrier();
  context_->fn().vkCmdFillBuffer(recording_->command_buffer_, view.buffer, view.offset, view.size, value);
  pending_ = true;
  return {};
}

Done Recorder::commands(const std::function<void(VkCommandBuffer)>& record) {
  if (auto ok = open(); !ok) return ok;
  barrier();
  record(recording_->command_buffer_);
  pending_ = true;
  return {};
}

std::expected<std::unique_ptr<Recording>, Refusal> Recorder::finish() {
  if (auto ok = open(); !ok) return std::unexpected(ok.error());
  // Tout ce que l'enregistrement écrit devient visible de l'hôte (après l'attente de son
  // exécution) et des transferts qui suivent sur la file.
  const VkMemoryBarrier memory{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                               .srcAccessMask = kWrites,
                               .dstAccessMask = kReadsAndWrites | VK_ACCESS_HOST_READ_BIT};
  const DeviceFunctions& fn = context_->fn();
  fn.vkCmdPipelineBarrier(recording_->command_buffer_, kWorkStages, kWorkStages | VK_PIPELINE_STAGE_HOST_BIT,
                          0, 1, &memory, 0, nullptr, 0, nullptr);
  if (auto ok = check(fn.vkEndCommandBuffer(recording_->command_buffer_), "vkEndCommandBuffer"); !ok) {
    return std::unexpected(ok.error());
  }
  recording_->finished_ = true;
  return std::move(recording_);
}

}  // namespace ajp::gpu::vk
