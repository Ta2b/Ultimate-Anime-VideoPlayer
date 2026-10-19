#include "gpu_timer.h"

#include <array>
#include <cstdio>
#include <format>
#include <utility>
#include <vector>

namespace ajp::bench {

namespace {

using gpu::vk::Done;
using gpu::vk::refuse;

template <class Pfn>
Done load(const gpu::vk::VkContext& context, Pfn& slot, const char* name) {
  const PFN_vkVoidFunction address = context.instance_functions().vkGetDeviceProcAddr(context.device(), name);
  if (address == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("vkGetDeviceProcAddr({}) : fonction absente", name));
  }
  slot = reinterpret_cast<Pfn>(address);
  return {};
}

Done load_queries(const gpu::vk::VkContext& context, QueryFunctions& fn) {
  if (auto ok = load(context, fn.create_query_pool, "vkCreateQueryPool"); !ok) return ok;
  if (auto ok = load(context, fn.destroy_query_pool, "vkDestroyQueryPool"); !ok) return ok;
  if (auto ok = load(context, fn.cmd_reset_query_pool, "vkCmdResetQueryPool"); !ok) return ok;
  if (auto ok = load(context, fn.cmd_write_timestamp, "vkCmdWriteTimestamp"); !ok) return ok;
  return load(context, fn.get_query_pool_results, "vkGetQueryPoolResults");
}

}  // namespace

std::expected<std::unique_ptr<GpuTimer>, Refusal> GpuTimer::create(gpu::vk::VkContext& context,
                                                                   gpu::vk::MemoryAllocator& memory,
                                                                   VkDeviceSize flush_bytes) {
  if (!context.owns_device()) {
    return refuse(RefusalCode::DeviceUnavailable,
                  "chronométrage : contexte adopté, sa file n'est pas au banc");
  }
  const gpu::vk::InstanceFunctions& inst = context.instance_functions();
  std::uint32_t family_count = 0;
  inst.vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device(), &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  inst.vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device(), &family_count, families.data());
  if (context.queue_family() >= family_count) {
    return refuse(RefusalCode::DeviceUnavailable,
                  std::format("chronométrage : famille de file {} absente", context.queue_family()));
  }
  const std::uint32_t valid_bits = families[context.queue_family()].timestampValidBits;
  if (valid_bits == 0) {
    return refuse(RefusalCode::FeatureUnavailable, "chronométrage : la file compute n'a pas de timestamps");
  }
  VkPhysicalDeviceProperties properties{};
  inst.vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
  if (!(properties.limits.timestampPeriod > 0.0F)) {
    return refuse(RefusalCode::FeatureUnavailable, std::format("chronométrage : période de timestamp {} ns",
                                                               properties.limits.timestampPeriod));
  }

  auto done = gpu::vk::TimelineSemaphore::create(context, 0);
  if (!done) return std::unexpected(done.error());
  std::unique_ptr<GpuTimer> timer(new GpuTimer(context, std::move(*done)));
  timer->valid_bits_ = valid_bits;
  timer->period_ns_ = static_cast<double>(properties.limits.timestampPeriod);
  if (auto ok = load_queries(context, timer->query_); !ok) return std::unexpected(ok.error());

  if (flush_bytes > 0) {
    auto flush = gpu::vk::Buffer::create(context, memory, flush_bytes, gpu::vk::MemoryUse::Device);
    if (!flush) return std::unexpected(flush.error());
    timer->flush_.emplace(std::move(*flush));
  }

  const VkDevice device = context.device();
  const VkQueryPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                        .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                        .queryCount = 2};
  if (auto ok = gpu::vk::check(timer->query_.create_query_pool(device, &pool_info, nullptr, &timer->pool_),
                               "vkCreateQueryPool");
      !ok) {
    return std::unexpected(ok.error());
  }
  const VkCommandPoolCreateInfo command_pool_info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                                  .queueFamilyIndex = context.queue_family()};
  if (auto ok = gpu::vk::check(
          context.fn().vkCreateCommandPool(device, &command_pool_info, nullptr, &timer->command_pool_),
          "vkCreateCommandPool");
      !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = timer->record_brackets(); !ok) return std::unexpected(ok.error());
  return timer;
}

GpuTimer::~GpuTimer() {
  // Rien ne se libère tant qu'une soumission peut encore écrire le pool ou le tampon.
  if (signaled_ > 0) {
    if (auto ok = context_.wait_on_host(done_.at(signaled_)); !ok) {
      std::fputs(std::format("chronométrage : attente finale refusée ({})\n", ok.error().detail).c_str(),
                 stderr);
    }
  }
  const VkDevice device = context_.device();
  if (pool_ != VK_NULL_HANDLE) query_.destroy_query_pool(device, pool_, nullptr);
  if (command_pool_ != VK_NULL_HANDLE) context_.fn().vkDestroyCommandPool(device, command_pool_, nullptr);
}

Done GpuTimer::record_brackets() {
  const gpu::vk::DeviceFunctions& fn = context_.fn();
  std::array<VkCommandBuffer, 2> buffers{};
  const VkCommandBufferAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                             .commandPool = command_pool_,
                                             .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                             .commandBufferCount = 2};
  if (auto ok = gpu::vk::check(fn.vkAllocateCommandBuffers(context_.device(), &allocate, buffers.data()),
                               "vkAllocateCommandBuffers");
      !ok) {
    return ok;
  }
  open_ = buffers[0];
  close_ = buffers[1];
  // Rejouables : ni ONE_TIME_SUBMIT, ni SIMULTANEOUS_USE (une exécution à la fois).
  const VkCommandBufferBeginInfo begin{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

  if (auto ok = gpu::vk::check(fn.vkBeginCommandBuffer(open_, &begin), "vkBeginCommandBuffer (ouverture)");
      !ok)
    return ok;
  if (flush_) fn.vkCmdFillBuffer(open_, flush_->handle(), 0, VK_WHOLE_SIZE, 0x5A5A5A5AU);
  query_.cmd_reset_query_pool(open_, pool_, 0, 2);
  // BOTTOM_OF_PIPE : écrit quand tout ce qui précède, vidage compris, est achevé.
  query_.cmd_write_timestamp(open_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_, 0);
  // Dépendance d'exécution vers tout ce qui suit dans l'ordre de soumission, enregistrement
  // compris : il ne commence qu'après le timestamp 0, sans sémaphore à attendre entre les deux
  // (une attente de sémaphore laisserait le pilote passer la main à un autre contexte pendant
  // la mesure).
  fn.vkCmdPipelineBarrier(open_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                          nullptr, 0, nullptr, 0, nullptr);
  if (auto ok = gpu::vk::check(fn.vkEndCommandBuffer(open_), "vkEndCommandBuffer (ouverture)"); !ok)
    return ok;

  if (auto ok = gpu::vk::check(fn.vkBeginCommandBuffer(close_, &begin), "vkBeginCommandBuffer (clôture)");
      !ok)
    return ok;
  query_.cmd_write_timestamp(close_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_, 1);
  return gpu::vk::check(fn.vkEndCommandBuffer(close_), "vkEndCommandBuffer (clôture)");
}

Done GpuTimer::submit(VkCommandBuffer command_buffer, std::optional<std::uint64_t> wait_value,
                      std::optional<std::uint64_t> signal_value) {
  const VkSemaphore semaphore = done_.handle();
  const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  const std::uint64_t wait = wait_value.value_or(0);
  const std::uint64_t signal = signal_value.value_or(0);
  const std::uint32_t waits = wait_value ? 1U : 0U;
  const std::uint32_t signals = signal_value ? 1U : 0U;
  const VkTimelineSemaphoreSubmitInfo timeline{.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                               .waitSemaphoreValueCount = waits,
                                               .pWaitSemaphoreValues = &wait,
                                               .signalSemaphoreValueCount = signals,
                                               .pSignalSemaphoreValues = &signal};
  const VkSubmitInfo info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .pNext = &timeline,
                          .waitSemaphoreCount = waits,
                          .pWaitSemaphores = &semaphore,
                          .pWaitDstStageMask = &stage,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &command_buffer,
                          .signalSemaphoreCount = signals,
                          .pSignalSemaphores = &semaphore};
  if (auto ok = gpu::vk::check(context_.fn().vkQueueSubmit(context_.queue(), 1, &info, VK_NULL_HANDLE),
                               "vkQueueSubmit (chronométrage)");
      !ok) {
    return ok;
  }
  if (signal_value) signaled_ = signal;
  return {};
}

std::expected<double, Refusal> GpuTimer::time(gpu::vk::Recording& recording) {
  // Trois soumissions retenues par une porte que l'hôte n'ouvre qu'une fois toutes en file : ni
  // le temps de soumission de l'hôte ni une attente de sémaphore ne tombent entre les deux
  // timestamps (l'enregistrement attend aussi la porte, ouverte bien avant qu'il ne commence).
  const std::uint64_t gate = signaled_ + 1;
  if (auto ok = submit(open_, gate, std::nullopt); !ok) return std::unexpected(ok.error());
  const std::array waits{done_.at(gate)};
  Done queued = context_.submit(recording, waits);
  if (queued) queued = submit(close_, std::nullopt, gate + 1);
  // La porte s'ouvre même après un refus : l'ouverture déjà en file ne doit pas rester bloquée.
  if (auto ok = done_.signal(gate); !ok) return std::unexpected(ok.error());
  if (!queued) {
    signaled_ = gate;
    return std::unexpected(queued.error());
  }
  if (auto ok = context_.wait_on_host(done_.at(signaled_)); !ok) return std::unexpected(ok.error());

  std::array<std::uint64_t, 2> ticks{};
  if (auto ok = gpu::vk::check(query_.get_query_pool_results(
                                   context_.device(), pool_, 0, 2, sizeof(ticks), ticks.data(),
                                   sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                               "vkGetQueryPoolResults");
      !ok) {
    return std::unexpected(ok.error());
  }
  // Compteur de valid_bits bits : la différence modulo 2^valid_bits survit à un tour du compteur.
  const std::uint64_t mask = valid_bits_ >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << valid_bits_) - 1U;
  const std::uint64_t elapsed = (ticks[1] - ticks[0]) & mask;
  return static_cast<double>(elapsed) * period_ns_ / 1.0e6;
}

}  // namespace ajp::bench
