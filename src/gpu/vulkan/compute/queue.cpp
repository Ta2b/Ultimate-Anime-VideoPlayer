// Soumission sur la file compute (par lots), attentes hôte et command buffers jetables.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <format>
#include <optional>
#include <span>
#include <vector>

#include <spdlog/logger.h>

#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/recording.h"

namespace ajp::gpu::vk {
namespace {

// Au-delà, le lot part sans attendre le prochain flush : borne la latence et la mémoire.
constexpr std::size_t kMaxBatch = 64;

// Rend le command buffer jetable à son pool quand la soumission est finie ou n'a pas eu lieu.
class OnceBuffer {
 public:
  OnceBuffer(const DeviceFunctions& fn, VkDevice device, VkCommandPool pool)
      : fn_(fn), device_(device), pool_(pool) {}
  OnceBuffer(const OnceBuffer&) = delete;
  OnceBuffer& operator=(const OnceBuffer&) = delete;
  OnceBuffer(OnceBuffer&&) = delete;
  OnceBuffer& operator=(OnceBuffer&&) = delete;
  ~OnceBuffer() {
    if (buffer != VK_NULL_HANDLE && !pending) fn_.vkFreeCommandBuffers(device_, pool_, 1, &buffer);
  }

  VkCommandBuffer buffer = VK_NULL_HANDLE;
  bool pending = false;  // soumis et pas encore attendu : le libérer serait invalide

 private:
  const DeviceFunctions& fn_;
  VkDevice device_;
  VkCommandPool pool_;
};

}  // namespace

Done VkContext::queue_submit(std::span<const VkSubmitInfo> infos, std::size_t lane) {
  // Les files des voies en plus ne sont qu'à nous : seule la principale passe par le verrou.
  const bool main = lane == 0;
  if (lost()) return refuse(RefusalCode::DeviceLost, "vkQueueSubmit : appareil perdu plus tôt, à relancer");
  if (main && lock_queue_) lock_queue_();
  const VkResult result =
      device_fn_.vkQueueSubmit(main ? queue_ : lane_queues_[lane - 1],
                               static_cast<std::uint32_t>(infos.size()), infos.data(), VK_NULL_HANDLE);
  if (main && unlock_queue_) unlock_queue_();
  ++queue_submits_;
  submit_infos_ += infos.size();
  if (result == VK_SUCCESS && !infos.empty() && infos.back().signalSemaphoreCount > 0) {
    const VkSubmitInfo& info = infos.back();
    const auto* timeline = static_cast<const VkTimelineSemaphoreSubmitInfo*>(info.pNext);
    last_submitted_[lane] = {.semaphore = info.pSignalSemaphores[0],
                             .value = timeline->pSignalSemaphoreValues[0]};
  }
  return observe(result, "vkQueueSubmit");
}

bool VkContext::settle(std::optional<TimelinePoint> point) {
  // Sans point : la dernière soumission de chaque voie.
  return point ? settle(std::span(&*point, 1)) : settle(std::span<const TimelinePoint>(last_submitted_));
}

std::chrono::nanoseconds VkContext::settle_timeout() const {
  const auto longest = std::chrono::nanoseconds(longest_wait_ns_.load(std::memory_order_relaxed));
  return std::clamp<std::chrono::nanoseconds>(
      longest * kSettleFactor, settle_floor_,
      std::max<std::chrono::nanoseconds>(settle_floor_, kSettleCeiling));
}

void VkContext::note_wait(std::chrono::steady_clock::time_point since) {
  const std::int64_t took = (std::chrono::steady_clock::now() - since).count();
  std::int64_t seen = longest_wait_ns_.load(std::memory_order_relaxed);
  while (took > seen && !longest_wait_ns_.compare_exchange_weak(seen, took, std::memory_order_relaxed)) {
  }
}

bool VkContext::settle(std::span<const TimelinePoint> points) {
  // Le lot part d'abord, même sous un GPU bloqué : il ne garde pas un enregistrement détruit, et un
  // travail encore en attente lirait la ressource après sa fin de vie.
  if (auto flushed = flush(); !flushed) log().error("fin de vie : {}", flushed.error().detail);
  const bool was_stalled = stalled();
  std::vector<VkSemaphore> semaphores;
  std::vector<std::uint64_t> values;
  for (const TimelinePoint& target : points) {
    if (target.semaphore == VK_NULL_HANDLE || target.value == 0) continue;
    semaphores.push_back(target.semaphore);
    values.push_back(target.value);
  }
  if (semaphores.empty()) return true;
  const VkSemaphoreWaitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                 .semaphoreCount = static_cast<std::uint32_t>(semaphores.size()),
                                 .pSemaphores = semaphores.data(),
                                 .pValues = values.data()};
  // Un GPU déjà tenu pour bloqué n'est plus attendu : un simple constat, qui le réarme si ces points sont
  // atteints (P2.16). Un appareil perdu ne se réarme pas.
  const std::chrono::nanoseconds timeout = was_stalled ? std::chrono::nanoseconds::zero() : settle_timeout();
  if (was_stalled && lost()) return false;
  const auto since = std::chrono::steady_clock::now();
  const VkResult result =
      device_fn_.vkWaitSemaphores(device_, &wait, static_cast<std::uint64_t>(timeout.count()));
  if (result == VK_SUCCESS) {
    if (was_stalled) {
      stalled_.store(false, std::memory_order_release);
      log().info("fin de vie : le GPU a rattrapé la valeur {}, les attentes reprennent", values.back());
    } else {
      note_wait(since);
    }
    return true;
  }
  if (result == VK_ERROR_DEVICE_LOST) mark_lost();
  if (was_stalled) return false;
  stalled_.store(true, std::memory_order_release);
  log().error(
      "fin de vie : le GPU n'a pas atteint la valeur {} en {} ms ({}) ; ses ressources restent en place",
      values.back(), std::chrono::duration_cast<std::chrono::milliseconds>(timeout).count(),
      result == VK_TIMEOUT ? "délai dépassé" : "appareil perdu");
  return false;
}

Done VkContext::submit(Recording& recording, std::span<const TimelinePoint> waits,
                       std::span<const TimelinePoint> signals) {
  if (auto ok = enqueue(recording, waits, signals); !ok) return ok;
  return flush();
}

Done VkContext::enqueue(Recording& recording, std::span<const TimelinePoint> waits,
                        std::span<const TimelinePoint> signals) {
  return enqueue(recording, waits, signals, 0);
}

Done VkContext::enqueue_join(std::span<const TimelinePoint> waits, std::span<const TimelinePoint> signals,
                             std::size_t lane) {
  if (lane >= queue_lanes()) return refuse(RefusalCode::InputMismatch, "voie sans file");
  for (const TimelinePoint& point : waits) {
    if (point.semaphore == VK_NULL_HANDLE)
      return refuse(RefusalCode::InputMismatch, "attente sur un sémaphore nul");
  }
  if (signals.empty() || signals.front().semaphore == VK_NULL_HANDLE)
    return refuse(RefusalCode::InputMismatch, "jointure sans signal");
  if (batched_ == kMaxBatch) {
    if (auto ok = flush(); !ok) return ok;
  }
  if (batched_ == batch_.size()) batch_.emplace_back();
  Batched& entry = batch_[batched_];
  entry.recording = nullptr;
  entry.lane = lane;
  entry.own_waits = 0;
  entry.wait_semaphores.clear();
  entry.wait_values.clear();
  for (const TimelinePoint& point : waits) {
    entry.wait_semaphores.push_back(point.semaphore);
    entry.wait_values.push_back(point.value);
  }
  entry.signal_semaphores.clear();
  entry.signal_values.clear();
  for (const TimelinePoint& point : signals) {
    entry.signal_semaphores.push_back(point.semaphore);
    entry.signal_values.push_back(point.value);
  }
  ++batched_;
  return {};
}

Done VkContext::enqueue(Recording& recording, std::span<const TimelinePoint> waits,
                        std::span<const TimelinePoint> signals, std::size_t lane) {
  if (lane >= queue_lanes()) return refuse(RefusalCode::InputMismatch, "voie sans file");
  if (&recording.context_ != this) {
    return refuse(RefusalCode::InputMismatch, "enregistrement d'un autre contexte Vulkan");
  }
  if (!recording.finished_) return refuse(RefusalCode::InputMismatch, "enregistrement non terminé");
  for (const TimelinePoint& point : waits) {
    if (point.semaphore == VK_NULL_HANDLE)
      return refuse(RefusalCode::InputMismatch, "attente sur un sémaphore nul");
  }
  for (const TimelinePoint& point : signals) {
    if (point.semaphore == VK_NULL_HANDLE)
      return refuse(RefusalCode::InputMismatch, "signal d'un sémaphore nul");
  }
  if (batched_ == kMaxBatch) {
    if (auto ok = flush(); !ok) return ok;
  }
  if (batched_ == batch_.size()) batch_.emplace_back();
  Batched& entry = batch_[batched_];
  entry.recording = &recording;
  entry.lane = lane;
  entry.wait_semaphores.clear();
  entry.wait_values.clear();
  // L'exécution précédente du même enregistrement, attendue côté GPU (dans le même lot au
  // besoin : un sémaphore timeline peut attendre une valeur signalée plus tôt dans l'appel).
  entry.own_waits = 0;
  if (recording.submitted_ > 0) {
    entry.wait_semaphores.push_back(recording.runs_.handle());
    entry.wait_values.push_back(recording.submitted_);
    entry.own_waits = 1;
  }
  for (const TimelinePoint& point : waits) {
    entry.wait_semaphores.push_back(point.semaphore);
    entry.wait_values.push_back(point.value);
  }
  entry.signal_semaphores.assign(1, recording.runs_.handle());
  entry.signal_values.assign(1, recording.submitted_ + 1);
  for (const TimelinePoint& point : signals) {
    entry.signal_semaphores.push_back(point.semaphore);
    entry.signal_values.push_back(point.value);
  }
  ++batched_;
  ++recording.submitted_;
  return {};
}

namespace {

// La valeur signalée pour ce sémaphore par le groupe, s'il le signale.
std::optional<std::uint64_t> signalled(std::span<const VkSemaphore> semaphores,
                                       std::span<const std::uint64_t> values, VkSemaphore semaphore) {
  for (std::size_t i = 0; i < semaphores.size(); ++i) {
    if (semaphores[i] == semaphore) return values[i];
  }
  return std::nullopt;
}

// Ajoute (sémaphore, valeur) à une liste, ou relève la valeur déjà présente : une attente comme un
// signal sur un timeline valent pour leur plus grande valeur.
void add_point(std::vector<VkSemaphore>& semaphores, std::vector<std::uint64_t>& values,
               VkSemaphore semaphore, std::uint64_t value) {
  for (std::size_t i = 0; i < semaphores.size(); ++i) {
    if (semaphores[i] == semaphore) {
      values[i] = std::max(values[i], value);
      return;
    }
  }
  semaphores.push_back(semaphore);
  values.push_back(value);
}

}  // namespace

bool VkContext::joins(const Group& group, const Batched& entry) {
  // Une autre voie, une jointure (sans commande), ou un groupe de jointure : un groupe à part.
  if (group.lane != entry.lane || entry.recording == nullptr || group.count == 0) return false;
  // Une attente que le groupe ne satisfait pas lui-même (sémaphore d'ailleurs, ou valeur au-delà
  // de ce qu'il signale) ouvrirait un groupe où les enregistrements d'avant l'attendraient aussi.
  // L'exécution précédente du même enregistrement, signalée par une soumission passée, se fond.
  for (std::size_t i = entry.own_waits; i < entry.wait_semaphores.size(); ++i) {
    const auto value = signalled(group.signal_semaphores, group.signal_values, entry.wait_semaphores[i]);
    if (!value || *value < entry.wait_values[i]) return false;
  }
  return true;
}

void VkContext::merge(Group& group, const Batched& entry) {
  // Le compte des command buffers : une jointure n'en a pas.
  const std::size_t buffers = entry.recording != nullptr ? 1 : 0;
  for (std::size_t i = 0; i < entry.wait_semaphores.size(); ++i) {
    // Signalée par le groupe : l'ordre de soumission et la barrière finale de chaque
    // enregistrement ordonnent déjà celui-ci après ; attendre bloquerait, le signal venant à la fin.
    const auto value = signalled(group.signal_semaphores, group.signal_values, entry.wait_semaphores[i]);
    if (value && *value >= entry.wait_values[i]) continue;
    add_point(group.wait_semaphores, group.wait_values, entry.wait_semaphores[i], entry.wait_values[i]);
  }
  for (std::size_t i = 0; i < entry.signal_semaphores.size(); ++i)
    add_point(group.signal_semaphores, group.signal_values, entry.signal_semaphores[i],
              entry.signal_values[i]);
  group.count += buffers;
}

Done VkContext::flush() {
  if (batched_ == 0) return {};
  if (before_flush_) {
    if (auto ok = before_flush_(); !ok) return ok;
  }
  batch_buffers_.resize(batched_);
  std::size_t groups = 0;
  std::size_t buffers = 0;
  for (std::size_t k = 0; k < batched_; ++k) {
    const Batched& entry = batch_[k];
    if (entry.recording != nullptr) batch_buffers_[buffers] = entry.recording->command_buffer_;
    if (groups == 0 || !joins(batch_groups_[groups - 1], entry)) {
      if (groups == batch_groups_.size()) batch_groups_.emplace_back();
      Group& group = batch_groups_[groups++];
      group.lane = entry.lane;
      group.entry = k;
      group.first = buffers;
      group.count = 0;
      group.wait_semaphores.clear();
      group.wait_values.clear();
      group.signal_semaphores.clear();
      group.signal_values.clear();
    }
    merge(batch_groups_[groups - 1], entry);
    if (entry.recording != nullptr) ++buffers;
  }
  batch_timelines_.resize(groups);
  batch_infos_.resize(groups);
  for (std::size_t g = 0; g < groups; ++g) {
    Group& group = batch_groups_[g];
    group.stages.assign(group.wait_semaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    batch_timelines_[g] = {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = static_cast<std::uint32_t>(group.wait_values.size()),
        .pWaitSemaphoreValues = group.wait_values.data(),
        .signalSemaphoreValueCount = static_cast<std::uint32_t>(group.signal_values.size()),
        .pSignalSemaphoreValues = group.signal_values.data()};
    batch_infos_[g] = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .pNext = &batch_timelines_[g],
                       .waitSemaphoreCount = static_cast<std::uint32_t>(group.wait_semaphores.size()),
                       .pWaitSemaphores = group.wait_semaphores.data(),
                       .pWaitDstStageMask = group.stages.data(),
                       .commandBufferCount = static_cast<std::uint32_t>(group.count),
                       .pCommandBuffers = group.count > 0 ? &batch_buffers_[group.first] : nullptr,
                       .signalSemaphoreCount = static_cast<std::uint32_t>(group.signal_semaphores.size()),
                       .pSignalSemaphores = group.signal_semaphores.data()};
  }
  const std::size_t count = batched_;
  batched_ = 0;
  // Un vkQueueSubmit par suite de groupes de la même voie, dans l'ordre du lot : un signal part
  // toujours avant l'attente d'une autre voie sur lui.
  Done ok;
  for (std::size_t g = 0; g < groups && ok;) {
    std::size_t end = g + 1;
    while (end < groups && batch_groups_[end].lane == batch_groups_[g].lane) ++end;
    ok = queue_submit(std::span(batch_infos_).subspan(g, end - g), batch_groups_[g].lane);
    if (!ok) {
      // Rien n'est parti depuis ce groupe : ses enregistrements et les suivants n'attendront pas
      // des exécutions jamais soumises.
      for (std::size_t k = batch_groups_[g].entry; k < count; ++k) {
        if (batch_[k].recording != nullptr) --batch_[k].recording->submitted_;
      }
    }
    g = end;
  }
  return ok;
}

Done VkContext::wait_on_host(const TimelinePoint& point, std::chrono::nanoseconds timeout) {
  if (point.semaphore == VK_NULL_HANDLE)
    return refuse(RefusalCode::InputMismatch, "attente d'un sémaphore nul");
  if (auto ok = flush(); !ok) return ok;
  ++host_waits_;
  const VkSemaphoreWaitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                 .semaphoreCount = 1,
                                 .pSemaphores = &point.semaphore,
                                 .pValues = &point.value};
  const auto nanoseconds = static_cast<std::uint64_t>(timeout.count() < 0 ? 0 : timeout.count());
  const auto since = std::chrono::steady_clock::now();
  auto done = observe(device_fn_.vkWaitSemaphores(device_, &wait, nanoseconds),
                      std::format("vkWaitSemaphores(valeur {})", point.value));
  if (done) note_wait(since);
  return done;
}

Done VkContext::run_once(const std::function<void(VkCommandBuffer)>& record) {
  if (auto ok = flush(); !ok) return ok;  // les copies voient tout ce qui a été soumis avant
  OnceBuffer once(device_fn_, device_, once_pool_);
  const VkCommandBufferAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                             .commandPool = once_pool_,
                                             .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                             .commandBufferCount = 1};
  if (auto ok = check(device_fn_.vkAllocateCommandBuffers(device_, &allocate, &once.buffer),
                      "vkAllocateCommandBuffers");
      !ok) {
    return ok;
  }
  const VkCommandBufferBeginInfo begin{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  if (auto ok = check(device_fn_.vkBeginCommandBuffer(once.buffer, &begin), "vkBeginCommandBuffer"); !ok)
    return ok;
  record(once.buffer);
  if (auto ok = check(device_fn_.vkEndCommandBuffer(once.buffer), "vkEndCommandBuffer"); !ok) return ok;

  const std::uint64_t value = ++once_count_;
  // Après tout ce que les autres voies ont soumis, comme sur une file unique.
  std::vector<VkSemaphore> waits;
  std::vector<std::uint64_t> wait_values;
  for (std::size_t lane = 1; lane < last_submitted_.size(); ++lane) {
    if (last_submitted_[lane].semaphore == VK_NULL_HANDLE) continue;
    waits.push_back(last_submitted_[lane].semaphore);
    wait_values.push_back(last_submitted_[lane].value);
  }
  const std::vector<VkPipelineStageFlags> stages(waits.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  const VkTimelineSemaphoreSubmitInfo timeline{
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .waitSemaphoreValueCount = static_cast<std::uint32_t>(wait_values.size()),
      .pWaitSemaphoreValues = wait_values.data(),
      .signalSemaphoreValueCount = 1,
      .pSignalSemaphoreValues = &value};
  const VkSubmitInfo info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .pNext = &timeline,
                          .waitSemaphoreCount = static_cast<std::uint32_t>(waits.size()),
                          .pWaitSemaphores = waits.data(),
                          .pWaitDstStageMask = stages.data(),
                          .commandBufferCount = 1,
                          .pCommandBuffers = &once.buffer,
                          .signalSemaphoreCount = 1,
                          .pSignalSemaphores = &once_done_};
  if (auto ok = queue_submit(std::span(&info, 1)); !ok) return ok;
  once.pending = true;
  if (auto ok = wait_on_host({.semaphore = once_done_, .value = value}); !ok) return ok;
  once.pending = false;
  return {};
}

}  // namespace ajp::gpu::vk
