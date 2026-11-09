#include "app/play/ring_watcher.h"

#include <algorithm>
#include <utility>

#include "core/clock.h"
#include "core/log.h"
#include "core/thread.h"

namespace ajp::app::play {
namespace {

constexpr std::uint64_t kTimeoutNs =
    100'000'000;                         // 100 ms : l'arrêt et les nouvelles entrées n'attendent pas plus
constexpr std::size_t kMaxEntries = 64;  // au-delà, les plus anciennes ne sont plus suivies

template <class Fn>
Fn load(PFN_vkGetDeviceProcAddr proc, VkDevice device, const char* name, const char* alias = nullptr) {
  auto fn = reinterpret_cast<Fn>(proc(device, name));
  if (fn == nullptr && alias != nullptr) fn = reinterpret_cast<Fn>(proc(device, alias));
  return fn;
}

}  // namespace

std::unique_ptr<RingWatcher> RingWatcher::start(VkDevice device, PFN_vkGetDeviceProcAddr device_proc,
                                                Waker waker, void* context) {
  if (device == VK_NULL_HANDLE || device_proc == nullptr || waker == nullptr) return nullptr;
  Api api{
      .wait = load<PFN_vkWaitSemaphores>(device_proc, device, "vkWaitSemaphores", "vkWaitSemaphoresKHR"),
      .counter = load<PFN_vkGetSemaphoreCounterValue>(device_proc, device, "vkGetSemaphoreCounterValue",
                                                      "vkGetSemaphoreCounterValueKHR"),
      .signal = load<PFN_vkSignalSemaphore>(device_proc, device, "vkSignalSemaphore", "vkSignalSemaphoreKHR"),
      .create = load<PFN_vkCreateSemaphore>(device_proc, device, "vkCreateSemaphore"),
      .destroy = load<PFN_vkDestroySemaphore>(device_proc, device, "vkDestroySemaphore")};
  return start_with(device, api, waker, context);
}

std::unique_ptr<RingWatcher> RingWatcher::start_with(VkDevice device, const Api& api, Waker waker,
                                                     void* context) {
  if (api.wait == nullptr || api.counter == nullptr || api.signal == nullptr || api.create == nullptr ||
      api.destroy == nullptr)
    return nullptr;
  const VkSemaphoreTypeCreateInfo type{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                       .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                       .initialValue = 0};
  const VkSemaphoreCreateInfo info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type};
  VkSemaphore wake = VK_NULL_HANDLE;
  if (api.create(device, &info, nullptr, &wake) != VK_SUCCESS) return nullptr;
  auto watcher = std::unique_ptr<RingWatcher>(new RingWatcher(device, api, wake, waker, context));
  if (!watcher->thread_.joinable()) return nullptr;
  return watcher;
}

RingWatcher::RingWatcher(VkDevice device, Api api, VkSemaphore wake, Waker waker, void* context)
    : device_(device), api_(api), wake_(wake), waker_(waker), context_(context) {
  auto started = start_thread(
      "veilleur",
      ThreadReport([](const Refusal& refusal) { log_channel("app")->warn("{}", refusal.detail); }),
      [this](const std::stop_token& stop, const ThreadReport& /*report*/) { run(stop); });
  if (started) {
    thread_ = std::move(*started);
  } else {
    log_channel("app")->warn("{} : sans veilleur d'écriture", started.error().detail);
  }
}

RingWatcher::~RingWatcher() {
  {
    const std::scoped_lock lock(mutex_);
    stopping_ = true;
    entries_.clear();
    nudge();
  }
  if (thread_.joinable()) {
    thread_.request_stop();
    thread_.join();
  }
  log_channel("app")->info("veilleur d'écriture : {} réveils de main", wakes_);
  api_.destroy(device_, wake_, nullptr);
}

void RingWatcher::nudge() {
  const VkSemaphoreSignalInfo signal{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = wake_, .value = ++wake_value_};
  (void)api_.signal(device_, &signal);
}

void RingWatcher::watch(VkSemaphore semaphore, std::uint64_t value) {
  if (semaphore == VK_NULL_HANDLE) return;
  const std::scoped_lock lock(mutex_);
  if (std::ranges::any_of(entries_,
                          [&](const Entry& e) { return e.semaphore == semaphore && e.value == value; }))
    return;
  if (entries_.size() >= kMaxEntries) entries_.erase(entries_.begin());
  entries_.push_back(Entry{.semaphore = semaphore, .value = value, .written_us = 0});
  nudge();
}

std::optional<std::int64_t> RingWatcher::take(VkSemaphore semaphore, std::uint64_t value) {
  const std::scoped_lock lock(mutex_);
  const auto it = std::ranges::find_if(
      entries_, [&](const Entry& e) { return e.semaphore == semaphore && e.value == value; });
  if (it == entries_.end() || it->written_us == 0) return std::nullopt;
  const std::int64_t at = it->written_us;
  entries_.erase(it);
  return at;
}

void RingWatcher::forget_all() {
  std::unique_lock lock(mutex_);
  entries_.clear();
  if (!thread_.joinable() || dead_) return;
  const std::uint64_t epoch = ++forget_epoch_;
  nudge();
  settled_.wait(lock, [&] { return seen_epoch_ >= epoch || stopping_ || dead_; });
}

bool RingWatcher::alive() const {
  const std::scoped_lock lock(mutex_);
  return thread_.joinable() && !dead_;
}

// Le thread sort sur une erreur : forget_all ne l'attend plus, main sonde l'anneau comme avant, et le journal
// le dit une fois (A10-03, GPU-10).
void RingWatcher::die(VkResult result) {
  {
    const std::scoped_lock lock(mutex_);
    dead_ = true;
  }
  settled_.notify_all();
  log_channel("app")->warn(
      "veilleur d'écriture : vkWaitSemaphores a échoué ({}), l'anneau est de nouveau sondé",
      static_cast<int>(result));
}

std::uint64_t RingWatcher::wakes() const {
  const std::scoped_lock lock(mutex_);
  return wakes_;
}

void RingWatcher::run(const std::stop_token& stop) {
  std::vector<VkSemaphore> semaphores;
  std::vector<std::uint64_t> values;
  while (!stop.stop_requested()) {
    {
      // Hors de toute attente : la liste se recopie, et forget_all peut rendre la main.
      std::unique_lock lock(mutex_);
      if (stopping_) return;
      seen_epoch_ = forget_epoch_;
      settled_.notify_all();
      semaphores.assign(1, wake_);
      values.assign(1, wake_value_ + 1);
      for (const Entry& e : entries_) {
        if (e.written_us != 0) continue;
        semaphores.push_back(e.semaphore);
        values.push_back(e.value);
      }
    }
    const VkSemaphoreWaitInfo info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                   .flags = VK_SEMAPHORE_WAIT_ANY_BIT,
                                   .semaphoreCount = static_cast<std::uint32_t>(semaphores.size()),
                                   .pSemaphores = semaphores.data(),
                                   .pValues = values.data()};
    const VkResult waited = api_.wait(device_, &info, kTimeoutNs);
    const std::int64_t now = host_now_us();
    if (waited != VK_SUCCESS && waited != VK_TIMEOUT) return die(waited);  // appareil perdu
    if (waited != VK_SUCCESS) continue;
    bool written = false;
    {
      // Les entrées de la liste seulement : forget_all a pu la vider, ses sémaphores ne se touchent plus.
      const std::scoped_lock lock(mutex_);
      for (Entry& e : entries_) {
        if (e.written_us != 0) continue;
        std::uint64_t value = 0;
        if (api_.counter(device_, e.semaphore, &value) == VK_SUCCESS && value >= e.value) {
          e.written_us = now;
          written = true;
        }
      }
      if (written) ++wakes_;
    }
    if (written) waker_(context_);
  }
}

}  // namespace ajp::app::play
