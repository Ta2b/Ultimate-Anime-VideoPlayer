#include "app/play/pumped_task.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

#include <SDL3/SDL_events.h>

#include "app/play/engine_install.h"

namespace ajp::app::play {

void pump_window_events() {
  SDL_PumpEvents();
}

namespace {

struct Abandoned {
  std::jthread thread;
  std::function<bool()> finished;
};

struct AbandonedRegistry {
  std::mutex mutex;
  std::vector<Abandoned> threads;
};

AbandonedRegistry& abandoned() {
  static AbandonedRegistry registry;
  return registry;
}

// Joint ceux qui ont fini ; rend le nombre de ceux qui restent. Sous le verrou du registre.
std::size_t reap(std::vector<Abandoned>& threads) {
  std::erase_if(threads, [](Abandoned& each) {
    if (!each.finished()) return false;
    if (each.thread.joinable()) each.thread.join();  // le corps a rendu son résultat : join immédiat
    return true;
  });
  return threads.size();
}

}  // namespace

void abandon_thread(std::jthread thread, std::function<bool()> finished) {
  AbandonedRegistry& registry = abandoned();
  const std::scoped_lock lock(registry.mutex);
  reap(registry.threads);
  registry.threads.push_back(Abandoned{.thread = std::move(thread), .finished = std::move(finished)});
}

std::size_t abandoned_running() {
  AbandonedRegistry& registry = abandoned();
  const std::scoped_lock lock(registry.mutex);
  return reap(registry.threads);
}

bool settle_abandoned_threads(std::chrono::milliseconds grace) {
  const auto until = std::chrono::steady_clock::now() + grace;
  while (abandoned_running() != 0) {
    if (std::chrono::steady_clock::now() >= until) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return true;
}

std::string pumped_cuda_arch(std::string_view gpu_name) {
  return run_pumped<std::string>("nvidia-smi", [gpu_name] { return detect_cuda_arch(gpu_name); })
      .value_or(std::string());
}

}  // namespace ajp::app::play
