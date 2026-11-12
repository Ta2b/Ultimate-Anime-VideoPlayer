#include "app/play/pumped_task.h"

#include <algorithm>
#include <format>
#include <mutex>
#include <string>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_video.h>

#include "app/play/close_watch.h"
#include "app/play/engine_install.h"
#include "app/play/relaunch_target.h"
#include "core/log.h"

namespace ajp::app::play {

void pump_window_events() {
  SDL_PumpEvents();
}

bool window_close_requested() {
  return SDL_HasEvent(SDL_EVENT_QUIT) || SDL_HasEvent(SDL_EVENT_WINDOW_CLOSE_REQUESTED);
}

namespace {

struct Left {
  std::jthread thread;
  std::function<bool()> finished;
  Abandoned kind = Abandoned::Background;
};

struct AbandonedRegistry {
  std::mutex mutex;
  std::vector<Left> threads;
};

AbandonedRegistry& abandoned() {
  static AbandonedRegistry registry;
  return registry;
}

// Joint ceux qui ont fini ; rend le nombre de ceux qui restent. Sous le verrou du registre.
std::size_t reap(std::vector<Left>& threads) {
  std::erase_if(threads, [](Left& each) {
    if (!each.finished()) return false;
    if (each.thread.joinable()) each.thread.join();  // le corps a rendu son résultat : join immédiat
    return true;
  });
  return threads.size();
}

}  // namespace

void abandon_thread(std::jthread thread, std::function<bool()> finished, Abandoned kind) {
  AbandonedRegistry& registry = abandoned();
  const std::scoped_lock lock(registry.mutex);
  reap(registry.threads);
  registry.threads.push_back(Left{.thread = std::move(thread), .finished = std::move(finished), .kind = kind});
}

std::size_t abandoned_running() {
  AbandonedRegistry& registry = abandoned();
  const std::scoped_lock lock(registry.mutex);
  return reap(registry.threads);
}

std::size_t abandoned_running(Abandoned kind) {
  AbandonedRegistry& registry = abandoned();
  const std::scoped_lock lock(registry.mutex);
  reap(registry.threads);
  return static_cast<std::size_t>(
      std::ranges::count_if(registry.threads, [kind](const Left& each) { return each.kind == kind; }));
}

void finish_after_close(std::string_view name, const std::function<bool()>& finished) {
  // Les fenêtres se cachent : la croix a répondu. SDL_GetWindows rend une copie à libérer.
  int count = 0;
  if (SDL_Window** windows = SDL_GetWindows(&count)) {
    for (int i = 0; i < count; ++i) SDL_HideWindow(windows[i]);
    SDL_free(static_cast<void*>(windows));
  }
  if (finished()) return;
  log_channel("app")->info("{} : fenêtre fermée, fin du travail attendue", name);
  const CloseWatch stuck(kCloseWaitLimit, [what = std::string(name)] {
    exit_stuck(std::format("{} : fenêtre fermée, travail bloqué depuis {} s", what, kCloseWaitLimit.count()));
  });
  while (!finished()) {
    pump_window_events();
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }
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
  return run_pumped<std::string>("nvidia-smi",
                                 [name = std::string(gpu_name)] { return detect_cuda_arch(name); })
      .value_or(std::string());
}

}  // namespace ajp::app::play
