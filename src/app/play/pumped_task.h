// Un travail long du thread de la fenêtre (chargement de TensorRT, nvidia-smi) passé sur un thread, la
// fenêtre pompée pendant l'attente (P2.12) : Windows ne la marque plus « Ne répond pas » au bout de cinq
// secondes. Les événements restent dans la file de SDL, lus ensuite par la boucle qui les attend.
#pragma once

#include <chrono>
#include <expected>
#include <format>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "core/thread.h"
#include "core/types.h"

namespace ajp::app::play {

// SDL_PumpEvents : la fenêtre répond au système, ses événements restent en file.
void pump_window_events();

template <class T, class Fn>
std::expected<T, Refusal> run_pumped(std::string_view name, Fn fn) {
  const auto task = start_task<T>(name, std::move(fn));
  if (task == nullptr) return std::unexpected(Refusal{RefusalCode::ThreadUnavailable, std::string(name)});
  constexpr auto kSlice = std::chrono::milliseconds(16);
  while (task->wait_for(kSlice) != std::future_status::ready) pump_window_events();
  return task->get();
}

// run_pumped borné à limit (P2.15) : un partage réseau muet bloque read() dans le noyau, hors de portée du
// rappel d'interruption de FFmpeg. Passé limit, le thread est laissé à son sort (détaché, il finira ou
// restera bloqué sans retenir la fermeture) et le refus dit IoFailed. pump : pomper la fenêtre (faux avant
// qu'elle n'existe).
template <class T, class Fn>
std::expected<T, Refusal> run_pumped_for(std::string_view name, std::chrono::milliseconds limit, bool pump,
                                         Fn fn) {
  struct Shared {
    std::mutex mutex;
    std::optional<std::expected<T, Refusal>> result;
  };
  auto shared = std::make_shared<Shared>();
  auto started =
      start_thread(name, ThreadReport([shared](const Refusal& refusal) {
                     const std::scoped_lock lock(shared->mutex);
                     if (!shared->result) shared->result = std::unexpected(refusal);
                   }),
                   [shared, fn = std::move(fn)](const std::stop_token&, const ThreadReport&) mutable {
                     auto done = fn();
                     const std::scoped_lock lock(shared->mutex);
                     shared->result = std::move(done);
                   });
  if (!started) return std::unexpected(std::move(started.error()));
  started->detach();
  constexpr auto kSlice = std::chrono::milliseconds(16);
  const auto until = std::chrono::steady_clock::now() + limit;
  while (std::chrono::steady_clock::now() < until) {
    {
      const std::scoped_lock lock(shared->mutex);
      if (shared->result) return std::move(*shared->result);
    }
    if (pump) pump_window_events();
    std::this_thread::sleep_for(kSlice);
  }
  const auto seconds = limit.count() / 1000;
  std::string detail = std::format("{} : pas de réponse en {} s", name, seconds);  // i18n-ok : journal
  return std::unexpected(Refusal{RefusalCode::IoFailed, std::move(detail)});
}

// detect_cuda_arch (engine_install.h) par run_pumped ; vide aussi si le thread est refusé.
std::string pumped_cuda_arch(std::string_view gpu_name = {});

}  // namespace ajp::app::play
