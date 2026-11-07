// Un travail long du thread de la fenêtre (chargement de TensorRT, nvidia-smi) passé sur un thread, la
// fenêtre pompée pendant l'attente (P2.12) : Windows ne la marque plus « Ne répond pas » au bout de cinq
// secondes. Les événements restent dans la file de SDL, lus ensuite par la boucle qui les attend.
#pragma once

#include <chrono>
#include <expected>
#include <future>
#include <string>
#include <string_view>
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

// detect_cuda_arch (engine_install.h) par run_pumped ; vide aussi si le thread est refusé.
std::string pumped_cuda_arch();

}  // namespace ajp::app::play
