// Un travail long du thread de la fenêtre (chargement de TensorRT, nvidia-smi) passé sur un thread, la
// fenêtre pompée pendant l'attente (P2.12) : Windows ne la marque plus « Ne répond pas » au bout de cinq
// secondes. Les événements restent dans la file de SDL, lus ensuite par la boucle qui les attend.
#pragma once

#include <chrono>
#include <expected>
#include <format>
#include <functional>
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

// Les threads que run_pumped et run_pumped_for laissent à leur sort, gardés plutôt que détachés (A30-05) :
// finished dit s'il a rendu la main. Au-delà de kMaxAbandonedThreads encore bloqués, une nouvelle sonde est
// refusée tout de suite au lieu d'ajouter un thread et un descripteur bloqués de plus (A33-07).
inline constexpr std::size_t kMaxAbandonedThreads = 2;
void abandon_thread(std::jthread thread, std::function<bool()> finished);
std::size_t abandoned_running();
// À la sortie du processus : attend au plus grace que les threads abandonnés finissent, et les joint.
// Faux si l'un reste bloqué : il pourrait journaliser après close_log ou toucher SDL après SDL_Quit, et
// l'appelant termine alors par std::quick_exit, ses fichiers déjà écrits (A30-05).
bool settle_abandoned_threads(std::chrono::milliseconds grace);

// La fermeture de la fenêtre demandée (SDL_EVENT_QUIT, croix) : vue sans la retirer de la file, la boucle
// qui suit la lit aussi.
bool window_close_requested();

namespace detail {

// Le travail sur un thread, la fenêtre pompée (pump) tant qu'il tourne. Passé limit (s'il y en a un), ou à
// la fermeture de la fenêtre (closable), le thread est laissé à son sort (abandon_thread : il finira ou
// restera bloqué sans retenir la fermeture) et le refus le dit.
template <class T, class Fn>
std::expected<T, Refusal> run_pumped_until(std::string_view name,
                                           std::optional<std::chrono::milliseconds> limit, bool pump,
                                           bool closable, Fn fn) {
  struct Shared {
    std::mutex mutex;
    std::optional<std::expected<T, Refusal>> result;
  };
  if (const std::size_t stuck = abandoned_running(); limit && stuck >= kMaxAbandonedThreads) {
    std::string detail = std::format("{} : {} essais précédents ne répondent toujours pas",  // i18n-ok
                                     name, stuck);
    return std::unexpected(Refusal{RefusalCode::IoFailed, std::move(detail)});
  }
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
  constexpr auto kSlice = std::chrono::milliseconds(16);
  const auto from = std::chrono::steady_clock::now();
  bool closed = false;
  while (!limit || std::chrono::steady_clock::now() - from < *limit) {
    {
      const std::scoped_lock lock(shared->mutex);
      if (shared->result) return std::move(*shared->result);
    }
    if (pump) pump_window_events();
    if (closable && window_close_requested()) {
      closed = true;
      break;
    }
    std::this_thread::sleep_for(kSlice);
  }
  abandon_thread(std::move(*started), [shared] {
    const std::scoped_lock lock(shared->mutex);
    return shared->result.has_value();
  });
  if (closed) {
    std::string detail = std::format("{} : fenêtre fermée pendant l'attente", name);  // i18n-ok : journal
    return std::unexpected(Refusal{RefusalCode::Cancelled, std::move(detail)});
  }
  const auto seconds = limit->count() / 1000;
  std::string detail = std::format("{} : pas de réponse en {} s", name, seconds);  // i18n-ok : journal
  return std::unexpected(Refusal{RefusalCode::IoFailed, std::move(detail)});
}

}  // namespace detail

// Le travail sur un thread, la fenêtre pompée pendant l'attente. Fermer la fenêtre pendant ce temps
// (chargement de TensorRT, cuInit bloqué après un TDR) rend Cancelled tout de suite, le thread laissé à son
// sort (B30-06) : la croix répond, au lieu d'attendre un travail qui peut ne jamais finir.
template <class T, class Fn>
std::expected<T, Refusal> run_pumped(std::string_view name, Fn fn) {
  return detail::run_pumped_until<T>(name, std::nullopt, true, true, std::move(fn));
}

// run_pumped borné à limit (P2.15) : un partage réseau muet bloque read() dans le noyau, hors de portée du
// rappel d'interruption de FFmpeg. Passé limit, le thread est laissé à son sort et le refus dit IoFailed.
// pump : pomper la fenêtre (faux avant qu'elle n'existe) ; elle peut alors aussi se fermer (Cancelled).
template <class T, class Fn>
std::expected<T, Refusal> run_pumped_for(std::string_view name, std::chrono::milliseconds limit, bool pump,
                                         Fn fn) {
  return detail::run_pumped_until<T>(name, limit, pump, pump, std::move(fn));
}

// detect_cuda_arch (engine_install.h) par run_pumped ; vide aussi si le thread est refusé.
std::string pumped_cuda_arch(std::string_view gpu_name = {});

}  // namespace ajp::app::play
