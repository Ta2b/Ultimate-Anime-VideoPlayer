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

// La clé du refus rendu quand la fenêtre se ferme pendant l'attente : ce n'est ni un échec de l'étage ni
// « Annuler et regarder », la lecture s'arrête (watch_now, prepare_chain, play).
inline constexpr std::string_view kWindowClosedKey = "window.closed";

// Combien de temps un travail qui emprunte l'état de l'appelant (run_pumped_borrowing) est encore attendu
// après la croix, avant d'être laissé à son sort.
inline constexpr std::chrono::seconds kBorrowGrace{15};

namespace detail {

// Le travail sur un thread, la fenêtre pompée (pump) tant qu'il tourne. Passé limit (s'il y en a un), ou à
// la fermeture de la fenêtre (closable), le thread est laissé à son sort (abandon_thread : il finira ou
// restera bloqué sans retenir la fermeture) et le refus le dit. borrowing : le travail lit l'état de
// l'appelant, qui ne doit pas disparaître sous lui ; la fermeture est alors attendue jusqu'à kBorrowGrace
// (la fenêtre reste pompée), et seulement passé ce délai le thread est abandonné.
template <class T, class Fn>
std::expected<T, Refusal> run_pumped_until(std::string_view name,
                                           std::optional<std::chrono::milliseconds> limit, bool pump,
                                           bool closable, Fn fn, bool borrowing = false) {
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
  const auto window_closed = [&name] {
    std::string detail = std::format("{} : fenêtre fermée pendant l'attente", name);  // i18n-ok : journal
    return std::unexpected(Refusal{
        .code = RefusalCode::Cancelled, .detail = std::move(detail), .key = std::string(kWindowClosedKey)});
  };
  bool closed = false;
  std::optional<std::chrono::steady_clock::time_point> close_seen;
  while (!limit || std::chrono::steady_clock::now() - from < *limit) {
    {
      const std::scoped_lock lock(shared->mutex);
      // Fini après la croix : le résultat est jeté, la fermeture prime.
      if (shared->result)
        return close_seen ? std::expected<T, Refusal>(window_closed()) : std::move(*shared->result);
    }
    if (pump) pump_window_events();
    if (closable && window_close_requested()) {
      const auto now = std::chrono::steady_clock::now();
      if (!close_seen) close_seen = now;
      if (!borrowing || now - *close_seen >= kBorrowGrace) {
        closed = true;
        break;
      }
    }
    std::this_thread::sleep_for(kSlice);
  }
  abandon_thread(std::move(*started), [shared] {
    const std::scoped_lock lock(shared->mutex);
    return shared->result.has_value();
  });
  if (closed) return window_closed();
  const auto seconds = limit->count() / 1000;
  std::string detail = std::format("{} : pas de réponse en {} s", name, seconds);  // i18n-ok : journal
  return std::unexpected(Refusal{RefusalCode::IoFailed, std::move(detail)});
}

}  // namespace detail

// Le travail sur un thread, la fenêtre pompée pendant l'attente. Fermer la fenêtre pendant ce temps
// (chargement de TensorRT, cuInit bloqué après un TDR) rend Cancelled tout de suite, le thread laissé à son
// sort (B30-06) : la croix répond, au lieu d'attendre un travail qui peut ne jamais finir. Le thread survit
// donc à l'appelant : fn ne capture RIEN par référence (copies, ou shared_ptr), sinon il lirait une pile ou
// un objet détruit. Un travail qui doit emprunter l'état de l'appelant passe par run_pumped_borrowing.
template <class T, class Fn>
std::expected<T, Refusal> run_pumped(std::string_view name, Fn fn) {
  return detail::run_pumped_until<T>(name, std::nullopt, true, true, std::move(fn));
}

// run_pumped pour un travail qui lit des objets de l'appelant (un backend, un artefact) : à la fermeture de
// la fenêtre le thread est attendu (au plus kBorrowGrace) au lieu d'être abandonné, sinon l'appelant rendrait
// la main et détruirait ce que le thread lit encore. Un travail qui peut ne jamais finir (cuInit après un
// TDR) ne doit capturer que des copies et passer par run_pumped.
template <class T, class Fn>
std::expected<T, Refusal> run_pumped_borrowing(std::string_view name, Fn fn) {
  return detail::run_pumped_until<T>(name, std::nullopt, true, true, std::move(fn), true);
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
