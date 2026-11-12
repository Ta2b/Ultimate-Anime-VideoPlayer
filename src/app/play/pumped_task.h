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
#include <cstdint>

#include "core/thread.h"
#include "core/types.h"

namespace ajp::app::play {

// SDL_PumpEvents : la fenêtre répond au système, ses événements restent en file.
void pump_window_events();

// Les threads que run_pumped_for laisse à son sort, gardés plutôt que détachés (A30-05) : finished dit s'il a
// rendu la main. Au-delà de kMaxAbandonedThreads sondes encore bloquées, une nouvelle sonde est refusée tout de
// suite au lieu d'ajouter un thread et un descripteur bloqués de plus (A33-07). C30-02, C33-05 : le plafond ne
// compte que les sondes ; un fil de fond abandonné (sous-titres en avance, pré-analyse des plans, question du
// lecteur par défaut) ne fait plus refuser la sonde d'un fichier local ni celle de l'épisode suivant.
enum class Abandoned : std::uint8_t { Probe, Background };
inline constexpr std::size_t kMaxAbandonedThreads = 2;
void abandon_thread(std::jthread thread, std::function<bool()> finished, Abandoned kind = Abandoned::Background);
// Tous les threads abandonnés encore en route, ou seulement ceux d'une sorte.
std::size_t abandoned_running();
std::size_t abandoned_running(Abandoned kind);
// À la sortie du processus : attend au plus grace que les threads abandonnés finissent, et les joint.
// Faux si l'un reste bloqué : il pourrait journaliser après close_log ou toucher SDL après SDL_Quit, et
// l'appelant termine alors par std::quick_exit, ses fichiers déjà écrits (A30-05).
bool settle_abandoned_threads(std::chrono::milliseconds grace);

// La fermeture de la fenêtre demandée (SDL_EVENT_QUIT, croix) : vue sans la retirer de la file, la boucle
// qui suit la lit aussi.
bool window_close_requested();

// C30-01 : la fenêtre fermée pendant un travail sans délai (run_pumped). Ce travail lit des objets de son
// appelant (backend TensorRT, artefact, chemins) : il n'est plus laissé à son sort. Les fenêtres se cachent
// (la croix répond tout de suite), puis l'attente continue, pompée, jusqu'à ce que finished soit vrai. Un
// travail qui ne rend jamais la main (cuInit après un TDR) est borné par kCloseWaitLimit : la sortie du
// processus par exit_stuck, comme une fermeture bloquée sur le GPU.
inline constexpr std::chrono::seconds kCloseWaitLimit{60};
void finish_after_close(std::string_view name, const std::function<bool()>& finished);

// Un travail lancé sur un thread, attendu plus tard par le thread de la fenêtre. C18-01 : la sonde du fichier
// donné part avant l'ouverture de la fenêtre et s'attend une fois la fenêtre ouverte, pompée.
template <class T>
class PumpedTask {
 public:
  // bounded : une sonde que wait bornera (run_pumped_for). Elle est refusée tout de suite quand
  // kMaxAbandonedThreads sondes restent bloquées. Le corps détruit son résultat lui-même si personne ne
  // l'attend plus (C30-01) : rien de ce qu'il a fabriqué ne meurt plus tard sur un autre thread.
  template <class Fn>
  static std::expected<PumpedTask, Refusal> start(std::string_view name, bool bounded, Fn fn) {
    if (const std::size_t stuck = abandoned_running(Abandoned::Probe); bounded && stuck >= kMaxAbandonedThreads) {
      std::string detail = std::format("{} : {} essais précédents ne répondent toujours pas",  // i18n-ok
                                       name, stuck);
      return std::unexpected(Refusal{RefusalCode::IoFailed, std::move(detail)});
    }
    auto shared = std::make_shared<Shared>();
    auto started =
        start_thread(name, ThreadReport([shared](const Refusal& refusal) {
                       const std::scoped_lock lock(shared->mutex);
                       if (!shared->result && !shared->abandoned) shared->result = std::unexpected(refusal);
                       shared->finished = true;
                     }),
                     [shared, fn = std::move(fn)](const std::stop_token&, const ThreadReport&) mutable {
                       std::optional<std::expected<T, Refusal>> done{fn()};
                       const std::scoped_lock lock(shared->mutex);
                       if (!shared->abandoned) shared->result = std::move(done);
                       shared->finished = true;
                       // abandonné : done est détruit ici, sur ce thread, sous le verrou
                     });
    if (!started) return std::unexpected(std::move(started.error()));
    return PumpedTask(std::string(name), std::move(shared), std::move(*started));
  }

  PumpedTask(PumpedTask&&) noexcept = default;
  PumpedTask& operator=(PumpedTask&&) noexcept = default;
  PumpedTask(const PumpedTask&) = delete;
  PumpedTask& operator=(const PumpedTask&) = delete;
  // Jamais attendu (un retour anticipé de l'appelant) : le thread est laissé à son sort comme une sonde.
  ~PumpedTask() {
    if (thread_.joinable()) leave(Abandoned::Probe);
  }

  // Attend le résultat, la fenêtre pompée (pump) tant qu'il manque. limit se compte depuis le lancement.
  // Passé limit, le thread est laissé à son sort et le refus dit IoFailed. À la fermeture de la fenêtre
  // (closable) : un travail borné est laissé à son sort, un travail sans limite est attendu fenêtre cachée
  // (finish_after_close) ; le refus dit Cancelled.
  std::expected<T, Refusal> wait(std::optional<std::chrono::milliseconds> limit, bool pump, bool closable) {
    constexpr auto kSlice = std::chrono::milliseconds(16);
    bool closed = false;
    while (!limit || std::chrono::steady_clock::now() - from_ < *limit) {
      if (auto done = take()) {
        thread_.join();  // le corps a rendu son résultat : join immédiat
        return std::move(*done);
      }
      if (pump) pump_window_events();
      if (closable && window_close_requested()) {
        closed = true;
        break;
      }
      std::this_thread::sleep_for(kSlice);
    }
    if (closed && !limit) {
      const std::shared_ptr<Shared> shared = shared_;
      finish_after_close(name_, [shared] {
        const std::scoped_lock lock(shared->mutex);
        return shared->finished;
      });
      if (shared_->finished) thread_.join();
    }
    if (thread_.joinable()) leave(Abandoned::Probe);
    if (closed) {
      std::string detail = std::format("{} : fenêtre fermée pendant l'attente", name_);  // i18n-ok : journal
      return std::unexpected(Refusal{RefusalCode::Cancelled, std::move(detail)});
    }
    const auto seconds = limit->count() / 1000;
    std::string detail = std::format("{} : pas de réponse en {} s", name_, seconds);  // i18n-ok : journal
    return std::unexpected(Refusal{RefusalCode::IoFailed, std::move(detail)});
  }

 private:
  struct Shared {
    std::mutex mutex;
    std::optional<std::expected<T, Refusal>> result;
    bool finished = false;
    bool abandoned = false;
  };

  PumpedTask(std::string name, std::shared_ptr<Shared> shared, std::jthread thread)
      : name_(std::move(name)), shared_(std::move(shared)), thread_(std::move(thread)) {}

  std::optional<std::expected<T, Refusal>> take() {
    const std::scoped_lock lock(shared_->mutex);
    if (!shared_->result) return std::nullopt;
    std::optional<std::expected<T, Refusal>> out = std::move(shared_->result);
    shared_->result.reset();
    return out;
  }

  // Le résultat qui viendrait encore n'a plus de lecteur : le corps le détruira lui-même.
  void leave(Abandoned kind) {
    {
      const std::scoped_lock lock(shared_->mutex);
      shared_->abandoned = true;
      shared_->result.reset();
    }
    abandon_thread(
        std::move(thread_),
        [shared = shared_] {
          const std::scoped_lock lock(shared->mutex);
          return shared->finished;
        },
        kind);
  }

  std::string name_;
  std::shared_ptr<Shared> shared_;
  std::chrono::steady_clock::time_point from_ = std::chrono::steady_clock::now();
  std::jthread thread_;
};

namespace detail {

template <class T, class Fn>
std::expected<T, Refusal> run_pumped_until(std::string_view name,
                                           std::optional<std::chrono::milliseconds> limit, bool pump,
                                           bool closable, Fn fn) {
  auto task = PumpedTask<T>::start(name, limit.has_value(), std::move(fn));
  if (!task) return std::unexpected(std::move(task.error()));
  return task->wait(limit, pump, closable);
}

}  // namespace detail

// Le travail sur un thread, la fenêtre pompée pendant l'attente. Fermer la fenêtre pendant ce temps
// (chargement de TensorRT, cuInit bloqué après un TDR) cache la fenêtre tout de suite (B30-06) et rend
// Cancelled une fois le travail fini (C30-01 : il lit des objets de l'appelant), au plus kCloseWaitLimit.
template <class T, class Fn>
std::expected<T, Refusal> run_pumped(std::string_view name, Fn fn) {
  return detail::run_pumped_until<T>(name, std::nullopt, true, true, std::move(fn));
}

// run_pumped borné à limit (P2.15) : un partage réseau muet bloque read() dans le noyau, hors de portée du
// rappel d'interruption de FFmpeg. Passé limit, le thread est laissé à son sort et le refus dit IoFailed.
// pump : pomper la fenêtre (faux avant qu'elle n'existe) ; elle peut alors aussi se fermer (Cancelled).
// fn est laissé à son sort : il ne capture rien par référence.
template <class T, class Fn>
std::expected<T, Refusal> run_pumped_for(std::string_view name, std::chrono::milliseconds limit, bool pump,
                                         Fn fn) {
  return detail::run_pumped_until<T>(name, limit, pump, pump, std::move(fn));
}

// detect_cuda_arch (engine_install.h) par run_pumped ; vide aussi si le thread est refusé.
std::string pumped_cuda_arch(std::string_view gpu_name = {});

}  // namespace ajp::app::play
