// Démarrage des threads de module : la seule façon de lancer un std::jthread dans src/.
//
// Deux promesses, pour tous les modules :
//   - le démarrage ne lève rien : un refus du système (std::system_error, mémoire)
//     devient Refusal ThreadUnavailable, rendu par start_thread ;
//   - aucune exception ne sort du corps : celle qui s'en échappe devient un refus
//     ThreadFailed, remis au rapporteur du thread, puis le thread se termine.
// Le corps reçoit le même rapporteur pour signaler ses propres refus.
// Fait partie de core_types : aucune dépendance hors bibliothèque standard.
#pragma once

#include <concepts>
#include <exception>
#include <expected>
#include <format>
#include <functional>
#include <future>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include "core/types.h"

namespace ajp {

// Le refus qu'est devenue l'exception en cours, à appeler dans un bloc catch.
// Ne lève jamais : sans mémoire pour le détail, le refus part sans détail.
inline Refusal refusal_from_exception(RefusalCode code, std::string_view where) noexcept {
  try {
    try {
      throw;
    } catch (const std::exception& error) {
      return Refusal{code, std::format("{} : {}", where, error.what())};
    } catch (...) {
      return Refusal{code, std::format("{} : exception inconnue", where)};
    }
  } catch (...) {
    return Refusal{code, {}};
  }
}

// Ce qu'un thread signale à son propriétaire, depuis le thread lui-même : le
// rappel pousse d'ordinaire dans la boîte de app, ou vers l'OSD.
class ThreadReport {
 public:
  ThreadReport() = default;
  explicit ThreadReport(std::function<void(const Refusal&)> sink) : sink_(std::move(sink)) {}

  explicit operator bool() const noexcept { return static_cast<bool>(sink_); }

  // Ne lève jamais : une exception du rappel est perdue, faute d'un autre
  // endroit où la signaler.
  void operator()(const Refusal& refusal) const noexcept {
    if (!sink_) return;
    try {
      sink_(refusal);
    } catch (...) {  // NOLINT(bugprone-empty-catch) : dernier recours, rien à prévenir
    }
  }

 private:
  std::function<void(const Refusal&)> sink_;
};

// Le corps d'un thread : il tourne jusqu'à la demande d'arrêt ou sa propre fin.
template <class Body>
concept ThreadBody =
    std::move_constructible<Body> && std::invocable<Body&, const std::stop_token&, const ThreadReport&>;

// Lance body(stop, report) sur un std::jthread nommé name (« source », « audio »…).
// Refus : rapporteur absent (FieldMissing), thread refusé par le système (ThreadUnavailable).
// Un refus de démarrage détruit body sans l'avoir appelé.
template <ThreadBody Body>
std::expected<std::jthread, Refusal> start_thread(std::string_view name, ThreadReport report, Body body) {
  // Tout ce qui alloue est dans le try (P2.33) : même le texte d'un refus ne fait pas lever start_thread.
  try {
    if (!report) {
      return std::unexpected(
          Refusal{RefusalCode::FieldMissing, std::format("thread {} : rapporteur absent", name)});
    }
    return std::jthread([where = std::format("thread {}", name), report = std::move(report),
                         body = std::move(body)](const std::stop_token& stop) mutable noexcept {
      try {
        body(stop, std::as_const(report));
      } catch (...) {
        report(refusal_from_exception(RefusalCode::ThreadFailed, where));
      }
    });
  } catch (...) {  // std::system_error de std::thread, ou mémoire insuffisante
    try {
      return std::unexpected(
          refusal_from_exception(RefusalCode::ThreadUnavailable, std::format("thread {}", name)));
    } catch (...) {
      return std::unexpected(Refusal{RefusalCode::ThreadUnavailable, {}});
    }
  }
}

// std::async(std::launch::async, fn) sans exception (P2.3) : l'exception du corps devient un refus
// ThreadFailed, un thread refusé par le système un futur déjà prêt sur ThreadUnavailable. Nul seulement
// sans mémoire pour ce futur.
// Deux choses à savoir (A20-05) : la destruction du dernier futur ATTEND la fin de la tâche (règle de
// std::async), donc une tâche longue reçoit un arrêt avant que son propriétaire ne la lâche ; et sous MSVC la
// tâche tourne sur le pool de Windows, sur un thread réutilisé : elle ne pose ni rôle (set_thread_role) ni
// thread_local qui lui survivrait. Pour un travail long ou de fond, start_thread.
template <class T, class Fn>
std::shared_ptr<std::future<std::expected<T, Refusal>>> start_task(std::string_view name, Fn fn) noexcept {
  using Result = std::expected<T, Refusal>;
  try {
    return std::make_shared<std::future<Result>>(
        std::async(std::launch::async, [where = std::format("tâche {}", name), fn = std::move(fn)]() mutable {
          try {
            return Result(fn());
          } catch (...) {
            return Result(std::unexpected(refusal_from_exception(RefusalCode::ThreadFailed, where)));
          }
        }));
  } catch (...) {
    try {
      std::promise<Result> refused;
      refused.set_value(std::unexpected(refusal_from_exception(RefusalCode::ThreadUnavailable, name)));
      return std::make_shared<std::future<Result>>(refused.get_future());
    } catch (...) {
      return nullptr;
    }
  }
}

}  // namespace ajp
