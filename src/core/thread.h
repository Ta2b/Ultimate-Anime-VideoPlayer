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
  if (!report) {
    return std::unexpected(
        Refusal{RefusalCode::FieldMissing, std::format("thread {} : rapporteur absent", name)});
  }
  try {
    return std::jthread([where = std::format("thread {}", name), report = std::move(report),
                         body = std::move(body)](const std::stop_token& stop) mutable noexcept {
      try {
        body(stop, std::as_const(report));
      } catch (...) {
        report(refusal_from_exception(RefusalCode::ThreadFailed, where));
      }
    });
  } catch (...) {  // std::system_error de std::thread, ou mémoire insuffisante
    return std::unexpected(
        refusal_from_exception(RefusalCode::ThreadUnavailable, std::format("thread {}", name)));
  }
}

}  // namespace ajp
