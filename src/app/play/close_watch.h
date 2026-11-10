// Un garde sur une attente sans délai (A10-04). pl_gpu_finish et la destruction des objets Vulkan attendent
// le GPU sans limite : sur un GPU bloqué (noyau en boucle, pilote figé), la fenêtre déjà cachée laissait un
// processus qui ne finissait jamais. Tant que le garde vit, un thread attend `limit` ; passé ce délai il
// appelle `on_stuck` (au journal, puis la sortie du processus). Détruit à temps, il ne fait rien.
#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stop_token>
#include <system_error>
#include <thread>
#include <utility>

namespace ajp::app::play {

class CloseWatch {
 public:
  CloseWatch(std::chrono::milliseconds limit, std::function<void()> on_stuck) {
    try {
      thread_ = std::jthread([this, limit, action = std::move(on_stuck)](const std::stop_token& stop) {
        std::unique_lock lock(mutex_);
        // Faux : le délai est passé sans demande d'arrêt.
        if (!wake_.wait_for(lock, stop, limit, [&] { return stop.stop_requested(); })) action();
      });
    } catch (const std::system_error&) {  // sans thread : pas de garde, la fermeture attend comme avant
    }
  }
  ~CloseWatch() = default;  // std::jthread demande l'arrêt et attend le thread
  CloseWatch(const CloseWatch&) = delete;
  CloseWatch& operator=(const CloseWatch&) = delete;

 private:
  std::mutex mutex_;
  std::condition_variable_any wake_;
  std::jthread thread_;  // en dernier : il s'arrête avant que le verrou ne disparaisse
};

}  // namespace ajp::app::play
