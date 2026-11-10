// Un relevé périodique qui ne tourne que tant qu'on le lit (A24-22) : le relevé de la mémoire vidéo et de la
// charge du GPU sert l'OSD, et continuait toutes les 500 ms jusqu'à la fin de l'épisode une fois l'OSD
// caché. Le fil du relevé attend ici entre deux tours ; sans lecture depuis idle_after, il dort jusqu'à la
// suivante, qui le réveille aussitôt.
#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <stop_token>

namespace ajp::app::play {

class SampleDemand {
 public:
  using Clock = std::chrono::steady_clock;

  // Un lecteur vient de demander la valeur : appelé de n'importe quel fil.
  void asked() {
    {
      const std::scoped_lock lock(mutex_);
      last_ = Clock::now();
    }
    wake_.notify_all();
  }

  // Attend le tour suivant : period, puis tant que personne n'a lu depuis idle_after. false : arrêt demandé.
  bool wait(const std::stop_token& stop, Clock::duration period, Clock::duration idle_after) {
    std::unique_lock lock(mutex_);
    wake_.wait_for(lock, stop, period, [] { return false; });
    wake_.wait(lock, stop, [&] { return last_ && Clock::now() - *last_ < idle_after; });
    return !stop.stop_requested();
  }

 private:
  std::mutex mutex_;
  std::condition_variable_any wake_;
  std::optional<Clock::time_point> last_;  // rien : jamais lu
};

}  // namespace ajp::app::play
