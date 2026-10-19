// Horloge de l'écran : l'horloge maître de la lecture sur un écran à rafraîchissement fixe.
// present la publie à chaque vsync (PresentDecision::clock) ; l'audio la reçoit par SyncTo et
// se rééchantillonne pour la suivre, la borne de 0,125 % étant appliquée par audio.
#pragma once

#include <cstdint>
#include <mutex>
#include <optional>

#include "core/clock.h"

namespace ajp::present {

class DisplayClock final : public MasterClock {
 public:
  // Thread main seulement.
  void publish(const ClockSnapshot& snapshot) {
    const std::scoped_lock lock(mutex_);
    snapshot_ = snapshot;
  }
  ClockSnapshot snapshot() const override {
    const std::scoped_lock lock(mutex_);
    return snapshot_;
  }

 private:
  mutable std::mutex mutex_;
  ClockSnapshot snapshot_;
};

// Écart de l'horloge audio sur l'horloge de l'écran à l'instant hôte host_us, en microsecondes
// média : positif quand l'audio est en avance. Absent si l'une des deux est invalide ou si
// elles viennent d'époques différentes (un Flush est en cours).
std::optional<std::int64_t> audio_offset_us(const ClockSnapshot& audio, const ClockSnapshot& display,
                                            std::int64_t host_us);

}  // namespace ajp::present
