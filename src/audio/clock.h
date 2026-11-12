// Horloge audio : publiée par le thread audio à chaque mesure du périphérique, lue par present et gpu quand
// l'audio est maître (VRR). Un std::mutex gardant trois copies de mots : la section est brève, mais un lecteur
// préempté dedans ferait attendre le thread audio jusqu'à sa reprise, sans héritage de priorité (C16-07). Un
// seqlock lèverait cette attente ; tant qu'aucune mesure n'en montre le besoin, le verrou reste.
#pragma once

#include <mutex>

#include "core/clock.h"

namespace ajp::audio {

class AudioClock final : public MasterClock {
 public:
  // Thread audio seulement.
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

}  // namespace ajp::audio
