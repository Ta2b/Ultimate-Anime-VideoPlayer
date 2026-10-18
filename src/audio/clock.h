// Horloge audio : publiée par le thread audio à chaque mesure du périphérique,
// lue sans attente par present et gpu quand l'audio est maître (VRR).
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
