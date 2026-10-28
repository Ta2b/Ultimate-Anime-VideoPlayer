// Horloge maître de la lecture. Avec VRR, l'audio la publie ; sur un écran à
// rafraîchissement fixe, present la publie et l'audio la suit en se
// rééchantillonnant. present et gpu la lisent sans attendre.
// Fait partie de core_types : aucune dépendance hors bibliothèque standard.
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>

namespace ajp {

// Instant hôte en microsecondes, sur l'horloge monotone commune à tous les threads.
inline std::int64_t host_now_us() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

// Point de référence d'une horloge : le temps média vaut media_us à l'instant
// hôte host_us, puis avance de rate microsecondes média par microseconde hôte.
struct ClockSnapshot {
  std::int64_t media_us = 0;
  std::int64_t host_us = 0;
  double rate = 0.0;        // 0 en pause, 1 à vitesse nominale
  std::uint32_t epoch = 0;  // époque du Flush dont vient la position
  bool valid = false;       // false : position inconnue depuis le dernier Flush

  // Temps média extrapolé à l'instant hôte host.
  std::int64_t media_at(std::int64_t host) const {
    return media_us + std::llround(static_cast<double>(host - host_us) * rate);
  }
};

// Ce que lisent present et gpu, quel que soit le maître. Un seul thread écrit.
class MasterClock {
 public:
  virtual ~MasterClock() = default;
  virtual ClockSnapshot snapshot() const = 0;
};

}  // namespace ajp
