// Horloge maître de la lecture. D'ordinaire l'audio la publie, sur tout écran ;
// sous calage sur l'écran (ou sans son), present la publie et l'audio la suit
// en se rééchantillonnant. present et gpu la lisent sans attendre.
// Fait partie de core_types : aucune dépendance hors bibliothèque standard.
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

namespace kirari {

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
    const auto here = static_cast<std::uint64_t>(host), origin = static_cast<std::uint64_t>(host_us);
    const double delta =
        host >= host_us ? static_cast<double>(here - origin) : -static_cast<double>(origin - here);
    const double offset = std::round(delta * rate);
    constexpr double limit = 0x1p63;
    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if (std::isnan(offset)) return media_us;
    if (offset >= limit || offset < -limit) {
      const double sum = static_cast<double>(media_us) + offset;
      if (sum >= limit) return maximum;
      if (sum <= -limit) return minimum;
      return static_cast<std::int64_t>(sum);
    }
    const auto step = static_cast<std::int64_t>(offset);
    if (step > 0 && media_us > maximum - step) return maximum;
    if (step < 0 && media_us < minimum - step) return minimum;
    return media_us + step;
  }
};

// Ce que lisent present et gpu, quel que soit le maître. Un seul thread écrit.
class MasterClock {
 public:
  virtual ~MasterClock() = default;
  virtual ClockSnapshot snapshot() const = 0;
};

}  // namespace kirari
