// Phase des vsyncs d'un écran à fréquence fixe, suivie sur les instants de présentation (µs, horloge hôte).
// Pur. Chaque présentation neuve tombe sur un vsync (file FIFO) : sa position modulo la période donne la
// phase, lissée en cercle (moyenne exponentielle de l'écart ramené à ±période/2) pour ne pas suivre la gigue
// de la boucle. La phase suit la dérive de l'écran et de l'horloge sans accumuler d'erreur : elle est
// recalée à chaque présentation.
#pragma once

#include <cstdint>

namespace ajp::core {

class VsyncPhase {
 public:
  static constexpr double kSmoothing = 0.08;

  // La période a changé (autre écran, autre mode) : la phase repart.
  void set_period(std::int64_t period_us);
  std::int64_t period_us() const { return period_us_; }
  // Une présentation neuve à host_us.
  void note(std::int64_t host_us);
  bool known() const { return known_; }
  // Le vsync (instant hôte) le plus proche de host_us ; host_us si la phase est inconnue.
  std::int64_t near(std::int64_t host_us) const;

 private:
  std::int64_t period_us_ = 0;
  double phase_us_ = 0.0;  // modulo period_us_, dans [0, period)
  bool known_ = false;
};

}  // namespace ajp::core
