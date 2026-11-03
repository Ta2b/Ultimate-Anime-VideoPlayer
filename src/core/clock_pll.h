// Boucle à verrouillage de phase du second ordre pour l'horloge audio (étape 6.9). Pure.
//
// Chaque lecture donne l'écart brut entre l'horloge audio et la référence ; la boucle le filtre
// (Kp 0,1, Ki 0,005 par lecture), suit une dérive constante sans erreur, recale d'un coup au-delà
// de 20 ms, et propose une correction de vitesse bornée à ±0,05 %.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ajp::core {

class ClockPll {
 public:
  static constexpr double kKp = 0.1;
  static constexpr double kKi = 0.005;
  static constexpr double kStepUs = 20'000.0;
  static constexpr double kMaxSpeed = 0.0005;

  struct Out {
    double offset_us = 0.0;  // l'écart filtré
    double speed = 1.0;      // facteur de vitesse proposé
  };

  // measured_us : écart brut de cette lecture ; interval_us : temps écoulé depuis la précédente.
  Out update(double measured_us, double interval_us) {
    if (!locked_ || std::abs(measured_us - phase_) > kStepUs) {
      phase_ = measured_us;
      rate_ = 0.0;
      locked_ = true;
    } else {
      const double error = measured_us - phase_;
      phase_ += kKp * error + rate_;
      rate_ += kKi * error;
    }
    const double speed = interval_us > 0.0 ? rate_ / interval_us : 0.0;
    return {.offset_us = phase_, .speed = 1.0 + std::clamp(speed, -kMaxSpeed, kMaxSpeed)};
  }
  void reset() { locked_ = false; }

 private:
  double phase_ = 0.0;
  double rate_ = 0.0;  // µs de dérive par lecture
  bool locked_ = false;
};

}  // namespace ajp::core
