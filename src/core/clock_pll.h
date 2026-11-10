// Boucle à verrouillage de phase du second ordre pour l'horloge audio (étape 6.9). Pure.
//
// Chaque lecture donne l'écart brut entre l'horloge audio et la référence ; la boucle le filtre, suit une
// dérive constante sans erreur, recale d'un coup au-delà de 20 ms, et propose une correction de vitesse
// bornée à ±0,05 %.
//
// La boucle est réglée en temps, pas en lectures (A12-11) : la dérive se compte en µs par µs et les gains
// (Kp 0,1 et Ki 0,005 pour une lecture toutes les 20 ms) s'appliquent au prorata de l'intervalle. Une
// lecture perdue ou un périphérique plus lent (quantum PipeWire de 21 ms contre 10 ms sous WASAPI) ne
// changent ni la vitesse proposée ni la constante de temps.
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

  // L'intervalle de lecture pour lequel kKp et kKi sont donnés.
  static constexpr double kReferenceUs = 20'000.0;
  // Au-delà, l'intervalle compte pour cela : un gain proportionnel au-dessus de 1 dépasserait la mesure.
  static constexpr double kMaxIntervalUs = 5.0 * kReferenceUs;

  // measured_us : écart brut de cette lecture ; interval_us : temps écoulé depuis la précédente (<= 0 :
  // inconnu, compté pour kReferenceUs).
  Out update(double measured_us, double interval_us) {
    if (!locked_ || std::abs(measured_us - phase_) > kStepUs) {
      phase_ = measured_us;
      rate_ = 0.0;
      locked_ = true;
    } else {
      const double dt = interval_us > 0.0 ? std::min(interval_us, kMaxIntervalUs) : kReferenceUs;
      const double share = dt / kReferenceUs;
      // La phase avance d'abord de la dérive suivie pendant dt, puis reçoit sa part de l'erreur restante.
      phase_ += rate_ * dt;
      const double error = measured_us - phase_;
      phase_ += std::min(kKp * share, 1.0) * error;
      rate_ += kKi * share * error / kReferenceUs;
    }
    return {.offset_us = phase_, .speed = 1.0 + std::clamp(rate_, -kMaxSpeed, kMaxSpeed)};
  }
  void reset() { locked_ = false; }

 private:
  double phase_ = 0.0;
  double rate_ = 0.0;  // dérive suivie, en µs par µs
  bool locked_ = false;
};

}  // namespace ajp::core
