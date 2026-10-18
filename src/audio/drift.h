// Régulateur de dérive : quand l'écran est maître, l'audio le suit en lisant
// sa file un peu plus vite ou plus lentement (SDL_SetAudioStreamFrequencyRatio).
//
// Régulateur proportionnel-intégral sur l'écart filtré. Le terme intégral
// apprend la dérive constante (1,001 en NTSC) ; la correction reste bornée à
// limits::kMaxDriftCorrection, et l'intégrale ne grossit plus quand elle sature.
#pragma once

#include <cstdint>

namespace ajp::audio {

struct DriftGains {
  double proportional = 0.1;  // par seconde d'écart : 10 ms d'écart donnent 0,1 %
  double integral = 0.005;    // par seconde² : amortissement proche de 0,7
  double smoothing_s = 1.0;   // constante de temps du filtre des mesures
};

class DriftRegulator {
 public:
  explicit DriftRegulator(DriftGains gains = {}) : gains_(gains) {}

  // error_us : temps audio moins temps maître (> 0 : l'audio est en avance).
  // elapsed_us : depuis la mesure précédente. Rend le rapport de fréquence à
  // appliquer, compris dans 1 ± limits::kMaxDriftCorrection.
  double update(std::int64_t error_us, std::int64_t elapsed_us);
  double ratio() const { return ratio_; }
  // Après un recalage : oublie le filtre, garde la dérive apprise.
  void restart() { primed_ = false; }
  // Changement d'horloge maître : oublie tout.
  void reset();

 private:
  DriftGains gains_;
  double filtered_s_ = 0.0;
  double integral_ = 0.0;  // secondes × secondes
  double ratio_ = 1.0;
  bool primed_ = false;
};

}  // namespace ajp::audio
