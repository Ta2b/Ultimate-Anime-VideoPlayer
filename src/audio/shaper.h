// Dernier étage avant le matériel : gain à rampe, limiteur doux, entrée en fondu (B16-03, B16-07, B16-08).
//
// Posé dans le rappel postmix de SDL, sur le thread du périphérique, donc sans verrou ni allocation. Le
// gain change par une rampe de kRampMs au lieu d'une marche ; un son qui repart (reprise, vidage, file
// neuve) entre en fondu ; au-delà de 100 % de gain, ou quand SDL doit mixer vers un autre nombre de canaux
// que celui de la file, le limiteur doux de converter.h arrondit les crêtes au lieu de les écrêter.
#pragma once

#include <span>

namespace ajp::audio {

// Durée des rampes, en millisecondes.
inline constexpr int kRampMs = 10;

class OutputShaper {
 public:
  struct Pass {
    int channels = 2;
    int sample_rate = 48'000;
    float gain = 1.0F;         // la cible
    bool fade_in = false;      // ce tampon est le premier d'un son qui repart
    bool force_limit = false;  // limiter même sous un gain de 1
  };

  // Traite des trames entrelacées de float32, sur place.
  void process(std::span<float> samples, const Pass& pass);
  float applied_gain() const { return applied_; }

 private:
  float applied_ = 1.0F;  // dernier gain atteint
};

}  // namespace ajp::audio
