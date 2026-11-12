// Dernier étage avant le matériel : gain à rampe, limiteur doux, entrée en fondu (B16-03, B16-07, B16-08).
//
// Posé dans le rappel postmix de SDL, sur le thread du périphérique, donc sans verrou ni allocation. Le
// gain change par une rampe de kRampMs au lieu d'une marche ; un son qui repart (reprise, vidage, file
// neuve) entre en fondu ; au-delà de 100 % de gain, ou quand SDL doit mixer vers un autre nombre de canaux
// que celui de la file, le limiteur doux de converter.h arrondit les crêtes au lieu de les écrêter.
#pragma once

#include <cstddef>
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

  // Traite des trames entrelacées de float32, sur place. Une rampe ou un fondu plus long que le tampon
  // continue dans le suivant (C16-03) : la position dans la rampe est gardée, rien ne repart de la cible.
  void process(std::span<float> samples, const Pass& pass);
  float applied_gain() const { return applied_; }

 private:
  float applied_ = 1.0F;       // le gain de la dernière trame traitée
  float target_ = 1.0F;        // la cible de la rampe en cours
  float ramp_from_ = 1.0F;     // d'où elle est partie
  std::size_t ramp_done_ = 0;  // trames déjà faites de la rampe de gain
  std::size_t fade_done_ = 0;  // trames déjà faites du fondu d'entrée
  bool fading_ = false;
};

}  // namespace ajp::audio
