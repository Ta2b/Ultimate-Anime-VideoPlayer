// Calage de phase de la vitesse sur un écran fixe (A12-06). Pur.
//
// core/pacing.h cale la fréquence : la vitesse qui fait tenir à chaque image un nombre entier de vsyncs.
// Mais la fréquence de l'écran n'est connue qu'à quelques millièmes de hertz près (0,002 Hz à 144 Hz : 14
// millionièmes), et rien ne rattrapait le reste : l'échéance des images glissait lentement sur la grille des
// vblanks, et une image était doublée ou sautée chaque fois qu'elle passait le milieu de deux vblanks (une
// toutes les huit minutes à 144 Hz).
//
// Cette boucle ramène l'échéance des images sur le vblank : chaque image neuve donne sa position sur la
// grille mesurée, en fraction de période (0 : pile sur un vblank ; ±0,5 : au milieu de deux), et la vitesse
// reçoit une correction proportionnelle à cette position lissée, bornée à kMaxTrim. Elle ne change que par
// pas de kStep et au plus une fois par kMinGapUs : l'audio, qui suit la vitesse par rééchantillonnage, ne
// reçoit pas un ordre par image.
//
// Une image en retard sur son vblank (position positive) demande d'aller plus vite : correction positive.
// Régime : pour une erreur de fréquence e, la position se stabilise à e / kGain (14 millionièmes : 0,07
// période), loin du milieu ; constante de temps d'une période / kGain (35 s à 144 Hz).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ajp::core {

class PhaseTrim {
 public:
  static constexpr double kGain = 200e-6;     // correction par période d'écart
  static constexpr double kMaxTrim = 100e-6;  // ±0,01 % : sous la borne de la synchro (pacing.h)
  static constexpr double kStep = 2e-6;       // pas de la correction publiée
  static constexpr double kSmoothing = 0.05;  // lissage de la position, par image
  static constexpr std::int64_t kMinGapUs = 1'000'000;
  static constexpr int kMinSamples = 24;  // positions vues avant la première correction

  // La position d'une image neuve sur la grille des vblanks : son numéro fractionnaire (VblankGrid::index_of
  // de l'instant où son pts tombe sur l'horloge). Rend vrai quand la correction publiée change.
  bool note(double vblank_index, std::int64_t now_us) {
    const double position = vblank_index - std::round(vblank_index);  // [−0,5 ; 0,5]
    if (samples_ == 0) {
      position_ = position;
    } else {
      // L'écart le plus court : une position qui passe de +0,49 à −0,49 n'a bougé que de 0,02.
      double delta = position - position_;
      delta -= std::round(delta);
      position_ += kSmoothing * delta;
      position_ -= std::round(position_);
    }
    samples_ = std::min(samples_ + 1, kMinSamples);
    if (samples_ < kMinSamples || now_us - changed_us_ < kMinGapUs) return false;
    const double wanted = std::clamp(kGain * position_, -kMaxTrim, kMaxTrim);
    const double stepped = std::round(wanted / kStep) * kStep;
    if (stepped == trim_) return false;
    trim_ = stepped;
    changed_us_ = now_us;
    return true;
  }

  // La correction de vitesse à ajouter à celle du calage (0 : aucune).
  double trim() const { return trim_; }
  // La position lissée, en fraction de période.
  double position() const { return position_; }
  // Autre écran, autre cadence, seek, calage coupé : tout repart.
  void reset() { *this = PhaseTrim{}; }

 private:
  double position_ = 0.0;
  double trim_ = 0.0;
  int samples_ = 0;
  std::int64_t changed_us_ = -kMinGapUs;
};

}  // namespace ajp::core
