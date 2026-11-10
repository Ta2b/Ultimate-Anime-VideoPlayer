// Phase des vsyncs d'un écran à fréquence fixe, suivie sur des instants où un vsync a été vu (µs, horloge
// hôte). Pur. Un instant n'est une mesure que si l'écran l'a imposé : le retour d'une acquisition qui a
// attendu la libération d'une image (file FIFO pleine). L'heure où une présentation est soumise ou rendue par
// le pilote n'en est pas une : elle suit l'heure choisie par la boucle, elle-même tirée de cette phase, et la
// phase se nourrirait d'elle-même sans jamais voir l'écran (A12-03).
//
// La position de chaque instant modulo la période est un point sur un cercle ; leur moyenne exponentielle
// (vecteur) donne la phase, et sa longueur dit s'ils s'accordent. La phase n'est connue (known) qu'après
// kMinNotes instants qui s'accordent : des instants quelconques, répartis sur la période, ne la rendent
// jamais connue. Elle suit la dérive de l'écran et de l'horloge sans accumuler d'erreur.
#pragma once

#include <cstdint>

namespace ajp::core {

class VsyncPhase {
 public:
  static constexpr double kSmoothing = 0.08;
  // Assez d'instants pour que la moyenne ait presque tout son poids (1 − 0,92²⁴ = 0,86). À 8, elle ne
  // pesait que huit points : des instants au hasard passaient le seuil d'accord pour 4 tirages sur 2 000, un
  // ou deux instants de suite au départ ; à 24, aucun sur 2 000 tirages de dix minutes. Le prix : la phase
  // d'un écran à 60 Hz se connaît après 0,4 s de vsyncs vus au lieu de 0,13 s.
  static constexpr int kMinNotes = 24;
  // Longueur du vecteur moyen au-dessus de laquelle les instants s'accordent : 0,8 vaut un écart type
  // d'environ un dixième de période ; des instants au hasard donnent 0,2 à 0,4.
  static constexpr double kAgreement = 0.8;

  // La période a changé (autre écran, autre mode) : la phase repart.
  void set_period(std::int64_t period_us);
  std::int64_t period_us() const { return period_us_; }
  // Un vsync vu à host_us.
  void note(std::int64_t host_us);
  bool known() const;
  // Le vsync (instant hôte) le plus proche de host_us ; host_us si la phase est inconnue.
  std::int64_t nearest(std::int64_t host_us) const;  // pas « near » : une macro de windows.h

 private:
  double phase_us() const;  // modulo period_us_, dans [0, period)
  std::int64_t period_us_ = 0;
  double cos_ = 0.0;  // moyenne exponentielle du point du cercle
  double sin_ = 0.0;
  double weight_ = 0.0;  // poids reçu par la moyenne : corrige son départ de zéro
  int notes_ = 0;
};

}  // namespace ajp::core
