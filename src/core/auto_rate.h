// Interpolation en mode auto : le multiplicateur ×k (k entier, de ×1 à ×6) suit ce que le GPU tient.
// Il descend vite quand une paire arrive en retard sur l'horloge, remonte lentement quand les
// paires passent, et un palier qui vient d'échouer se re-tente de plus en plus tard (hystérésis :
// pas d'oscillation entre k et k+1). Pur : le lecteur lui donne l'issue de chaque paire.
#pragma once

#include <cstdint>

namespace ajp::core {

inline constexpr int kAutoMaxStep = 6;  // ×6 de 23,976 : 143,86 i/s, sous un écran de 165 Hz

// Le plus grand palier dont la cadence tient sous la fréquence de l'écran (à 1 % près) pour une source
// de `source_fps` i/s ; 1 si l'écran est inconnu. Sans VRR, les paliers qui ne tiennent pas un nombre
// entier de rafraîchissements font du judder : le menu l'avertit, il ne les interdit pas.
int auto_ceiling(double display_hz, double source_fps);

class AutoRate {
 public:
  // Paires propres avant de tenter un palier de plus (3 s à 24 i/s), doublé à chaque échec récent.
  static constexpr std::uint32_t kRisePairs = 72;
  static constexpr int kMaxBackoff = 4;
  // Un palier qui échoue encore au back-off maximal devient le plafond appris : on ne le retente plus
  // qu'après 5 min (7200 paires à 24 i/s), au lieu d'un à-coup toutes les 48 s.
  static constexpr std::uint32_t kCapPairs = 7200;
  // Paires ignorées après une descente ou un saut : celles déjà en file ont été faites
  // au palier d'avant et reviendraient en retard à tort (descente en cascade).
  static constexpr std::uint32_t kGracePairs = 6;

  // `ceiling` : le palier le plus haut permis (1 à kAutoMaxStep). Le départ est ×2 (ou le plafond).
  explicit AutoRate(int ceiling = kAutoMaxStep);

  int step() const { return step_; }
  int learned_cap() const { return cap_; }  // 0 : aucun
  int ceiling() const { return ceiling_; }
  // Plafond changé (écran déplacé, réglage) : le palier courant le suit.
  void set_ceiling(int ceiling);

  // L'issue d'une paire : en retard (des instants sautés pour rattraper l'horloge) ou à l'heure.
  // Rend le palier à utiliser pour la paire suivante.
  int pair(bool late);
  // Seek, pause, reprise : la mesure repart après kGracePairs paires.
  void settle() { grace_ = kGracePairs; }

 private:
  int ceiling_;
  int step_;
  std::uint32_t grace_ = 0;  // paires à ignorer encore
  std::uint32_t clean_ = 0;  // paires à l'heure depuis le dernier changement
  int cap_ = 0;              // plafond appris sur ce contenu ; 0 : aucun
  int backoff_ = 0;          // échecs récents de montée
  bool probing_ = false;     // le palier courant vient d'être tenté
};

}  // namespace ajp::core
