// Grille du Rapide à pas constant (chantier 7.5, core/steady_rate.h) : N images de sortie par image source,
// N constant dans un plan. Enveloppe CadenceAccumulator : les instants d'une paire (g, d) sont g + j·(d −
// g)/N exacts (au µs près, sans dérive sur des heures), la source droite comprise quand elle tombe sur la
// grille. Le facteur ne change qu'à la demande (SteadyRate::take_changed) : retarget garde la phase, le
// dernier instant émis reste où il est et le suivant tombe un pas du nouveau facteur plus loin.
//
// À N = 1, chaque paire n'émet que sa source droite (aucune interpolation) ; en remontant au-dessus de 1, la
// grille repart de la source gauche. À fréquence fixe, SteadyRate ne propose que des facteurs qui calent
// l'écran : la grille uniforme tombe d'elle-même sur les vsyncs, sans calage de plus.
#pragma once

#include <cstdint>
#include <expected>
#include <optional>

#include "core/cadence.h"

namespace ajp::core {

class SteadyGrid {
 public:
  // La source suivante, dans l'ordre de présentation, au facteur `factor` (1 ≤ factor ≤ kMaxInstantsPerPair ;
  // hors bornes : ramené à 1).
  CadenceStep push(const SourceStamp& source, Ratio factor);

  // La paire n'a pas été interpolée (coupe, dessin tenu, moteur absent) : la grille repart de la source
  // droite.
  void resync();
  // Seek ou changement de plan : la prochaine source redevient l'origine.
  void reset();
  std::expected<void, Refusal> flush(std::uint32_t epoch);

  // Le facteur de la dernière paire émise.
  Ratio factor() const { return factor_; }

 private:
  CadenceStep alone(const SourceStamp& source, GridEvent event) const;

  std::optional<CadenceAccumulator> grid_;  // absent à N = 1
  std::optional<SourceStamp> left_;
  Ratio factor_{1, 1};
  std::uint32_t epoch_ = 0;
  bool started_ = false;
};

}  // namespace ajp::core
