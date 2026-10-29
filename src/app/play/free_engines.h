// Ce que chaque moteur d'interpolation accepte du mode Auto libre (docs/auto-libre.md) : le nombre
// d'instants qu'une paire peut recevoir, chacun à un tau arbitraire de ]0, 1[.
//
// - RIFE (entier, fusionné, variante mouvement) : une entrée par instant, le tau est un plan de l'entrée (ou
//   le facteur du warp) ; la grille libre n'a aucune contrainte.
// - DistilDRBA : drba_call(t) prend n'importe quel t de ]0, 1[ (fenêtre centrée sur a avant le milieu, sur b
//   après) ; une passe du graphe par instant.
// - GMFSS (entier ou fusionné, base ou union) : le graphe « paire » tourne une fois par paire, puis chaque
//   instant est un splat à gmfss_side_time(côté, t) et un graphe « fusion » ; t est arbitraire.
// - Au plus kMaxInstantsPerPair instants par paire pour tous : c'est la borne du ring
// (RingStage::kMaxOutputs).
// - Apple (flux optique du système) : instants fixés par le système, pas de tau arbitraire : Auto.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "app/play/settings.h"
#include "core/cadence.h"

namespace ajp::app::play {

// Le nombre maximal d'instants par paire pour ce moteur ; 0 : le moteur ne calcule pas à un tau arbitraire,
// le mode libre y replie sur Auto.
constexpr std::uint32_t free_instants_limit(MotionEngine engine) {
  switch (engine) {
    case MotionEngine::Rife:
    case MotionEngine::Gmfss:
    case MotionEngine::GmfssUnion:
    case MotionEngine::DistilDrba:
      return static_cast<std::uint32_t>(core::kMaxInstantsPerPair);
    case MotionEngine::Apple:
      return 0;
  }
  return 0;
}

}  // namespace ajp::app::play
