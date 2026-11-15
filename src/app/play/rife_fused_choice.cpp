#include "app/play/rife_fused_choice.h"

#include "kernels/spec/rife.h"

namespace ajp::app::play {

std::string rife_fused_declined(const PlaySettings& settings, bool backend_does_it, int rife_planes,
                                PrecisionTier tier) {
  if (!settings.rife_fused) return "réglage éteint";
  // Très élevé (GMFSS au milieu de chaque paire) fait ses autres instants par RIFE : le fusionné aussi.
  if (settings.motion_engine != MotionEngine::Rife && settings.motion_engine != MotionEngine::Gmfss)
    return "moteur du mouvement autre que RIFE";
  if (!backend_does_it) return "ce moteur ne le fait pas";
  if (tier != PrecisionTier::Balanced) return "cran de précision hors Équilibre";
  if (rife_planes != kernels::kRifeInputChannels) return "modèle à entrée v2";
  return {};
}

}  // namespace ajp::app::play
