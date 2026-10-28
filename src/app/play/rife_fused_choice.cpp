#include "app/play/rife_fused_choice.h"

#include "kernels/spec/rife.h"

namespace ajp::app::play {

std::string rife_fused_declined(const PlaySettings& settings, bool backend_does_it, int rife_planes) {
  if (!settings.rife_fused) return "réglage éteint";
  if (settings.motion_engine != MotionEngine::Rife) return "moteur du mouvement autre que RIFE";
  if (!backend_does_it) return "ce moteur ne le fait pas";
  if (settings.motion_precision != PrecisionTier::Balanced) return "cran de précision hors Équilibre";
  if (rife_planes != kernels::kRifeInputChannels) return "modèle à entrée v2";
  return {};
}

}  // namespace ajp::app::play
