#include "app/play/drba_fused_choice.h"

namespace ajp::app::play {

std::string drba_fused_declined(const PlaySettings& settings, bool backend_does_it) {
  if (!settings.drba_fused) return "réglage éteint";
  if (settings.motion_engine != MotionEngine::DistilDrba) return "moteur du mouvement autre que DistilDRBA";
  if (!backend_does_it) return "ce moteur ne le fait pas";
  if (settings.motion_precision != PrecisionTier::Balanced) return "cran de précision hors Équilibre";
  return {};
}

}  // namespace ajp::app::play
