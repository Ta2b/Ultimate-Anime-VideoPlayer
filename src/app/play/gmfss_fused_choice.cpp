#include "app/play/gmfss_fused_choice.h"

namespace ajp::app::play {

std::string gmfss_fused_declined(const PlaySettings& settings, bool backend_does_it) {
  if (!settings.gmfss_fused) return "réglage éteint";
  if (!is_gmfss(settings.motion_engine)) return "moteur du mouvement autre que GMFSS";
  if (!backend_does_it) return "ce moteur ne le fait pas";
  if (settings.gmfss_precision != PrecisionTier::Balanced) return "cran de précision hors Équilibre";
  return {};
}

}  // namespace ajp::app::play
