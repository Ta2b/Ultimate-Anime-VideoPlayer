#include "app/play/drba_fused_choice.h"

#include <utility>

namespace ajp::app::play {

std::string drba_fused_declined(const PlaySettings& settings, bool backend_does_it) {
  if (!settings.drba_fused) return "réglage éteint";
  if (settings.motion_engine != MotionEngine::DistilDrba) return "moteur du mouvement autre que DistilDRBA";
  if (!backend_does_it) return "ce moteur ne le fait pas";
  return {};
}

DrbaTier drba_tier(PrecisionTier asked, bool fused_possible, TierSupport graphs) {
  if (asked == PrecisionTier::Quality && graphs.quality)
    return {.tier = PrecisionTier::Quality, .fused = false, .why = {}};
  if (asked == PrecisionTier::Performance && graphs.performance && !fused_possible)
    return {.tier = PrecisionTier::Performance, .fused = false, .why = {}};
  std::string why;
  if (asked == PrecisionTier::Performance && fused_possible) {
    why = "la chaîne fusionnée FP16 est plus rapide que le graphe FP8";
  } else if (asked != PrecisionTier::Balanced) {
    why = "graphe absent ou moteur sans FP8, Équilibre à la place";
  }
  return {.tier = PrecisionTier::Balanced, .fused = fused_possible, .why = std::move(why)};
}

}  // namespace ajp::app::play
