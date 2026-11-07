#include "app/play/rife_plan.h"

#include <algorithm>

#include "kernels/spec/detect.h"

namespace ajp::app::play {

bool pair_cut(const PairFacts& facts) {
  const bool measured = facts.scores && kernels::is_scene_cut(facts.scores->scene_score /
                                                              std::clamp(facts.useful_area, 0.25, 1.0));
  if (facts.known_cut && *facts.known_cut) {
    // Une coupe des plans sur deux images presque identiques est un effet (éclair, fumée) : pas de coupe.
    return !facts.scores || facts.scores->scene_score >= kPlanCutFloor;
  }
  // P3.31 : « même plan » selon la pré-analyse ne l'emporte plus sur une coupe franche mesurée par RIFE.
  return measured;
}

bool rife_runs(const PairFacts& facts) {
  if (facts.factor <= 1 || facts.factor > kMaxRifeFactor || !facts.same_epoch) return false;
  if (facts.gap_us <= 0 || facts.gap_us > kMaxPairGapUs || !facts.scores) return false;
  return !pair_cut(facts) && !kernels::is_held(facts.scores->worst_block);
}

}  // namespace ajp::app::play
