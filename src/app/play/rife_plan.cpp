#include "app/play/rife_plan.h"

#include "kernels/spec/detect.h"

namespace ajp::app::play {

bool pair_cut(const PairFacts& facts) {
  if (facts.known_cut) {
    // Une coupe des plans sur deux images presque identiques est un effet (éclair, fumée) : pas de coupe.
    if (*facts.known_cut && facts.scores && facts.scores->scene_score < kPlanCutFloor) return false;
    return *facts.known_cut;
  }
  return facts.scores && kernels::is_scene_cut(facts.scores->scene_score);
}

bool rife_runs(const PairFacts& facts) {
  if (facts.factor <= 1 || facts.factor > kMaxRifeFactor || !facts.same_epoch) return false;
  if (facts.gap_us <= 0 || facts.gap_us > kMaxPairGapUs || !facts.scores) return false;
  return !pair_cut(facts) && !kernels::is_held(facts.scores->worst_block);
}

}  // namespace ajp::app::play
