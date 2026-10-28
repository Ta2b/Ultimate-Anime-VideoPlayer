#include "app/play/rife_plan.h"

#include "kernels/spec/detect.h"

namespace ajp::app::play {

bool rife_runs(const PairFacts& facts) {
  if (facts.factor <= 1 || facts.factor > kMaxRifeFactor || !facts.same_epoch) return false;
  if (facts.gap_us <= 0 || facts.gap_us > kMaxPairGapUs || !facts.scores) return false;
  return !kernels::is_scene_cut(facts.scores->scene_score) && !kernels::is_held(facts.scores->worst_block);
}

}  // namespace ajp::app::play
