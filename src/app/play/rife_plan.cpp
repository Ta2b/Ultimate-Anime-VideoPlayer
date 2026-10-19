#include "app/play/rife_plan.h"

#include "kernels/spec/detect.h"

namespace ajp::app::play {

std::uint32_t rife_outputs(const PairFacts& facts) {
  if (facts.factor <= 1 || facts.factor > kMaxRifeFactor || !facts.same_epoch) return 1;
  if (facts.gap_us <= 0 || facts.gap_us > kMaxPairGapUs || !facts.scores) return 1;
  if (kernels::is_scene_cut(facts.scores->scene_score) || kernels::is_held(facts.scores->worst_block))
    return 1;
  return facts.factor;
}

std::int64_t rife_output_pts(std::int64_t pts_a, std::int64_t pts_b, std::uint32_t k, std::uint32_t outputs) {
  if (outputs <= 1 || k + 1 >= outputs) return pts_b;
  return pts_a + (pts_b - pts_a) * static_cast<std::int64_t>(k + 1) / static_cast<std::int64_t>(outputs);
}

}  // namespace ajp::app::play
