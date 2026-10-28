// Plan d'interpolation d'une paire : quand RIFE tourne.
#include "app/play/rife_plan.h"

#include <catch2/catch_test_macros.hpp>

#include "kernels/spec/detect.h"

namespace ajp::app::play {
namespace {

PairFacts moving(std::uint32_t factor) {
  return PairFacts{.factor = factor,
                   .same_epoch = true,
                   .gap_us = 41'708,
                   .scores = gpu::PairScores{.scene_score = 0.02, .worst_block = 2000}};
}

}  // namespace

TEST_CASE("RIFE : tourne sur une paire qui bouge", "[app][rife]") {
  CHECK(rife_runs(moving(4)));
  CHECK(rife_runs(moving(2)));
}

TEST_CASE("RIFE : b sort seul sur coupe, dessin tenu, trou, nouvelle époque ou ×1", "[app][rife]") {
  PairFacts cut = moving(4);
  cut.scores->scene_score = kernels::kSceneCutThreshold + 0.01;
  CHECK_FALSE(rife_runs(cut));

  PairFacts held = moving(4);
  held.scores->worst_block = 100;  // 0,39 niveau : sous le seuil de 0,70
  CHECK_FALSE(rife_runs(held));

  PairFacts gap = moving(4);
  gap.gap_us = kMaxPairGapUs + 1;
  CHECK_FALSE(rife_runs(gap));
  gap.gap_us = 0;
  CHECK_FALSE(rife_runs(gap));

  PairFacts first = moving(4);
  first.same_epoch = false;
  CHECK_FALSE(rife_runs(first));

  PairFacts unmeasured = moving(4);
  unmeasured.scores.reset();
  CHECK_FALSE(rife_runs(unmeasured));

  CHECK_FALSE(rife_runs(moving(1)));
  CHECK_FALSE(rife_runs(moving(kMaxRifeFactor + 1)));
}

}  // namespace ajp::app::play
