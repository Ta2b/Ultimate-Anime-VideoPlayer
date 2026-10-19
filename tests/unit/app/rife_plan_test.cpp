// Plan d'interpolation d'une paire : quand RIFE tourne, et les instants des sorties.
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

TEST_CASE("RIFE : une paire qui bouge donne ×N sorties, b en dernier", "[app][rife]") {
  CHECK(rife_outputs(moving(4)) == 4);
  CHECK(rife_outputs(moving(2)) == 2);
  CHECK(rife_output_pts(0, 41'708, 0, 4) == 10'427);
  CHECK(rife_output_pts(0, 41'708, 1, 4) == 20'854);
  CHECK(rife_output_pts(0, 41'708, 2, 4) == 31'281);
  CHECK(rife_output_pts(0, 41'708, 3, 4) == 41'708);  // b
  CHECK(rife_output_pts(1'000, 2'000, 0, 1) == 2'000);
}

TEST_CASE("RIFE : b sort seul sur coupe, dessin tenu, trou, nouvelle époque ou ×1", "[app][rife]") {
  PairFacts cut = moving(4);
  cut.scores->scene_score = kernels::kSceneCutThreshold + 0.01;
  CHECK(rife_outputs(cut) == 1);

  PairFacts held = moving(4);
  held.scores->worst_block = 100;  // 0,39 niveau : sous le seuil de 0,70
  CHECK(rife_outputs(held) == 1);

  PairFacts gap = moving(4);
  gap.gap_us = kMaxPairGapUs + 1;
  CHECK(rife_outputs(gap) == 1);
  gap.gap_us = 0;
  CHECK(rife_outputs(gap) == 1);

  PairFacts first = moving(4);
  first.same_epoch = false;
  CHECK(rife_outputs(first) == 1);

  PairFacts unmeasured = moving(4);
  unmeasured.scores.reset();
  CHECK(rife_outputs(unmeasured) == 1);

  CHECK(rife_outputs(moving(1)) == 1);
  CHECK(rife_outputs(moving(kMaxRifeFactor + 1)) == 1);
}

}  // namespace ajp::app::play
