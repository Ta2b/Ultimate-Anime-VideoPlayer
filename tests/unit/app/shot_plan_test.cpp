// L'Auto plan par plan (app/play/shot_plan.h) : le choix de chaque plan sur son rythme, l'héritage des plans
// trop courts ou sans coupe devant, et le livre que lit l'étage gpu.
#include "app/play/shot_plan.h"

#include <memory>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using namespace kirari::app::play;
using kirari::media::Shot;
using kirari::media::ShotRhythm;

namespace {

Shot shot(std::int64_t start_ms, std::int64_t end_ms, ShotRhythm rhythm, bool cut_before = true) {
  Shot s;
  s.start_us = start_ms * 1000;
  s.end_us = end_ms * 1000;
  s.cut_before = cut_before;
  s.rhythm = rhythm;
  return s;
}

}  // namespace

TEST_CASE("plan par plan : choix, héritage et livre", "[app][shots]") {
  // Animation limitée en deux, sans grain : DistilDRBA, sans TemporalFix.
  const ShotRhythm limited{.pairs = 48, .held = 24, .strong = 1, .grain = 0.4F, .still = 0.8F};
  // Sakuga en un : RIFE ; grain visible sur un décor immobile : TemporalFix.
  const ShotRhythm sakuga{.pairs = 48, .held = 2, .strong = 20, .grain = 1.5F, .still = 0.5F};
  // Un éclair de trois images : trop court pour se juger.
  const ShotRhythm flash{.pairs = 3, .held = 0, .strong = 3};
  // Un dessin tenu en deux mais rapide reste à RIFE.
  const ShotRhythm fast_twos{.pairs = 48, .held = 24, .strong = 12};

  CHECK(judge_shot(limited) == ShotChoice{.motion = ShotMotion::Drba, .temporal = false});
  CHECK(judge_shot(sakuga) == ShotChoice{.motion = ShotMotion::Rife, .temporal = true});
  CHECK(judge_shot(fast_twos).motion == ShotMotion::Rife);

  const auto shots = std::make_shared<const std::vector<Shot>>(
      std::vector<Shot>{shot(0, 2000, limited), shot(2000, 2125, flash), shot(2125, 4000, sakuga),
                        shot(4000, 6000, limited, false),  // reprise d'analyse : le plan réel continue
                        shot(8000, 9000, flash)});         // après un trou, trop court : le choix par défaut
  const std::vector<ShotChoice> plan = plan_shots(*shots);
  REQUIRE(plan.size() == 5);
  CHECK(plan[0].motion == ShotMotion::Drba);
  CHECK(plan[1].motion == ShotMotion::Drba);  // l'éclair garde le plan d'avant
  CHECK_FALSE(plan[1].judged);
  CHECK(plan[2].motion == ShotMotion::Rife);
  CHECK(plan[3].motion == ShotMotion::Rife);  // sans coupe devant : jamais au milieu d'un plan
  CHECK_FALSE(plan[3].judged);
  CHECK(plan[4] == ShotChoice{.judged = false});

  ShotPlanBook book;
  CHECK_FALSE(book.active());
  CHECK_FALSE(book.at(1'000'000));
  book.set_shots(shots);
  CHECK(book.active());
  CHECK(book.at(1'000'000)->motion == ShotMotion::Drba);
  CHECK(book.at(3'000'000)->motion == ShotMotion::Rife);
  CHECK_FALSE(book.at(7'000'000));  // le trou
  CHECK(book.cut_between(1'990'000, 2'010'000) == true);
  CHECK(book.cut_between(3'990'000, 4'010'000) == std::nullopt);  // plan sans coupe connue
  CHECK_FALSE(book.describe(1'000'000).empty());
}
