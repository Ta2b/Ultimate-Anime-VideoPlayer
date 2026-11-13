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

TEST_CASE("RIFE : poses_copy, a tient jusqu'à 1/2 puis b paraît, seulement après un dessin tenu",
          "[app][rife]") {
  CHECK(poses_copy(true, true, 0.25F) == PoseCopy::Before);
  CHECK(poses_copy(true, true, 0.49F) == PoseCopy::Before);
  CHECK(poses_copy(true, true, 0.5F) == PoseCopy::After);
  CHECK(poses_copy(true, true, 1.0F) == PoseCopy::After);
  CHECK(poses_copy(false, true, 0.25F) == PoseCopy::None);
  CHECK(poses_copy(true, false, 0.75F) == PoseCopy::None);
}

}  // namespace ajp::app::play

namespace ajp::app::play {

TEST_CASE("RIFE : une coupe si les plans ou la mesure la voient (P3.31)", "[app][rife]") {
  PairFacts measured_cut = moving(4);
  measured_cut.scores->scene_score = kernels::kSceneCutThreshold + 0.01;
  measured_cut.known_cut = false;  // une coupe franche que la pré-analyse a manquée : RIFE la voit
  CHECK_FALSE(rife_runs(measured_cut));
  PairFacts missed = moving(4);
  missed.known_cut = true;  // une coupe douce que RIFE ne voit pas
  missed.scores->scene_score = 0.13;
  CHECK_FALSE(rife_runs(missed));
  CHECK(pair_cut(missed));
  PairFacts effect = moving(4);
  effect.known_cut = true;  // les plans coupent sur deux images presque identiques : un effet
  effect.scores->scene_score = kPlanCutFloor - 0.01;
  CHECK_FALSE(pair_cut(effect));
  CHECK(rife_runs(effect));
}

}  // namespace ajp::app::play

namespace ajp::app::play {

TEST_CASE("RIFE : le score de coupe est rapporté à la zone hors bandes noires (P3.32)", "[app][rife]") {
  PairFacts scope = moving(4);
  scope.scores->scene_score = 0.10;  // une coupe en 2,39:1 : diluée par les bandes (×0,74)
  CHECK_FALSE(pair_cut(scope));
  scope.useful_area = 0.74;
  CHECK(pair_cut(scope));
}

TEST_CASE("RIFE : le plancher des plans se compare au score rapporté et ne défait pas une coupe mesurée",
          "[app][rife]") {
  // A09-22. En 2,39:1, une coupe connue des plans à 0,07 brut vaut 0,095 hors bandes : au-dessus du plancher.
  PairFacts scope = moving(4);
  scope.known_cut = true;
  scope.scores->scene_score = 0.07;
  CHECK_FALSE(pair_cut(scope));  // sans bandes, 0,07 reste un effet
  scope.useful_area = 0.74;
  CHECK(pair_cut(scope));
  // Zone utile de 0,64 : 0,08 brut vaut 0,125. Plans et mesure disent « coupe », la réponse aussi.
  PairFacts window = moving(4);
  window.known_cut = true;
  window.scores->scene_score = 0.08;
  window.useful_area = 0.64;
  CHECK(measured_cut(window));
  CHECK(pair_cut(window));
}

TEST_CASE(
    "RIFE : dans un plan continu selon les plans, une rafale de coupes mesurées n'en garde que la première",
    "[app][rife]") {
  // Le journal d'une séance : « plans : plan, RIFE : coupe » de 0,12 à 0,23 sur presque chaque paire.
  CutBurst burst;
  std::int64_t at_us = 1'000'000;
  constexpr std::int64_t kPair = 41'708;
  const auto pair_at = [&](double score) {
    PairFacts facts = moving(4);
    facts.known_cut = false;
    facts.scores->scene_score = score;
    facts.cut_burst = burst.after_cut(at_us);
    burst.note(at_us, measured_cut(facts));
    at_us += kPair;
    return pair_cut(facts);
  };
  CHECK(pair_at(0.18));         // la première : peut-être une coupe que la pré-analyse a manquée
  CHECK_FALSE(pair_at(0.15));   // les suivantes : du mouvement
  CHECK_FALSE(pair_at(0.005));  // un dessin tenu entre deux ne rompt pas la rafale
  CHECK_FALSE(pair_at(0.21));
  CHECK_FALSE(pair_at(0.02));
  // Une coupe isolée, bien après la rafale, compte de nouveau.
  at_us += kCutBurstUs;
  CHECK(pair_at(0.16));
}

TEST_CASE("RIFE : la rafale ne touche ni les coupes des plans ni les paires sans plans", "[app][rife]") {
  PairFacts known = moving(4);
  known.known_cut = true;
  known.scores->scene_score = 0.15;
  known.cut_burst = true;
  CHECK(pair_cut(known));  // une coupe des plans reste une coupe
  PairFacts unknown = moving(4);
  unknown.scores->scene_score = 0.15;
  unknown.cut_burst = true;
  CHECK(pair_cut(unknown));  // hors de la chronologie, la mesure décide seule
  PairFacts calm = moving(4);
  calm.known_cut = false;
  calm.cut_burst = true;
  CHECK_FALSE(pair_cut(calm));
  CHECK(rife_runs(calm));
}

TEST_CASE("RIFE : CutBurst, strictement avant et au plus kCutBurstUs", "[app][rife]") {
  CutBurst burst;
  CHECK_FALSE(burst.after_cut(0));
  burst.note(1'000, false);
  CHECK_FALSE(burst.after_cut(2'000));  // une paire sans coupe ne compte pas
  burst.note(1'000, true);
  CHECK_FALSE(burst.after_cut(1'000));  // la même paire relue
  CHECK_FALSE(burst.after_cut(500));    // avant elle (retour en arrière)
  CHECK(burst.after_cut(1'000 + kCutBurstUs));
  CHECK_FALSE(burst.after_cut(1'001 + kCutBurstUs));
}

}  // namespace ajp::app::play

namespace ajp::app::play {

TEST_CASE("RIFE : sur une paire forte, le milieu sert les instants qu'il prolonge de moins que le seuil",
          "[app][rife][strong_middle]") {
  constexpr float kSingle = 15.0F;
  // 30 px : à t = 1/3 le milieu prolongé déplace de 10 px, servi ; à t = 1/6, de 20 px, RIFE repasse.
  CHECK(middle_reaches(30.0F, 1.0F / 3.0F, kSingle));
  CHECK(middle_reaches(30.0F, 2.0F / 3.0F, kSingle));
  CHECK_FALSE(middle_reaches(30.0F, 1.0F / 6.0F, kSingle));
  CHECK_FALSE(middle_reaches(30.0F, 5.0F / 6.0F, kSingle));
  // 60 px : même t = 1/3 repasse (20 px) ; le milieu lui-même est toujours servi.
  CHECK_FALSE(middle_reaches(60.0F, 1.0F / 3.0F, kSingle));
  CHECK(middle_reaches(60.0F, 0.5F, kSingle));
  // Sous le seuil (paire calme), tout instant l'est : la règle ne prend rien de plus qu'une paire calme.
  CHECK(middle_reaches(14.9F, 0.0F, kSingle));
}

}  // namespace ajp::app::play
