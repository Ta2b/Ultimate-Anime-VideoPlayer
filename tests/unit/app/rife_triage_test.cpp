// Tri des paires de RIFE par la carte de la scène (rife_triage.h) : un seul test pour le module, qui fait
// passer les mêmes paires sans carte, sans le réglage, puis avec la carte (tenue, impossible, douteuse).
#include "app/play/rife_triage.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "kernels/spec/motion_doubt.h"

namespace kirari::app::play {
namespace {

// 8 × 6 tuiles de dessin sûres et immobiles : une scène calme.
kernels::SceneMotion calm_scene() {
  kernels::SceneMotion s;
  s.camera_fit = true;
  s.tile_px = 32;
  s.cols = 8;
  s.rows = 6;
  s.tiles.assign(48, kernels::SceneTile{.kind = kernels::SceneClass::Drawing, .confidence = 0.9F});
  return s;
}

// Une paire qui bouge (la mesure des blocs laisse tourner RIFE), avec sa carte.
PairFacts moving(const kernels::SceneMotion& scene, bool triage) {
  return PairFacts{.factor = 4,
                   .same_epoch = true,
                   .gap_us = 41'708,
                   .scores = gpu::PairScores{.scene_score = 0.02, .worst_block = 2000},
                   .scene = scene,
                   .flow_triage = triage};
}

}  // namespace

TEST_CASE("tri des paires : sans carte rien ne change, avec elle tenue, fondu et fondu par endroits",
          "[app][rife][triage]") {
  // La carte d'un dessin tenu : rien ne bouge, rien ne change.
  kernels::SceneMotion held = calm_scene();
  held.drawing_held = true;
  held.background_held = true;

  // Sans le réglage, ou sans carte : la décision d'aujourd'hui, et aucun fondu.
  RifeTriage off;
  for (PairFacts facts : {moving(held, false), PairFacts{moving(held, true)}}) {
    if (facts.flow_triage) facts.scene.reset();
    CHECK(rife_runs(facts));
    CHECK_FALSE(pair_held(facts));
    CHECK(off.sort(facts, rife_runs(facts)) == PairSort::Today);
  }
  CHECK(off.counts().read == 0);

  // Dessin tenu par le flux (le grain trompait la mesure des blocs) : RIFE ne tourne pas, la paire est
  // comptée.
  RifeTriage triage;
  const PairFacts by_flow = moving(held, true);
  CHECK(pair_held(by_flow));
  CHECK_FALSE(rife_runs(by_flow));
  CHECK(triage.sort(by_flow, rife_runs(by_flow)) == PairSort::Held);
  CHECK(triage.counts().held == 1);
  // Un effet qui change sur le dessin tenu : plus tenu.
  kernels::SceneMotion rain = held;
  for (std::size_t i = 0; i < 4; ++i) rain.tiles[i].changed = true;
  CHECK_FALSE(flow_held(rain));
  CHECK(rife_runs(moving(rain, true)));

  // Scène calme : RIFE tel quel.
  const kernels::SceneMotion calm = calm_scene();
  CHECK(triage.sort(moving(calm, true), true) == PairSort::Today);
  CHECK(triage.counts().read == 1);

  // Presque rien qui corresponde : toute la paire fondue.
  kernels::SceneMotion lost = calm_scene();
  lost.unmatched = 0.7;
  CHECK(triage.sort(moving(lost, true), true) == PairSort::Fade);

  // Un effet dans un coin : fondu par endroits, le bord adouci, le coin opposé intact.
  triage.reset();
  kernels::SceneMotion smoke = calm_scene();
  smoke.tiles[0] = {.kind = kernels::SceneClass::Effect, .confidence = 0.2F, .changed = true};
  CHECK(triage.sort(moving(smoke, true), true) == PairSort::Masked);
  const std::vector<float> doubt = tile_doubt(smoke);
  CHECK(doubt[0] == 1.0F);
  CHECK(doubt[47] == 0.0F);
  const DoubtGrid grid = doubt_grid(doubt, smoke.cols, smoke.rows);
  REQUIRE(grid.cols == 8);
  REQUIRE(grid.rows == 6);
  CHECK(grid.weights[0] == kernels::kDoubtFull);  // l'effet et la tuile qui le borde : fondu entier
  CHECK(grid.weights[2] > 0);                     // deux tuiles plus loin : le bord adouci
  CHECK(grid.weights[2] < kernels::kDoubtFull);
  CHECK(grid.weights[47] == 0);
  // La paire suivante, sans l'effet, garde la moitié du doute : la zone ne clignote pas.
  CHECK(triage.sort(moving(calm, true), true) == PairSort::Masked);
  // Une coupe l'oublie.
  PairFacts cut = moving(calm, true);
  cut.scores->scene_score = 0.5;
  CHECK(triage.sort(cut, rife_runs(cut)) == PairSort::Today);
  CHECK(triage.sort(moving(calm, true), true) == PairSort::Today);
  CHECK(triage.counts().masked == 2);
  CHECK(triage.counts().faded == 1);

  // Une grande carte (1080p : 60 × 34 tuiles) se regroupe sous kDoubtMaxCells en gardant le plus fort doute.
  std::vector<float> wide(60 * 34, 0.0F);
  wide[59] = 1.0F;
  const DoubtGrid pooled = doubt_grid(wide, 60, 34);
  CHECK(pooled.cols <= kernels::kDoubtMaxCells);
  CHECK(pooled.rows <= kernels::kDoubtMaxCells);
  CHECK(pooled.weights[static_cast<std::size_t>(pooled.cols - 1)] > 0);
  CHECK(std::ranges::count_if(pooled.weights, [](std::uint8_t w) { return w > 0; }) < 8);
}

}  // namespace kirari::app::play
