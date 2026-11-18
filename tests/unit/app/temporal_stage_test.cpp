// La stabilisation temporelle dans l'étage gpu (upscale_temporal.cpp) sur le banc sans GPU (upscale_rig.h) :
// les sept images lues par passe, dans l'ordre des entrées du moteur, le reflet aux bords, une ingestion par
// image, et l'image suivante de RIFE stabilisée elle aussi.
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "upscale_rig.h"

namespace ajp::app::play {
using namespace upscale_test;

namespace {

// Le milieu d'abord : la sortie du moteur factice porte l'image de sa première entrée, et l'étage doit lire
// les noms plutôt que leur ordre.
test::FakeEngine temporal_engine() {
  return {"temporalfix",
          {"input_3", "input_0", "input_1", "input_2", "input_4", "input_5", "input_6"},
          {"output"}};
}

bool ran(const Rig& rig, const std::string& ids) {
  const std::vector<std::string> log = rig.backend.log();
  return std::ranges::find(log, "temporalfix " + ids) != log.end();
}

}  // namespace

TEST_CASE("Étage gpu : la stabilisation lit trois images de chaque côté, reflétées aux bords",
          "[app][stage][temporal]") {
  Rig rig;
  test::FakeEngine fix = temporal_engine();
  UpscaleSetup s = rig.setup(false);
  s.temporal_fix = &fix;
  rig.start(std::move(s));
  rig.feed(1, 6);
  REQUIRE(rig.stage->post(StreamEnd{.epoch = 0}));
  REQUIRE(rig.collect(6));
  rig.stage.reset();

  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "2", "3", "4", "5", "6"});
  CHECK(rig.backend.count("engine temporalfix") == 6);
  CHECK(rig.backend.count("engine span") == 6);
  for (int id = 1; id <= 6; ++id) CHECK(rig.backend.count("ingest " + std::to_string(id)) == 1);
  // Dans l'ordre des entrées du moteur : t, t-3, t-2, t-1, t+1, t+2, t+3.
  CHECK(ran(rig, "1,4,3,2,2,3,4"));  // le début : les images d'après, reflétées
  CHECK(ran(rig, "3,6,1,2,4,5,6"));
  CHECK(ran(rig, "4,1,2,3,5,6,1"));
  CHECK(ran(rig, "6,3,4,5,5,4,3"));  // la fin du flux : les images d'avant, reflétées
}

TEST_CASE("Étage gpu : avec RIFE, l'image lue en avance est stabilisée une fois, sa fenêtre entière",
          "[app][stage][temporal]") {
  Rig rig;
  test::FakeEngine fix = temporal_engine();
  UpscaleSetup s = rig.setup(true);
  s.temporal_fix = &fix;
  rig.start(std::move(s));
  rig.feed(1, 5);
  REQUIRE(rig.stage->post(StreamEnd{.epoch = 0}));
  REQUIRE(rig.collect(5));
  REQUIRE(logged(rig, "engine temporalfix", 5));
  rig.stage.reset();

  CHECK(rig.backend.count("engine temporalfix") == 5);
  for (int id = 1; id <= 5; ++id) CHECK(rig.backend.count("ingest " + std::to_string(id)) == 1);
  CHECK(ran(rig, "2,5,4,1,3,4,5"));  // lue derrière 1 : les images 3 à 5 étaient déjà là
}

}  // namespace ajp::app::play
