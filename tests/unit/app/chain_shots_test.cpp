#include "app/play/chain_shots.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {

TEST_CASE("Pré-analyse : aucun plan à lire pour une source HDR sans grade natif", "[app][shots]") {
  PlaySettings settings;
  settings.grade = true;
  settings.hdr_native = false;
  ShotWork work;
  open_shots(work, "inexistant.mkv", settings, 0, true);
  CHECK(work.current.probe == nullptr);
  CHECK(work.current.timeline == nullptr);
}

TEST_CASE("Pré-analyse : rien sans grade", "[app][shots]") {
  PlaySettings settings;
  settings.grade = false;
  ShotWork work;
  open_shots(work, "inexistant.mkv", settings, 0, false);
  CHECK(work.current.probe == nullptr);
  CHECK(work.current.timeline == nullptr);
}

TEST_CASE("Pré-analyse : sans grade, elle part quand l'interpolation lit les coupes des plans (A42-09)",
          "[app][shots]") {
  PlaySettings settings;
  settings.grade = false;
  settings.rife_factor = 2;
  ShotWork work;
  work.shot_cuts = true;
  open_shots(work, "inexistant.mkv", settings, 0, true);
  REQUIRE(work.current.probe != nullptr);  // elle échouera sur ce fichier, sans gêner la lecture
  CHECK(work.current.timeline != nullptr);
  // Ni grade ni interpolation : elle s'arrête.
  settings.rife_factor = 0;
  open_shots(work, "inexistant.mkv", settings, 0, true);
  CHECK(work.current.probe == nullptr);
}

TEST_CASE("Limiteur de flashs : l'échelle du grade suit son pic", "[app][shots]") {
  CHECK(flash_gain_of(1000.0) == Catch::Approx(690.0 / 203.0));
  CHECK(flash_gain_of(400.0) == Catch::Approx(320.0 / 203.0));
  CHECK(flash_gain_of(2000.0) == Catch::Approx(1304.0 / 203.0));
  CHECK(flash_gain_of(100.0) == 1.0);  // jamais sous la source
}

}  // namespace ajp::app::play
