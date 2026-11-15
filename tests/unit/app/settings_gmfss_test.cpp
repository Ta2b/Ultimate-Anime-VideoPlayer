// GMFSS Fortuna Union dans play.toml (settings_toml.cpp) : le niveau Élevé (gmfss_middle, gmfss_motion) et
// les noms de l'ancien GMFSS (0.10.0).
#include <catch2/catch_test_macros.hpp>

#include "app/play/settings.h"

namespace ajp::app::play {

TEST_CASE("play.toml : GMFSS union, le niveau Élevé et les noms de l'ancien GMFSS",
          "[app][settings][gmfss]") {
  // L'ancien GMFSS (0.10.0) : « gmfss » se lit en RIFE (A22-18), « gmfss_union » en GMFSS union seul.
  const auto old_gmfss = parse_settings("[interpolation]\nengine = \"gmfss\"\n");
  REQUIRE(old_gmfss);
  CHECK(old_gmfss->motion_engine == MotionEngine::Rife);
  const auto old_union = parse_settings("[interpolation]\nengine = \"gmfss_union\"\n");
  REQUIRE(old_union);
  CHECK(old_union->motion_engine == MotionEngine::Gmfss);
  // Le niveau Élevé et son seuil se relisent ; un play.toml qui ne les nomme pas garde RIFE seul.
  PlaySettings high;
  high.gmfss_middle = true;
  high.gmfss_motion = 22;
  const auto high_back = parse_settings(format_settings(high));
  REQUIRE(high_back);
  CHECK(*high_back == high);
  const auto without = parse_settings("version = 1\n[interpolation]\nengine = \"rife\"\n");
  REQUIRE(without);
  CHECK_FALSE(without->gmfss_middle);
  CHECK(without->gmfss_motion == kDefaultGmfssMotion);
  CHECK_FALSE(parse_settings("version = 1\n[interpolation]\ngmfss_motion = 200\n"));
  CHECK(interp_level(MotionEngine::Rife, "rife_v4.26.onnx", true) == InterpLevel::High);
  CHECK(interp_level(MotionEngine::Gmfss, "") == InterpLevel::GmfssUnion);
}

}  // namespace ajp::app::play
