// GMFSS Fortuna Union dans play.toml (settings_toml.cpp) : le niveau Élevé (gmfss_middle, gmfss_motion), le
// demi-flot (gmfss_half_flow) et les noms de l'ancien GMFSS (0.10.0) ; la paire au demi-flot (gmfss_model.h).
#include <filesystem>
#include <fstream>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/gmfss_model.h"
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
  // Le demi-flot se relit ; absent, la paire au flot entier.
  PlaySettings half;
  half.motion_engine = MotionEngine::Gmfss;
  half.gmfss_half_flow = true;
  const auto half_back = parse_settings(format_settings(half));
  REQUIRE(half_back);
  CHECK(*half_back == half);
  CHECK_FALSE(without->gmfss_half_flow);
}

TEST_CASE("GMFSS : la paire au demi-flot trouvée à côté des deux graphes, choisie si demandée",
          "[app][gmfss]") {
  const auto dir = std::filesystem::temp_directory_path() / "ajp-gmfss-half";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto touch = [&](std::string_view name) { std::ofstream(dir / name) << "onnx"; };
  touch(kGmfssPairFile);
  touch(kGmfssFusionFile);
  const std::vector<std::filesystem::path> dirs{dir};
  auto found = find_gmfss(dirs);
  REQUIRE(found);
  CHECK(found->half_pair.empty());
  CHECK(gmfss_chosen(*found, true).pair == dir / kGmfssPairFile);  // sans le graphe : le flot entier
  touch(kGmfssHalfPairFile);
  found = find_gmfss(dirs);
  REQUIRE(found);
  CHECK(found->half_pair == dir / kGmfssHalfPairFile);
  CHECK(gmfss_chosen(*found, true).pair == dir / kGmfssHalfPairFile);
  CHECK(gmfss_chosen(*found, false).pair == dir / kGmfssPairFile);
  CHECK(gmfss_chosen(*found, true).fusion == dir / kGmfssFusionFile);
  std::filesystem::remove_all(dir);
}

}  // namespace ajp::app::play
