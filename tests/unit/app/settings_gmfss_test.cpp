// GMFSS Fortuna Union dans play.toml (settings_toml.cpp) : le niveau Élevé (gmfss_middle, gmfss_motion), le
// demi-flot (gmfss_half_flow) et les noms de l'ancien GMFSS (0.10.0) ; la paire au demi-flot (gmfss_model.h).
#include <filesystem>
#include <fstream>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/chain_setup.h"
#include "app/play/chain_shape.h"
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
  // La composition à la taille de travail se relit ; absente (un fichier d'avant), éteinte.
  PlaySettings composed;
  composed.gmfss_middle = true;
  composed.gmfss_compose = true;
  const auto composed_back = parse_settings(format_settings(composed));
  REQUIRE(composed_back);
  CHECK(*composed_back == composed);
  CHECK_FALSE(without->gmfss_compose);
  // Le mélange léger en 540 lignes se relit ; absent, éteint.
  PlaySettings light;
  light.gmfss_middle = true;
  light.gmfss_light = true;
  const auto light_back = parse_settings(format_settings(light));
  REQUIRE(light_back);
  CHECK(*light_back == light);
  CHECK_FALSE(without->gmfss_light);
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

namespace ajp::app::play {

TEST_CASE("GMFSS : la cadence Auto plafonnée, seul à ×2 de la source (×2,5 au demi-flot), mélangé à ×3 (×4)",
          "[app][gmfss]") {
  ChainEngines c;
  CHECK(gmfss_auto_cap(c, 0, 23.976) == 0);  // RIFE : l'écran
  c.gmfss_use = GmfssUse::Only;
  CHECK(gmfss_auto_cap(c, 0, 23.976) == 48);
  CHECK(gmfss_auto_cap(c, 30, 23.976) == 30);  // un plafond choisi plus bas reste
  c.played.gmfss_half_flow = true;
  CHECK(gmfss_auto_cap(c, 0, 23.976) == 60);
  CHECK(gmfss_auto_cap(c, 165, 24.0) == 60);
  c.gmfss_use = GmfssUse::Middle;  // Élevé : budgété au pire cas, chaque paire forte
  CHECK(gmfss_auto_cap(c, 0, 24.0) == 96);
  c.played.gmfss_half_flow = false;
  CHECK(gmfss_auto_cap(c, 0, 24.0) == 72);
  c.played.gmfss_light = true;  // 540 lignes : comme le demi-flot
  CHECK(gmfss_auto_cap(c, 0, 24.0) == 96);
  c.played.gmfss_light = false;
  c.played.motion_engine = MotionEngine::Gmfss;  // Très élevé par le mélange : 60 i/s
  CHECK(gmfss_auto_cap(c, 0, 23.976) == 60);
  c.played.gmfss_light = true;  // 540 lignes : ×3,5
  CHECK(gmfss_auto_cap(c, 0, 24.0) == 84);
  c.played.gmfss_half_flow = true;  // et le demi-flot : ×4
  CHECK(gmfss_auto_cap(c, 0, 24.0) == 96);
  c.played.gmfss_half_flow = false;
  c.played.gmfss_light = false;
  c.gmfss_use = GmfssUse::Off;
  CHECK(gmfss_auto_cap(c, 0, 24.0) == 0);
}

}  // namespace ajp::app::play

TEST_CASE("GMFSS : l'Auto Rapide laisse la place au Stable", "[app][gmfss]") {
  CHECK(ajp::app::play::free_refusal(ajp::app::play::MotionEngine::Rife, true, true).empty());
  CHECK_FALSE(ajp::app::play::free_refusal(ajp::app::play::MotionEngine::Rife, true, true, true).empty());
  CHECK_FALSE(ajp::app::play::free_refusal(ajp::app::play::MotionEngine::Gmfss, true, true, true).empty());
}

TEST_CASE("GMFSS : le coût d'une paire du mélange, à l'échelle des pixels, moindre au demi-flot",
          "[app][gmfss]") {
  using ajp::app::play::gmfss_cost;
  CHECK(gmfss_cost({.width = 1280, .height = 720}, false).total_us == 50300);
  CHECK(gmfss_cost({.width = 1280, .height = 720}, false).pair_us == 35000);
  CHECK(gmfss_cost({.width = 1280, .height = 720}, true).total_us == 27300);
  CHECK(gmfss_cost({.width = 640, .height = 360}, true).total_us == 6825);
}

TEST_CASE("GMFSS : Très élevé par la chaîne du mélange, ou à chaque instant sans RIFE", "[app][gmfss]") {
  using namespace ajp::app::play;
  PlaySettings s;
  s.motion_engine = MotionEngine::Gmfss;
  s.rife_factor = 2;
  CHECK(gmfss_wanted(s) == GmfssUse::Middle);
  s.gmfss_every_instant = true;
  CHECK(gmfss_wanted(s) == GmfssUse::Only);
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK(back->gmfss_every_instant);
  CHECK_FALSE(parse_settings("version = 1\n")->gmfss_every_instant);
  // Très élevé par le mélange : ×2,5 de la source, RIFE reprenant les paires que GMFSS ne tient pas.
  ChainEngines c;
  c.gmfss_use = GmfssUse::Middle;
  c.played.motion_engine = MotionEngine::Gmfss;
  CHECK(gmfss_auto_cap(c, 0, 24.0) == 60);
}
