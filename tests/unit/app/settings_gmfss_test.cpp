// GMFSS Fortuna Union dans play.toml (settings_toml.cpp) : le niveau Élevé (gmfss_middle, gmfss_motion), le
// demi-flot (gmfss_half_flow) et les noms de l'ancien GMFSS (0.10.0) ; la paire au demi-flot (gmfss_model.h).
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/chain_setup.h"
#include "app/play/chain_shape.h"
#include "app/play/gmfss_model.h"
#include "app/play/interp_retired.h"
#include "app/play/model_choice.h"
#include "app/play/player.h"
#include "app/play/settings.h"
#include "prepare/onnx_build.h"

namespace kirari::app::play {

TEST_CASE("play.toml : GMFSS union, le niveau Élevé et les noms de l'ancien GMFSS",
          "[app][settings][gmfss]") {
  // L'ancien GMFSS (0.10.0) : « gmfss » se lit en RIFE (A22-18). GMFSS union, retiré du menu en novembre
  // 2026, tombe sur RIFE 4.25 (interp_retired.h).
  const auto old_gmfss = parse_settings("[interpolation]\nengine = \"gmfss\"\n");
  REQUIRE(old_gmfss);
  CHECK(old_gmfss->motion_engine == MotionEngine::Rife);
  const auto old_union = parse_settings("[interpolation]\nengine = \"gmfss_union\"\n");
  REQUIRE(old_union);
  CHECK(old_union->motion_engine == MotionEngine::Rife);
  CHECK(old_union->rife_model == "rife_v4.25.onnx");
  // L'ancien niveau Élevé (GMFSS au milieu) tombe sur RIFE 4.25 ; son seuil se relit.
  PlaySettings high;
  high.gmfss_middle = true;
  high.gmfss_motion = 22;
  const auto high_back = parse_settings(format_settings(high));
  REQUIRE(high_back);
  CHECK_FALSE(high_back->gmfss_middle);
  CHECK(high_back->motion_engine == MotionEngine::Rife);
  CHECK(high_back->rife_model == "rife_v4.25.onnx");
  CHECK(high_back->gmfss_motion == 22);
  const auto without = parse_settings("version = 1\n[interpolation]\nengine = \"rife\"\n");
  REQUIRE(without);
  CHECK_FALSE(without->gmfss_middle);
  CHECK(without->gmfss_motion == kDefaultGmfssMotion);
  CHECK_FALSE(parse_settings("version = 1\n[interpolation]\ngmfss_motion = 200\n"));
  CHECK(interp_level(MotionEngine::Rife, "rife_v4.25.onnx", true) == InterpLevel::GmfssMiddle);
  CHECK(interp_level(MotionEngine::Rife, "rife_v4.22_lite.onnx", true) == InterpLevel::GmfssMiddle);
  CHECK(interp_level(MotionEngine::Gmfss, "") == InterpLevel::GmfssUnion);
  CHECK(interp_level(MotionEngine::Gmfss, "", true) == InterpLevel::GmfssUnion);
  // Le demi-flot se relit ; absent (un fichier d'avant), il est pris par défaut ; éteint, il le reste.
  PlaySettings half;
  half.motion_engine = MotionEngine::Gmfss;
  half.gmfss_half_flow = false;
  const auto half_back = parse_settings(format_settings(half));
  REQUIRE(half_back);
  CHECK_FALSE(half_back->gmfss_half_flow);
  CHECK(without->gmfss_half_flow);
  // La composition à la taille de travail se relit ; absente (un fichier d'avant), éteinte.
  PlaySettings composed;
  composed.gmfss_middle = true;
  composed.gmfss_compose = true;
  const auto composed_back = parse_settings(format_settings(composed));
  REQUIRE(composed_back);
  CHECK(composed_back->gmfss_compose);
  CHECK_FALSE(without->gmfss_compose);
  // Le mélange léger en 540 lignes se relit ; absent, éteint.
  PlaySettings light;
  light.gmfss_middle = true;
  light.gmfss_light = true;
  const auto light_back = parse_settings(format_settings(light));
  REQUIRE(light_back);
  CHECK(light_back->gmfss_light);
  CHECK_FALSE(without->gmfss_light);
}

TEST_CASE("play.toml avec le natif (défaut) : GMFSS union, seul ou au milieu, se relit tel quel",
          "[app][settings][gmfss]") {
  // keep_gmfss est posé par le lecteur au lancement ; le test le rend éteint quoi qu'il arrive.
  struct Kept {
    Kept() { keep_gmfss(true); }
    ~Kept() { keep_gmfss(false); }
    Kept(const Kept&) = delete;
    Kept& operator=(const Kept&) = delete;
  };
  {
    const Kept kept;
    const auto alone = parse_settings("[interpolation]\nengine = \"gmfss_union\"\n");
    REQUIRE(alone);
    CHECK(alone->motion_engine == MotionEngine::Gmfss);
    PlaySettings high;
    high.gmfss_middle = true;
    high.rife_model = "rife_v4.26.onnx";  // l'ancien Élevé : RIFE 4.26, retiré, tombe sur 4.25
    const auto high_back = parse_settings(format_settings(high));
    REQUIRE(high_back);
    CHECK(high_back->gmfss_middle);
    CHECK(high_back->motion_engine == MotionEngine::Rife);
    CHECK(high_back->rife_model == "rife_v4.25.onnx");
    CHECK(interp_level(high_back->motion_engine, high_back->rife_model, high_back->gmfss_middle) ==
          InterpLevel::GmfssMiddle);
  }
  // Sous --no-gmfss-native, le même fichier retombe sur RIFE 4.25.
  const auto alone = parse_settings("[interpolation]\nengine = \"gmfss_union\"\n");
  REQUIRE(alone);
  CHECK(alone->motion_engine == MotionEngine::Rife);
  CHECK(alone->rife_model == "rife_v4.25.onnx");
}

TEST_CASE("GMFSS : la paire au demi-flot trouvée à côté des deux graphes, choisie si demandée",
          "[app][gmfss]") {
  const auto dir = std::filesystem::temp_directory_path() / "kirari-gmfss-half";
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

// Essai du PC du 19 novembre : les deux graphes n'étaient trouvés qu'une fois liés dans le dossier de RIFE et
// RIFE nommé par --rife-model. Ils se cherchent aussi dans --gmfss-dir et dans le voisin « gmfss » de chaque
// dossier de RIFE.
TEST_CASE("GMFSS : cherché dans --gmfss-dir, puis à côté de RIFE et dans le dossier gmfss voisin",
          "[app][gmfss]") {
  const auto home = std::filesystem::temp_directory_path() / "kirari-gmfss-places";
  std::filesystem::remove_all(home);
  const auto touch_both = [](const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);
    std::ofstream(dir / kGmfssPairFile) << "onnx";
    std::ofstream(dir / kGmfssFusionFile) << "onnx";
  };
  const PlayerPaths paths{.home = home, .heavy = home / "data", .models = home / "build"};
  const std::vector<std::filesystem::path> dirs = motion_model_dirs(paths, std::nullopt);
  // Voisins sans doublon : « rife/ » et « rife » donnent le même ; data/rife et data/models/rife aussi.
  const std::vector<std::filesystem::path> trailing{home / "data/models/rife/", home / "data/models/rife"};
  CHECK(gmfss_places(trailing) == std::vector<std::filesystem::path>{home / "data/models/rife/",
                                                                     home / "data/models/gmfss",
                                                                     home / "data/models/rife"});
  CHECK_FALSE(find_gmfss(dirs));
  touch_both(home / "kirari/models/gmfss");  // export_gmfss.py --out ~/kirari/models/gmfss
  auto found = find_gmfss(dirs);
  REQUIRE(found);
  CHECK(found->pair == home / "kirari/models/gmfss" / kGmfssPairFile);
  touch_both(home / "data/models/gmfss");  // le dossier des données passe devant ~/kirari
  found = find_gmfss(dirs);
  REQUIRE(found);
  CHECK(found->fusion == home / "data/models/gmfss" / kGmfssFusionFile);
  // --gmfss-dir passe devant tout, même un dossier quelconque (~/ajp-work/gmfss-out).
  touch_both(home / "ajp-work/gmfss-out");
  const std::optional<std::filesystem::path> given = home / "ajp-work/gmfss-out";
  found = find_gmfss(motion_model_dirs(paths, std::nullopt, given));
  REQUIRE(found);
  CHECK(found->pair == home / "ajp-work/gmfss-out" / kGmfssPairFile);
  std::filesystem::remove_all(home);
}

// Essai du PC : une paire au flot entier d'un export d'avant le format 2 (source{k}_1 de 67 canaux, sans
// feature0_1) faisait refuser GMFSS au chargement du moteur, après sa construction. Une paire périmée cède
// maintenant la place à l'autre paire du dossier, dans un sens comme dans l'autre ; sans paire au format 2,
// GMFSS est écarté avant toute construction.
TEST_CASE("GMFSS : une paire d'avant le format 2 est écartée avant la construction", "[app][gmfss]") {
  const auto dir = std::filesystem::temp_directory_path() / "kirari-gmfss-format";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto graph_with = [&](std::string_view file, std::initializer_list<std::string_view> outputs) {
    prepare::onnx_build::Message graph;
    for (const std::string_view name : outputs) {
      const std::array<std::int64_t, 4> dims{1, 3, 32, 32};
      graph.message(12, prepare::onnx_build::value_info(name, 10, dims));
    }
    prepare::onnx_build::Message model;
    model.varint(1, 8).message(7, graph);
    std::ofstream(dir / file, std::ios::binary)
        .write(reinterpret_cast<const char*>(model.data().data()),
               static_cast<std::streamsize>(model.data().size()));
  };
  graph_with(kGmfssPairFile, {"half0", "half1", "feature0_1", "feature1_1", "source0_2", "motion0_1"});
  graph_with(kGmfssHalfPairFile, {"source0_1", "source1_1", "source0_2", "motion0_1"});  // format 1
  graph_with(kGmfssFusionFile, {"out"});
  const std::vector<std::filesystem::path> dirs{dir};
  const auto found = find_gmfss(dirs);
  REQUIRE(found);
  CHECK(gmfss_pair_current(found->pair));
  const auto stale = gmfss_pair_current(found->half_pair);
  REQUIRE_FALSE(stale);
  CHECK(stale.error().code == RefusalCode::GraphInvalid);
  CHECK(stale.error().detail.find("feature0_1") != std::string::npos);
  std::string dropped;
  auto usable = gmfss_usable(*found, true, dropped);
  REQUIRE(usable);
  CHECK(usable->pair == found->pair);  // le demi-flot périmé : le flot entier
  CHECK(dropped.find(kGmfssHalfPairFile) != std::string::npos);
  usable = gmfss_usable(*found, false, dropped);
  REQUIRE(usable);
  CHECK(dropped.empty());
  // Le cas du PC : la paire au flot entier périmée, celle au demi-flot au format 2, prise même sans half.
  graph_with(kGmfssPairFile, {"source0_1", "source1_1", "source0_2", "motion0_1"});
  graph_with(kGmfssHalfPairFile, {"half0", "half1", "feature0_1", "feature1_1", "source0_2", "motion0_1"});
  usable = gmfss_usable(*found, false, dropped);
  REQUIRE(usable);
  CHECK(usable->pair == found->half_pair);
  CHECK(dropped.find(kGmfssPairFile) != std::string::npos);
  usable = gmfss_usable(*found, true, dropped);
  REQUIRE(usable);
  CHECK(usable->pair == found->half_pair);
  CHECK(dropped.empty());
  // Sans paire au format 2 dans le dossier : GMFSS écarté, le refus dit quoi faire.
  GmfssModels alone = *found;
  alone.half_pair.clear();
  usable = gmfss_usable(alone, true, dropped);
  REQUIRE_FALSE(usable);
  CHECK(usable.error().detail.find("export_gmfss.py") != std::string::npos);
  graph_with(kGmfssHalfPairFile, {"source0_1", "source1_1"});
  usable = gmfss_usable(*found, true, dropped);
  REQUIRE_FALSE(usable);
  CHECK(usable.error().detail.find(kGmfssHalfPairFile) != std::string::npos);
  CHECK(dropped.empty());
  std::filesystem::remove_all(dir);
}

}  // namespace kirari::app::play

namespace kirari::app::play {

TEST_CASE(
    "GMFSS : la cadence Auto plafonnée seul à ×2 de la source (×2,5 au demi-flot), le mélange à l'écran",
    "[app][gmfss]") {
  ChainEngines c;
  CHECK(gmfss_auto_cap(c, 0, 23.976) == 0);  // RIFE : l'écran
  c.gmfss_use = GmfssUse::Only;
  c.played.gmfss_half_flow = false;  // le flot entier
  CHECK(gmfss_auto_cap(c, 0, 23.976) == 48);
  CHECK(gmfss_auto_cap(c, 30, 23.976) == 30);  // un plafond choisi plus bas reste
  c.played.gmfss_half_flow = true;
  CHECK(gmfss_auto_cap(c, 0, 23.976) == 60);
  CHECK(gmfss_auto_cap(c, 165, 24.0) == 60);
  c.played.gmfss_half_flow = false;
  c.played.gmfss_light = true;  // 540 lignes : comme le demi-flot
  CHECK(gmfss_auto_cap(c, 0, 24.0) == 60);
  // Le mélange, Élevé comme Très élevé : les clés guident RIFE jusqu'à l'écran, le plafond choisi reste.
  c.gmfss_use = GmfssUse::Middle;
  CHECK(gmfss_auto_cap(c, 0, 24.0) == 0);
  CHECK(gmfss_auto_cap(c, 120, 24.0) == 120);
  c.played.motion_engine = MotionEngine::Gmfss;
  CHECK(gmfss_auto_cap(c, 0, 24.0) == 0);
  c.gmfss_use = GmfssUse::Off;
  CHECK(gmfss_auto_cap(c, 0, 24.0) == 0);
}

}  // namespace kirari::app::play

TEST_CASE("GMFSS : l'Auto Rapide reste au choix", "[app][gmfss]") {
  CHECK(kirari::app::play::free_refusal(kirari::app::play::MotionEngine::Rife, true, true).empty());
  CHECK(kirari::app::play::free_refusal(kirari::app::play::MotionEngine::Gmfss, true, true).empty());
}

TEST_CASE("GMFSS : le coût d'une paire du mélange, à l'échelle des pixels, moindre au demi-flot",
          "[app][gmfss]") {
  using kirari::app::play::gmfss_cost;
  CHECK(gmfss_cost({.width = 1280, .height = 720}, false).total_us == 50300);
  CHECK(gmfss_cost({.width = 1280, .height = 720}, false).pair_us == 35000);
  CHECK(gmfss_cost({.width = 1280, .height = 720}, true).total_us == 27300);
  CHECK(gmfss_cost({.width = 640, .height = 360}, true).total_us == 6825);
}

TEST_CASE("GMFSS : Très élevé par la chaîne du mélange, ou à chaque instant sans RIFE", "[app][gmfss]") {
  using namespace kirari::app::play;
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
  // Très élevé par le mélange : l'écran, RIFE reprenant les paires que GMFSS ne tient pas.
  ChainEngines c;
  c.gmfss_use = GmfssUse::Middle;
  c.played.motion_engine = MotionEngine::Gmfss;
  CHECK(gmfss_auto_cap(c, 0, 24.0) == 0);
}

TEST_CASE("GMFSS : le plancher des clés dans play.toml, borné, 0 dans un ancien fichier", "[app][gmfss]") {
  using kirari::app::play::format_settings;
  using kirari::app::play::parse_settings;
  CHECK(parse_settings("version = 1\n")->gmfss_key_floor == 0);
  kirari::app::play::PlaySettings s;
  s.gmfss_key_floor = 15;
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK(back->gmfss_key_floor == 15);
  CHECK_FALSE(parse_settings("version = 1\n[interpolation]\ngmfss_key_floor = 99\n"));
}
