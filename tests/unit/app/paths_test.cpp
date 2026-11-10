// Les emplacements du lecteur (paths.h) : deux racines, des noms fixes, un dossier aux caractères non ASCII
// gardé tel quel. Les séparateurs sont ceux du système : le test tourne sur les trois.
#include "app/play/paths.h"

#include <filesystem>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::app::play;
namespace fs = std::filesystem;

TEST_CASE("Chemins : le dossier des données et celui des réglages", "[app][paths]") {
  const fs::path root = fs::path(u8"données") / u8"アニメ";
  PlayerPaths p;
  p.engines = root / "engines";
  CHECK(paths::data_root(p) == root);
  CHECK(paths::derived_models(p) == root / "modeles");
  CHECK(paths::luts(p) == root / "luts");
  CHECK(paths::backends_install(p) == root / "backends");
  CHECK(paths::shader_cache(p) == root / "shaders.cache");
  CHECK(paths::shot_cache(p) == root / "plans");
  // Sans --data-dir, les réglages vivent avec les données ; avec, dans le dossier donné.
  CHECK(paths::settings_dir(p) == root);
  p.settings = fs::path("ailleurs");
  CHECK(paths::settings_dir(p) == fs::path("ailleurs"));
  CHECK(paths::play_settings(paths::settings_dir(p)) == fs::path("ailleurs") / "play.toml");
  CHECK(paths::engine_settings(root).filename() == "settings.toml");
  CHECK(paths::user_keys(root).filename() == "user.toml");
  CHECK(paths::bench(root).filename() == "bench.toml");
  CHECK(paths::resume(root).filename() == "resume.tsv");
  CHECK(paths::auto_memory(root).filename() == "auto_memory.toml");
  CHECK(paths::update_state(root) == root / "update.toml");
}

TEST_CASE("Chemins : les caches qui se refont seuls ont leur dossier (A36-08)", "[app][paths]") {
  PlayerPaths p;
  p.heavy = fs::path("local") / "ajp";
  p.cache = fs::path("cache") / "ajp";
  p.engines = p.cache / "engines";
  CHECK(paths::data_root(p) == p.heavy);  // pas le parent des moteurs construits
  CHECK(paths::luts(p) == p.heavy / "luts");
  CHECK(paths::backends_install(p) == p.heavy / "backends");
  CHECK(paths::cache_root(p) == p.cache);
  CHECK(paths::derived_models(p) == p.cache / "modeles");
  CHECK(paths::shader_cache(p) == p.cache / "shaders.cache");
  CHECK(paths::shot_cache(p) == p.cache / "plans");
}
