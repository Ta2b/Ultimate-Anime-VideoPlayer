// Dossiers des données : choisis une fois (SDL3 ou --data-dir), puis passés partout.
#include "app/data_paths.h"

#include <catch2/catch_test_macros.hpp>

#include "support.h"

using namespace ajp;
using namespace ajp::app;
using namespace ajp::app::test;

TEST_CASE("Dossier des données : rendu absolu, créé s'il manque, refusé si c'est un fichier",
          "[app][paths]") {
  TempDir dir;
  const auto paths = data_paths_in(dir / "a" / "b", dir / "res");
  REQUIRE(paths);
  CHECK(paths->user_data.is_absolute());
  CHECK(fs::is_directory(paths->user_data));
  CHECK(paths->settings() == dir / "a" / "b" / "settings.toml");
  CHECK(paths->user_profile().filename() == "user.toml");
  CHECK(paths->engines_lock() == dir / "res" / "engines.lock");
  const core::InstallLayout layout = paths->layout(core::Os::Linux);
  CHECK(layout.models == dir / "a" / "b" / "models");
  CHECK(layout.downloads == dir / "a" / "b" / "downloads");

  write_file(dir / "fichier", "x");
  const auto refused = data_paths_in(dir / "fichier", dir / "res");
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FileUnwritable);
}

TEST_CASE("Ressources livrées : ../share/ajp sous Linux, à côté de l'exécutable ailleurs", "[app][paths]") {
  CHECK(resources_from_base(core::Os::Linux, "/opt/ajp/bin/") == fs::path("/opt/ajp/share/ajp"));
  CHECK(resources_from_base(core::Os::Windows, "C:/ajp/") == fs::path("C:/ajp/"));
  CHECK(resources_from_base(core::Os::MacOs, "/Applications/ajp.app/Contents/Resources/") ==
        fs::path("/Applications/ajp.app/Contents/Resources/"));
}

TEST_CASE("Dossiers du système par SDL3 : préférences créées, dossier de l'exécutable connu",
          "[app][paths]") {
  CHECK_FALSE(executable_dir().empty());
  const auto paths = system_data_paths(host_os());
  REQUIRE(paths);
  CHECK(paths->user_data.is_absolute());
  CHECK(fs::is_directory(paths->user_data));
  CHECK(paths->user_data.filename() == "ajp");
}

TEST_CASE("Chemins en UTF-8 dans les deux sens", "[app][paths]") {
  const fs::path path = path_from_utf8("/données/épisode 01 — 「テスト」.mkv");
  CHECK(utf8(path) == "/données/épisode 01 — 「テスト」.mkv");
}
