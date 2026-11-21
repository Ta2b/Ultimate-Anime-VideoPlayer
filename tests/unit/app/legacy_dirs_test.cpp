#include "app/play/legacy_dirs.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/launch_data.h"

namespace fs = std::filesystem;
using kirari::app::play::adopt_legacy_dir;
using kirari::app::play::legacy_pref_dir;

TEST_CASE("Dossiers d'ajp : l'ancien dossier à deux niveaux d'après le nouveau", "[app][legacy_dirs]") {
  CHECK(legacy_pref_dir(fs::path("/donnees") / "kirari" / "") == fs::path("/donnees") / "ajp" / "ajp");
  CHECK(legacy_pref_dir(fs::path("/donnees") / "kirari") == fs::path("/donnees") / "ajp" / "ajp");
  CHECK(legacy_pref_dir({}).empty());
}

TEST_CASE("Dossiers d'ajp : repris une fois, réglages et moteurs compris", "[app][legacy_dirs]") {
  const fs::path root = fs::temp_directory_path() / "kirari-legacy-dirs-test";
  fs::remove_all(root);
  const fs::path legacy = root / "ajp" / "ajp";
  const fs::path target = root / "kirari";
  fs::create_directories(legacy / "engines");
  std::ofstream(legacy / "play.toml") << "[image]\n";
  std::ofstream(legacy / "engines" / "span.plan") << "plan";
  fs::create_directories(target);  // SDL_GetPrefPath vient de le créer, vide
  std::vector<std::string> said;
  CHECK(adopt_legacy_dir(legacy, target / "", [&](const std::string& line) { said.push_back(line); }));
  CHECK(fs::is_regular_file(target / "play.toml"));
  CHECK(fs::is_regular_file(target / "engines" / "span.plan"));
  CHECK_FALSE(fs::exists(root / "ajp"));  // le parent vide part aussi
  CHECK(said.size() == 1);
  // Une deuxième fois, plus rien à reprendre.
  CHECK_FALSE(adopt_legacy_dir(legacy, target));
  fs::remove_all(root);
}

TEST_CASE("Dossiers d'ajp : un dossier Kirari déjà servi n'est pas écrasé", "[app][legacy_dirs]") {
  const fs::path root = fs::temp_directory_path() / "kirari-legacy-dirs-kept-test";
  fs::remove_all(root);
  const fs::path legacy = root / "ajp" / "ajp";
  const fs::path target = root / "kirari";
  fs::create_directories(legacy);
  fs::create_directories(target);
  std::ofstream(legacy / "play.toml") << "ancien";
  std::ofstream(target / "play.toml") << "nouveau";
  std::ofstream(root / "ajp" / "autre.txt") << "autre";
  CHECK_FALSE(adopt_legacy_dir(legacy, target));
  std::string text;
  // Lu et fermé tout de suite : Windows refuse de retirer un fichier encore ouvert.
  std::ifstream(target / "play.toml") >> text;
  CHECK(text == "nouveau");
  CHECK(fs::is_regular_file(legacy / "play.toml"));
  fs::remove_all(root);
}

TEST_CASE("Dossiers d'ajp : un seul niveau pour les caches (~/.cache/ajp)", "[app][legacy_dirs]") {
  const fs::path root = fs::temp_directory_path() / "kirari-legacy-cache-test";
  fs::remove_all(root);
  fs::create_directories(root / "ajp" / "engines");
  CHECK(adopt_legacy_dir(root / "ajp", root / "kirari"));  // la cible n'existe pas encore
  CHECK(fs::is_directory(root / "kirari" / "engines"));
  CHECK(fs::exists(root));  // le parent (~/.cache) ne s'appelle pas ajp : il reste
  fs::remove_all(root);
}

#if defined(__linux__)
TEST_CASE("Le banc à XDG isolé laisse la migration des caches à la prochaine lecture normale",
          "[app][legacy_dirs][migration_xdg]") {
  const fs::path root = fs::temp_directory_path() / "kirari-xdg-cache-migration";
  fs::remove_all(root);
  const auto home = root / "utilisateur";
  const auto data = home / ".local/share/kirari";
  const auto legacy = home / ".cache/ajp";
  const auto xdg = root / "autre-cache";
  fs::create_directories(legacy / "engines");
  fs::create_directories(data / "modeles");
  fs::create_directories(xdg / "ajp/plans");
  std::ofstream(legacy / "engines/rife.plan") << "moteur";
  const auto isolated = kirari::app::play::locate_launch_data(data, home, xdg, false);
  CHECK(isolated.cache == xdg / "kirari");
  CHECK_FALSE(fs::exists(isolated.cache));
  CHECK(fs::is_regular_file(legacy / "engines/rife.plan"));
  CHECK(fs::is_directory(data / "modeles"));
  CHECK(fs::is_directory(xdg / "ajp/plans"));

  const auto normal = kirari::app::play::locate_launch_data(data, home, xdg, true);
  CHECK(fs::is_regular_file(normal.cache / "engines/rife.plan"));
  CHECK(fs::is_directory(normal.cache / "modeles"));
  CHECK(fs::is_directory(normal.cache / "plans"));
  CHECK_FALSE(fs::exists(legacy / "engines"));
  CHECK_FALSE(fs::exists(data / "modeles"));
  CHECK_FALSE(fs::exists(xdg / "ajp"));
  fs::remove_all(root);
}
#endif
