#include "app/play/legacy_dirs.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

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
