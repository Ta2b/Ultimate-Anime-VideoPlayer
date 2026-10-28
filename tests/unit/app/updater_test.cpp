#include "app/play/updater.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::app::play;
namespace fs = std::filesystem;

namespace {

void write(const fs::path& file, const std::string& text) {
  fs::create_directories(file.parent_path());
  std::ofstream(file, std::ios::binary) << text;
}

std::string read(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

}  // namespace

TEST_CASE("mises à jour : quand vérifier", "[updater]") {
  UpdateState state;
  CHECK(update_due(state, "notify", "daily", 1000));        // jamais vérifié
  CHECK_FALSE(update_due(state, "never", "launch", 1000));  // Jamais
  state.last_check_unix = 1000;
  CHECK_FALSE(update_due(state, "notify", "daily", 1000 + 3600));
  CHECK(update_due(state, "notify", "daily", 1000 + 86400));
  CHECK_FALSE(update_due(state, "auto", "weekly", 1000 + 3 * 86400));
  CHECK(update_due(state, "auto", "weekly", 1000 + 7 * 86400));
  CHECK(update_due(state, "auto", "launch", 1001));
  CHECK(update_due(state, "auto", "daily", 500));  // l'horloge a reculé
}

TEST_CASE("mises à jour : l'état fait l'aller-retour", "[updater]") {
  UpdateState state{.last_check_unix = 1700000000,
                    .latest_version = "0.6.0",
                    .notes_url = "https://x/y",
                    .staged_version = "0.6.0",
                    .staged_file = "/d/a.tar.gz",
                    .previous_version = "0.5.0",
                    .last_error = "non"};
  const auto read_back = parse_update_state(format_update_state(state));
  REQUIRE(read_back);
  CHECK(*read_back == state);
  CHECK(load_update_state(fs::temp_directory_path() / "ajp-no-update.toml") == UpdateState{});
}

TEST_CASE("mises à jour : manifeste attendu selon le canal", "[updater]") {
  CHECK(manifest_url("a/b", "stable") == "https://github.com/a/b/releases/latest/download/manifest.toml");
  CHECK(manifest_url("a/b", "preview") == "https://github.com/a/b/releases/download/preview/manifest.toml");
}

#if !defined(_WIN32)
TEST_CASE("mises à jour : appliquer puis revenir en arrière", "[updater]") {
  const fs::path root = fs::temp_directory_path() / "ajp-updater-test";
  fs::remove_all(root);
  const fs::path install = root / "install";
  const fs::path data = root / "data";
  write(install / "ajp", "ancien");
  write(install / "lib" / "a.so", "lib ancienne");
  // L'archive de la release : un dossier racine.
  write(root / "src" / "ajp-linux-x64" / "ajp", "nouveau");
  write(root / "src" / "ajp-linux-x64" / "lib" / "a.so", "lib nouvelle");
  write(root / "src" / "ajp-linux-x64" / "lib" / "b.so", "lib en plus");
  const fs::path archive = data / "updates" / "0.6.0" / "ajp-linux-x64.tar.gz";
  fs::create_directories(archive.parent_path());
  const std::string command =
      "tar -czf '" + archive.string() + "' -C '" + (root / "src").string() + "' ajp-linux-x64";
  REQUIRE(std::system(command.c_str()) == 0);

  const UpdatePaths paths{.data = data, .install = install, .executable = install / "ajp"};
  UpdateState state{.latest_version = "0.6.0", .staged_version = "0.6.0", .staged_file = archive.string()};
  const auto applied = apply_staged_update(paths, state);
  REQUIRE(applied);
  CHECK(read(install / "ajp") == "nouveau");
  CHECK(read(install / "lib" / "a.so") == "lib nouvelle");
  CHECK(read(install / "lib" / "b.so") == "lib en plus");
  CHECK(read(data / "updates" / "previous" / "ajp") == "ancien");
  CHECK(state.staged_version.empty());
  CHECK_FALSE(state.previous_version.empty());

  const auto back = roll_back_update(paths, state);
  REQUIRE(back);
  CHECK(read(install / "ajp") == "ancien");
  CHECK(read(install / "lib" / "a.so") == "lib ancienne");
  CHECK_FALSE(roll_back_update(paths, state));  // plus rien à remettre

  // Sans archive préparée, rien ne se passe.
  UpdateState empty;
  CHECK_FALSE(apply_staged_update(paths, empty));
  fs::remove_all(root);
}
#endif

TEST_CASE("un AppImage ne se met pas à jour lui-même : le lecteur renvoie vers la page des releases", "[updater][appimage]") {
  const ajp::app::play::UpdatePaths paths{.data = "/tmp", .install = "/tmp", .executable = "/tmp/ajp"};
  setenv("AJP_PACKAGED", "appimage", 1);
  const std::string reason = ajp::app::play::update_managed_reason(paths);
  unsetenv("AJP_PACKAGED");
  CHECK(reason.starts_with("AppImage"));
  CHECK_FALSE(ajp::app::play::update_managed_reason(paths).starts_with("AppImage"));
}
