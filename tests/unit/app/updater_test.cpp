#include "app/play/updater.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/ed25519.h"

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
                    .last_error = "non",
                    .staged_sha256 = "ab",
                    .staged_size = 12,
                    .unconfirmed_version = "0.6.0",
                    .unconfirmed_launches = 2,
                    .skipped_version = "0.5.9"};
  const auto read_back = parse_update_state(format_update_state(state));
  REQUIRE(read_back);
  CHECK(*read_back == state);
  CHECK(load_update_state(fs::temp_directory_path() / "ajp-no-update.toml") == UpdateState{});
}

TEST_CASE("mises à jour : manifeste attendu selon le canal", "[updater]") {
  CHECK(manifest_url("a/b", "stable") == "https://github.com/a/b/releases/latest/download/manifest.toml");
  CHECK(manifest_url("a/b", "preview") == "https://github.com/a/b/releases/download/preview/manifest.toml");
}

TEST_CASE("mises à jour : un échec de curl se lit, un manifeste absent n'est pas une panne", "[updater]") {
  CHECK(curl_status_line("ajp-http 404") == 404);
  CHECK_FALSE(curl_status_line("curl: (22) The requested URL returned error: 404"));
  CHECK_FALSE(curl_status_line("ajp-http 404x"));
  CHECK(curl_failure(22, 404).code == ajp::RefusalCode::TrackUnavailable);
  const ajp::Refusal forbidden = curl_failure(22, 403);
  CHECK(forbidden.code != ajp::RefusalCode::TrackUnavailable);
  CHECK(forbidden.detail.find("403") != std::string::npos);
  for (const int code : {6, 7, 28, 35, 60, 3}) {
    const ajp::Refusal refusal = curl_failure(code, 0);
    CHECK(refusal.code != ajp::RefusalCode::TrackUnavailable);
    CHECK(refusal.detail.find("code 22") == std::string::npos);
    CHECK(!refusal.detail.empty());
  }
  CHECK(curl_failure(3, 0).detail.find("3") != std::string::npos);
}

#if !defined(_WIN32)
TEST_CASE("mises à jour : appliquer puis revenir en arrière", "[updater]") {
  const fs::path root = fs::temp_directory_path() / "ajp-updater-test";
  fs::remove_all(root);
  const fs::path install = root / "install";
  const fs::path data = root / "data";
  write(install / "ajp", "ancien");
  write(install / "lib" / "a.so", "lib ancienne");
  write(install / "lib" / "old.so", "retirée de la 0.6.0");
  write(install / ".ajp-files", "ajp\nlib/a.so\nlib/old.so\n");  // la liste de la version installée
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
  CHECK_FALSE(fs::exists(install / "lib" / "old.so"));  // plus livrée : retirée
  CHECK(read(data / "updates" / "previous" / "ajp") == "ancien");
  CHECK(state.staged_version.empty());
  CHECK_FALSE(state.previous_version.empty());

  const auto back = roll_back_update(paths, state);
  REQUIRE(back);
  CHECK(read(install / "ajp") == "ancien");
  CHECK(read(install / "lib" / "a.so") == "lib ancienne");
  CHECK(read(install / "lib" / "old.so") == "retirée de la 0.6.0");
  CHECK_FALSE(fs::exists(install / "lib" / "b.so"));  // ajoutée par la 0.6.0 : partie avec elle
  CHECK_FALSE(roll_back_update(paths, state));        // plus rien à remettre

  // Sans archive préparée, rien ne se passe.
  UpdateState empty;
  CHECK_FALSE(apply_staged_update(paths, empty));
  fs::remove_all(root);
}

TEST_CASE("mises à jour : archive revérifiée, lecteur exigé, garde qui remet l'ancienne", "[updater]") {
  const fs::path root = fs::temp_directory_path() / "ajp-updater-guard-test";
  fs::remove_all(root);
  const fs::path install = root / "install";
  const fs::path data = root / "data";
  write(install / "ajp", "ancien");
  write(root / "src" / "ajp-linux-x64" / "ajp", "nouveau");
  write(root / "vide" / "ajp-linux-x64" / "lisez-moi", "pas de lecteur");
  const fs::path archive = data / "updates" / "0.6.0" / "ajp-linux-x64.tar.gz";
  const fs::path hollow = data / "updates" / "0.6.0" / "vide.tar.gz";
  fs::create_directories(archive.parent_path());
  REQUIRE(
      std::system(("tar -czf '" + archive.string() + "' -C '" + (root / "src").string() + "' ajp-linux-x64")
                      .c_str()) == 0);
  REQUIRE(
      std::system(("tar -czf '" + hollow.string() + "' -C '" + (root / "vide").string() + "' ajp-linux-x64")
                      .c_str()) == 0);
  const UpdatePaths paths{.data = data, .install = install, .executable = install / "ajp"};

  // Une archive qui ne correspond plus à son SHA-256 : refusée, l'installation reste.
  UpdateState changed{.staged_version = "0.6.0",
                      .staged_file = archive.string(),
                      .staged_sha256 = std::string(64, '0'),
                      .staged_size = fs::file_size(archive)};
  CHECK_FALSE(apply_staged_update(paths, changed));
  CHECK(read(install / "ajp") == "ancien");
  REQUIRE(
      std::system(("tar -czf '" + archive.string() + "' -C '" + (root / "src").string() + "' ajp-linux-x64")
                      .c_str()) == 0);

  // Une archive sans le lecteur : refusée.
  UpdateState empty_archive{.staged_version = "0.6.0", .staged_file = hollow.string()};
  CHECK_FALSE(apply_staged_update(paths, empty_archive));
  CHECK(read(install / "ajp") == "ancien");

  // Appliquée, puis des lancements sans lecture menée à son terme.
  UpdateState state{.staged_version = "0.6.0", .staged_file = archive.string()};
  REQUIRE(apply_staged_update(paths, state));
  CHECK(state.unconfirmed_version == "0.6.0");
  write(install / "lib.so.old", "reste");
  // Des fermetures voulues (launch_open retiré à la sortie) ne comptent pas.
  for (int launch = 1; launch <= 2 * kMaxUnconfirmedLaunches; ++launch) {
    CHECK_FALSE(guard_new_version(paths, state, "0.6.0"));
    CHECK(close_launch(state));
  }
  CHECK(state.unconfirmed_launches == 0);
  // Puis des lancements qui plantent (jamais refermés) : après trois, l'ancienne revient.
  for (int launch = 1; launch <= kMaxUnconfirmedLaunches; ++launch)
    CHECK_FALSE(guard_new_version(paths, state, "0.6.0"));
  CHECK_FALSE(fs::exists(install / "lib.so.old"));
  CHECK(guard_new_version(paths, state, "0.6.0"));
  CHECK(read(install / "ajp") == "ancien");
  CHECK(state.skipped_version == "0.6.0");
  CHECK(state.unconfirmed_version.empty());

  // Une version confirmée par une lecture : la garde ne fait plus rien.
  UpdateState good{.unconfirmed_version = "0.7.0", .unconfirmed_launches = 1};
  CHECK_FALSE(confirm_version(good, "0.6.0"));
  CHECK(confirm_version(good, "0.7.0"));
  CHECK_FALSE(guard_new_version(paths, good, "0.7.0"));
  fs::remove_all(root);
}
#endif

// Linux seulement : pas d'AppImage sous macOS (update_managed_reason y répond d'abord « page des releases »)
// ni sous Windows (setenv et unsetenv y sont absents).
#if !defined(__APPLE__) && !defined(_WIN32)
TEST_CASE("un AppImage ne se met pas à jour lui-même : le lecteur renvoie vers la page des releases",
          "[updater][appimage]") {
  const ajp::app::play::UpdatePaths paths{.data = "/tmp", .install = "/tmp", .executable = "/tmp/ajp"};
  setenv("AJP_PACKAGED", "appimage", 1);
  const std::string reason = ajp::app::play::update_managed_reason(paths);
  unsetenv("AJP_PACKAGED");
  CHECK(reason.starts_with("AppImage"));
  CHECK_FALSE(ajp::app::play::update_managed_reason(paths).starts_with("AppImage"));
}
#endif

TEST_CASE("mises à jour : signature fausse, rien d'autre n'est téléchargé", "[updater]") {
  std::vector<std::string> asked;
  const UpdateFetch fake = [&](std::string_view url, const fs::path& file, int,
                               const std::stop_token&) -> std::expected<void, ajp::Refusal> {
    asked.emplace_back(url);
    write(file, url.ends_with(".sig") ? std::string(128, '0') : std::string("version = \"9.0.0\"\n"));
    return {};
  };
  const auto checked = check_for_update("stable", std::stop_token{}, fake,
                                        {"c438afee757ea9095f5e9d56380da968bfca8af4ff272c4773b51808bca6ce2d"});
  REQUIRE_FALSE(checked);
  CHECK(checked.error().code == ajp::RefusalCode::HashMismatch);
  REQUIRE(asked.size() == 2);  // le manifeste et sa signature, jamais une archive
  CHECK(asked[1].ends_with("manifest.toml.sig"));
}

#if defined(__linux__)
// De bout en bout, comme le lecteur : vérifier (manifeste signé par une clé de test), préparer (archive
// téléchargée et vérifiée), relancer (l'état écrit puis relu), appliquer.
TEST_CASE("mises à jour : vérifier, préparer, relancer, appliquer", "[updater]") {
  const std::string manifest =
      "version = \"99.0.0\"\nchannel = \"stable\"\ndate = \"2026-11-07\"\nexpires = \"2099-12-31\"\n"
      "[linux-x64]\nurl = \"https://example.invalid/ajp-linux-x64.tar.gz\"\n"
      "sha256 = \"6ec20ccae9a53cf2b47c765c124b7de6bdaed3b7f8a438381501cbde525b9638\"\nsize = 138\n";
  // tar.gz de ajp-linux-x64/ajp (« nouveau »), écrit par tar --sort=name --mtime=2026-01-01 | gzip -n.
  const auto archive = from_hex(
      "1f8b0800000000000203edd13b0ec2301000d13d8a2f106127bbf6795c5080a280128c727c4c3a52d01909655eb39f76f2f5de"
      "8d97a9"
      "acdd1af5244df82a996db3dacf6d0f167a1b06d5f73ff83e2671263f5096479e9d9383ca1ffdebd5a87f54fdd23fedfba718c4"
      "79fa37"
      "37ddcaf39c8b00000000000000000000000000f83f2f45882f0700280000");
  REQUIRE(archive);
  const UpdateFetch fake = [&](std::string_view url, const fs::path& file, int,
                               const std::stop_token&) -> std::expected<void, ajp::Refusal> {
    if (url.ends_with(".sig"))
      write(file,
            "722f81d4e2982752f0d8a6d65437ce8e86e1ab86e9b775285c9d650beae012a5e2c94162d4877cd76f9362f6fa7ffef8"
            "d0434"
            "25e45b5d334daee5fde78258106\n");
    else if (url.ends_with(".tar.gz"))
      write(file, std::string(archive->begin(), archive->end()));
    else
      write(file, manifest);
    return {};
  };
  const fs::path root = fs::temp_directory_path() / "ajp-updater-e2e";
  fs::remove_all(root);
  const UpdatePaths paths{
      .data = root / "data", .install = root / "install", .executable = root / "install" / "ajp"};
  write(paths.executable, "ancien");

  // La clé de test vient en second, comme une clé de secours : le manifeste qu'elle signe est cru aussi.
  const auto checked = check_for_update("stable", std::stop_token{}, fake,
                                        {"c438afee757ea9095f5e9d56380da968bfca8af4ff272c4773b51808bca6ce2d",
                                         "e1a95bd8c0582a15d671a96fef74c36952d67ee243b4e5570ab44bd22015b18c"});
  REQUIRE(checked);
  REQUIRE(checked->newer);
  const auto file = download_update(paths, *checked->newer, std::stop_token{}, fake);
  REQUIRE(file);
  UpdateState prepared{.staged_version = "99.0.0", .staged_file = file->string()};
  prepared.staged_sha256 = checked->newer->linux_x64->sha256;
  prepared.staged_size = checked->newer->linux_x64->size;
  auto relaunched = parse_update_state(format_update_state(prepared));  // le lecteur repart
  REQUIRE(relaunched);
  const auto applied = apply_staged_update(paths, *relaunched);
  REQUIRE(applied);
  CHECK(read(paths.executable) == "nouveau");
  CHECK(relaunched->unconfirmed_version == "99.0.0");
  fs::remove_all(root);
}
#endif

TEST_CASE("mises à jour : une clé révoquée n'est plus crue", "[updater]") {
  const std::vector<std::string> keys = update_public_keys({});
  REQUIRE_FALSE(keys.empty());
  const std::vector<std::string> left = update_public_keys({keys.front()});
  CHECK(std::ranges::find(left, keys.front()) == left.end());
  // Gardée dans update.toml d'un lancement à l'autre.
  const auto state = parse_update_state(format_update_state(UpdateState{.revoked_keys = {keys.front()}}));
  REQUIRE(state);
  CHECK(state->revoked_keys == std::vector<std::string>{keys.front()});
}
