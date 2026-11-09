#include "app/play/updater.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/data_dirs.h"

using namespace ajp::app::play;
namespace fs = std::filesystem;

namespace {

void write(const fs::path& file, const std::string& text) {
  fs::create_directories(file.parent_path());
  std::ofstream(file, std::ios::binary) << text;
}

// Un manifeste du canal stable signé par la clé de test kTestKey (sa signature : kTestSignature).
constexpr std::string_view kSignedManifest =
    "version = \"99.0.0\"\nchannel = \"stable\"\ndate = \"2026-11-07\"\nexpires = \"2099-12-31\"\n"
    "[linux-x64]\nurl = \"https://example.invalid/ajp-linux-x64.tar.gz\"\n"
    "sha256 = \"6ec20ccae9a53cf2b47c765c124b7de6bdaed3b7f8a438381501cbde525b9638\"\nsize = 138\n";
constexpr std::string_view kTestSignature =
    "722f81d4e2982752f0d8a6d65437ce8e86e1ab86e9b775285c9d650beae012a5e2c94162d4877cd76f9362f6fa7ffef8d0434"
    "25e45b5d334daee5fde78258106\n";
constexpr std::string_view kTestKey = "e1a95bd8c0582a15d671a96fef74c36952d67ee243b4e5570ab44bd22015b18c";
constexpr std::string_view kOtherKey = "c438afee757ea9095f5e9d56380da968bfca8af4ff272c4773b51808bca6ce2d";

// Le serveur de la release : le manifeste signé et sa signature ; chaque adresse demandée est notée.
UpdateFetch signed_release(std::vector<std::string>& asked) {
  return [&asked](std::string_view url, const fs::path& file, int,
                  const std::stop_token&) -> std::expected<void, ajp::Refusal> {
    asked.emplace_back(url);
    write(file, std::string(url.ends_with(".sig") ? kTestSignature : kSignedManifest));
    return {};
  };
}

}  // namespace

TEST_CASE("mises à jour : quand vérifier", "[updater]") {
  UpdateState state;
  CHECK(update_due(state, UpdateMode::Notify, UpdateFrequency::Daily, 1000));        // jamais vérifié
  CHECK_FALSE(update_due(state, UpdateMode::Never, UpdateFrequency::Launch, 1000));  // Jamais
  state.last_check_unix = 1000;
  state.last_attempt_unix = 1000;
  CHECK_FALSE(update_due(state, UpdateMode::Notify, UpdateFrequency::Daily, 1000 + 3600));
  CHECK(update_due(state, UpdateMode::Notify, UpdateFrequency::Daily, 1000 + 86400));
  CHECK_FALSE(update_due(state, UpdateMode::Notify, UpdateFrequency::Weekly, 1000 + 3 * 86400));
  CHECK(update_due(state, UpdateMode::Notify, UpdateFrequency::Weekly, 1000 + 7 * 86400));
  CHECK(update_due(state, UpdateMode::Notify, UpdateFrequency::Launch, 1001));
  CHECK(update_due(state, UpdateMode::Notify, UpdateFrequency::Daily, 500));  // l'horloge a reculé
}

TEST_CASE("mises à jour : une tentative sans réseau se refait au bout d'une heure, pas d'une semaine",
          "[updater]") {
  // A26-07 : la dernière réponse date de huit jours, la dernière tentative (sans réponse) de dix minutes.
  const std::int64_t now = 100 * 86400;
  UpdateState state{.last_check_unix = now - 8 * 86400, .last_attempt_unix = now - 600};
  CHECK_FALSE(update_due(state, UpdateMode::Notify, UpdateFrequency::Weekly, now));
  CHECK(update_due(state, UpdateMode::Notify, UpdateFrequency::Weekly, now - 600 + kUpdateRetrySeconds));
}

TEST_CASE("mises à jour : l'état fait l'aller-retour", "[updater]") {
  UpdateState state{.last_check_unix = 1700000000,
                    .last_attempt_unix = 1700000100,
                    .latest_version = "0.6.0",
                    .notes_url = "https://x/y",
                    .last_error = "non",
                    .revoked_keys = {"ab"}};
  const auto read_back = parse_update_state(format_update_state(state));
  REQUIRE(read_back);
  CHECK(*read_back == state);
  CHECK(load_update_state(fs::temp_directory_path() / "ajp-no-update.toml") == UpdateState{});
}

TEST_CASE("mises à jour : l'update.toml d'une version qui s'appliquait elle-même se relit", "[updater]") {
  // Écrit par la 0.10.3 (préparation, garde, retour arrière) : ses champs sont ignorés, les autres gardés.
  const auto state = parse_update_state(
      "last_check_unix = 1700000000\nlatest_version = \"0.11.0\"\nnotes_url = \"https://x/y\"\n"
      "staged_version = \"0.11.0\"\nstaged_file = \"/d/a.tar.gz\"\nprevious_version = \"0.10.2\"\n"
      "last_error = \"\"\nstaged_sha256 = \"ab\"\nstaged_size = 12\nunconfirmed_version = \"0.10.3\"\n"
      "unconfirmed_launches = 2\nskipped_version = \"0.10.1\"\nlaunch_open = true\nrevoked_keys = "
      "[\"cd\"]\n");
  REQUIRE(state);
  CHECK(state->last_check_unix == 1700000000);
  CHECK(state->latest_version == "0.11.0");
  CHECK(state->notes_url == "https://x/y");
  CHECK(state->revoked_keys == std::vector<std::string>{"cd"});
}

TEST_CASE("mises à jour : un seul manifeste, celui de la dernière release", "[updater]") {
  CHECK(manifest_url("a/b") == "https://github.com/a/b/releases/latest/download/manifest.toml");
  CHECK(releases_page("a/b") == "https://github.com/a/b/releases/latest");
}

TEST_CASE("mises à jour : un échec de curl se lit", "[updater]") {
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

TEST_CASE("mises à jour : signature fausse, rien d'autre n'est téléchargé", "[updater]") {
  std::vector<std::string> asked;
  const auto checked =
      check_for_update(std::stop_token{}, signed_release(asked), {std::string(kOtherKey)}, "0.10.3");
  REQUIRE_FALSE(checked);
  CHECK(checked.error().code == ajp::RefusalCode::HashMismatch);
  REQUIRE(asked.size() == 2);  // le manifeste et sa signature, jamais une archive
  CHECK(asked[1].ends_with("manifest.toml.sig"));
}

TEST_CASE("mises à jour : une version signée plus récente se signale, sans rien télécharger d'autre",
          "[updater]") {
  std::vector<std::string> asked;
  // La clé de test vient en second, comme une clé de secours : le manifeste qu'elle signe est cru aussi.
  const std::vector<std::string> keys{std::string(kOtherKey), std::string(kTestKey)};
  const auto older = check_for_update(std::stop_token{}, signed_release(asked), keys, "0.10.3");
  REQUIRE(older);
  REQUIRE(older->newer);
  CHECK(older->newer->version == "99.0.0");
  CHECK(asked.size() == 2);
  // Une préversion de la même version est plus ancienne que la finale (A26-04).
  const auto preview = check_for_update(std::stop_token{}, signed_release(asked), keys, "99.0.0-rc.1");
  REQUIRE(preview);
  CHECK(preview->newer);
  const auto same = check_for_update(std::stop_token{}, signed_release(asked), keys, "99.0.0");
  REQUIRE(same);
  CHECK_FALSE(same->newer);
}

TEST_CASE("mises à jour : une release sans manifeste n'est jamais lue « à jour »", "[updater]") {
  // SEC-05 : un 404 sur le manifeste de « latest » est un refus dit, pas « le lecteur est à jour ».
  const UpdateFetch missing = [](std::string_view, const fs::path&, int,
                                 const std::stop_token&) -> std::expected<void, ajp::Refusal> {
    return std::unexpected(curl_failure(22, 404));
  };
  UpdateState state{.latest_version = "0.10.0"};
  const auto checked = run_update_check(state, std::stop_token{}, missing, 5000);
  REQUIRE_FALSE(checked);
  CHECK(checked.error().code == ajp::RefusalCode::TrackUnavailable);
  // Le serveur a répondu : la vérification compte, et l'erreur se garde.
  CHECK(state.last_check_unix == 5000);
  CHECK(state.last_attempt_unix == 5000);
  CHECK_FALSE(state.last_error.empty());
}

TEST_CASE("mises à jour : sans réseau, seule la tentative est notée", "[updater]") {
  const UpdateFetch offline = [](std::string_view, const fs::path&, int,
                                 const std::stop_token&) -> std::expected<void, ajp::Refusal> {
    return std::unexpected(curl_failure(6, 0));
  };
  UpdateState state{.last_check_unix = 1000};
  REQUIRE_FALSE(run_update_check(state, std::stop_token{}, offline, 9000));
  CHECK(state.last_check_unix == 1000);
  CHECK(state.last_attempt_unix == 9000);
}

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

TEST_CASE("mises à jour : les archives préparées d'avant la simple notification se retirent", "[updater]") {
  const fs::path root = fs::temp_directory_path() / "ajp-update-leftovers";
  fs::remove_all(root);
  write(root / "updates" / "0.10.3" / "ajp-linux-x64.tar.gz", "archive");
  write(root / "updates" / "previous" / "ajp", "ancien");
  CHECK(purge_update_leftovers(root) == 1);
  CHECK_FALSE(fs::exists(root / "updates"));
  // Un dossier qui contient autre chose que les nôtres reste.
  write(root / "updates" / "notes.txt", "à moi");
  CHECK(purge_update_leftovers(root) == 0);
  CHECK(fs::exists(root / "updates" / "notes.txt"));
  fs::remove_all(root);
}
