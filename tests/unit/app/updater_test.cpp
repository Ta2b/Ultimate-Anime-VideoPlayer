#include "app/play/updater.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/data_dirs.h"
#include "app/play/machine_calibration.h"

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

// Deux clés de test, A (l'active) et B (la clé de secours), et des manifestes qu'elles signent (openssl
// pkeyutl -sign -rawin) : version 99.0.0, valables jusqu'en 2099, avec chacun sa liste de clés révoquées.
constexpr std::string_view kKeyA = "d7a96b1b6b7414d7549caa4e9ea50fe3062948f28c43ddadc6b600630bd4c6db";
constexpr std::string_view kKeyB = "0151614c9d7df76a095a922f18042ff9014be3dca017f987e7fa1b2dbd4befe8";
struct SignedFile {
  std::string_view text;
  std::string_view signature;
};
constexpr std::string_view kARevokesBText =
    "version = \"99.0.0\"\n"
    "expires = \"2099-12-31\"\n"
    "revoked_keys = [\"0151614c9d7df76a095a922f18042ff9014be3dca017f987e7fa1b2dbd4befe8\"]\n";
constexpr std::string_view kARevokesBSig =
    "2ab81cecf78e00cb5e86f7093daf589f19ba21674b5f00ef0ea84ce0178da99324d2bfd025b222b60f3ed3809da56f42"
    "967d71726ecb01cb3794d7960f901805";
constexpr std::string_view kBRevokesAText =
    "version = \"99.0.0\"\n"
    "expires = \"2099-12-31\"\n"
    "revoked_keys = [\"d7a96b1b6b7414d7549caa4e9ea50fe3062948f28c43ddadc6b600630bd4c6db\"]\n";
constexpr std::string_view kBRevokesASig =
    "008c97b4ec15a78362661a2a0d9683fe475bb1517f4aa7b8e7b0ba6c6b5521e20551042f25909800e814b97aba193785"
    "4397f5e12f94833ffb2c0158fa14dc0b";
constexpr std::string_view kARevokesOtherText =
    "version = \"99.0.0\"\n"
    "expires = \"2099-12-31\"\n"
    "revoked_keys = [\"1111111111111111111111111111111111111111111111111111111111111111\"]\n";
constexpr std::string_view kARevokesOtherSig =
    "1d745a7b7b14a39c3b53eafdd235a06663c9a74b8b5a5f4c11640f6528195fc21b0df6426a358f710603fc7c5a9a9baa"
    "72fe6af694fad9987b0d5b7050cc5e0d";
constexpr std::string_view kBRevokesSelfText =
    "version = \"99.0.0\"\n"
    "expires = \"2099-12-31\"\n"
    "revoked_keys = [\"0151614c9d7df76a095a922f18042ff9014be3dca017f987e7fa1b2dbd4befe8\"]\n";
constexpr std::string_view kBRevokesSelfSig =
    "fb0f6834023f03097da2a15c8eee79b85127e7114b19c4564b2e494fe40827f2547e80964e86b479df457c520d8c1645"
    "e67f2e4e9b414e7977318badfbe93404";

// Le serveur d'une release dont le manifeste et la signature sont ceux-là (la signature peut ne pas aller
// avec).
UpdateFetch release_of(std::string_view text, std::string_view signature, int& calls) {
  return [text, signature, &calls](std::string_view url, const fs::path& file, int,
                                   const std::stop_token&) -> std::expected<void, ajp::Refusal> {
    ++calls;
    write(file, std::string(url.ends_with(".sig") ? signature : text));
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

TEST_CASE("mises à jour : la version installée ne s'annonce plus après l'installation (B26-01)",
          "[updater]") {
  UpdateState state{.latest_version = "0.9.2", .notes_url = "https://x/y"};
  forget_installed_announce(state, "0.9.2");  // installée : plus rien à annoncer
  CHECK(state.latest_version.empty());
  CHECK(state.notes_url.empty());
  state = {.latest_version = "0.9.2", .notes_url = "https://x/y"};
  forget_installed_announce(state, "0.10.0");  // plus ancienne que l'installée
  CHECK(state.latest_version.empty());
  state = {.latest_version = "0.9.2", .notes_url = "https://x/y"};
  forget_installed_announce(state, "0.9.1");  // encore nouvelle : gardée
  CHECK(state.latest_version == "0.9.2");
  CHECK(state.notes_url == "https://x/y");
  state = {.latest_version = "pas une version", .notes_url = "https://x/y"};
  forget_installed_announce(state, "0.9.1");  // illisible : rien d'annoncé
  CHECK(state.latest_version.empty());

  // Par le fichier : le lecteur relancé après l'installation lit update.toml tel que la vérification l'a
  // laissé.
  const fs::path file = fs::temp_directory_path() / "ajp-update-installed" / "update.toml";
  write(file, std::format("latest_version = \"{}\"\nnotes_url = \"https://x/y\"\nrevoked_keys = [\"cd\"]\n",
                          installed_version()));
  const UpdateState read_back = load_update_state(file);
  CHECK(read_back.latest_version.empty());
  CHECK(read_back.revoked_keys == std::vector<std::string>{"cd"});
  fs::remove_all(file.parent_path());
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

TEST_CASE("mises à jour : l'update.toml d'une version future se relit sans perdre les révocations (A36-09)",
          "[updater]") {
  // Un champ, un type et une table inconnus : les clés connues restent, les clés révoquées surtout.
  const auto state = parse_update_state(
      "last_check_unix = 1800000000\nrevoked_keys = [\"ef\"]\nchannel_hint = \"stable\"\nretries = [1, 2]\n"
      "[futur]\nmirror = \"https://m\"\n");
  REQUIRE(state);
  CHECK(state->last_check_unix == 1800000000);
  CHECK(state->revoked_keys == std::vector<std::string>{"ef"});
  const auto reread = parse_update_state(format_update_state(*state));
  REQUIRE(reread);
  CHECK(*reread == *state);
}

TEST_CASE("mises à jour : la réécriture garde les clés d'une version future, retire celles d'avant (A36-05)",
          "[updater]") {
  const std::string previous =
      "last_check_unix = 1\nchannel_hint = \"stable\"\nretries = [1, 2]\nstaged_version = \"0.9.0\"\n"
      "previous_version = \"0.8.0\"\nunconfirmed_launches = 2\n[futur]\nmirror = \"https://m\"\n";
  const std::string merged =
      merge_update_state(previous, UpdateState{.last_check_unix = 7, .latest_version = "1.0.0"});
  CHECK(merged.contains("channel_hint"));
  CHECK(merged.contains("https://m"));
  CHECK_FALSE(merged.contains("staged_version"));
  CHECK_FALSE(merged.contains("previous_version"));
  CHECK_FALSE(merged.contains("unconfirmed_"));
  const auto reread = parse_update_state(merged);
  REQUIRE(reread);
  CHECK(reread->last_check_unix == 7);
  CHECK(reread->latest_version == "1.0.0");
  // Un ancien fichier illisible ne bloque pas l'écriture.
  CHECK(
      parse_update_state(merge_update_state("= cassé", UpdateState{.last_check_unix = 3}))->last_check_unix ==
      3);
}

TEST_CASE("mises à jour : deux fils qui changent update.toml ne perdent rien de l'autre (A26-10)",
          "[updater]") {
  const std::filesystem::path file = std::filesystem::temp_directory_path() / "ajp-update-concurrent.toml";
  std::filesystem::remove(file);
  std::vector<std::thread> writers;
  std::atomic<int> refused = 0;  // Catch2 ne vérifie pas depuis un autre fil
  for (int i = 0; i < 8; ++i)
    writers.emplace_back([&file, &refused, i] {
      const auto kept = change_update_state(file, [i](UpdateState& state) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));  // une vérification qui prend du temps
        state.revoked_keys.push_back(std::to_string(i));
      });
      if (!kept) ++refused;
    });
  for (std::thread& writer : writers) writer.join();
  CHECK(refused == 0);
  CHECK(load_update_state(file).revoked_keys.size() == 8);
  std::filesystem::remove(file);
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
  // Le manifeste et sa signature, deux fois (une release qui change entre les deux requêtes donne une
  // signature fausse : B26-03), jamais une archive.
  REQUIRE(asked.size() == 4);
  CHECK(asked[1].ends_with("manifest.toml.sig"));
  CHECK(asked[3].ends_with("manifest.toml.sig"));
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

TEST_CASE("banc : les restes d'une calibration tuée et des clips d'avant se retirent (B36-09)", "[updater]") {
  const fs::path root = fs::temp_directory_path() / "ajp-bench-leftovers";
  fs::remove_all(root);
  write(root / "calibration-essais-1a" / "essai.log", "vieux");
  write(root / "calibration-essais-2b" / "essai.log", "récent");
  write(root / "bench-clip-v3.mkv", "clip d'avant");
  write(root / std::string(kBenchClipName), "clip d'aujourd'hui");
  write(root / "calibration.toml", "à moi");
  write(root / "bench-clip-notes.txt", "à moi aussi");
  // Une calibration tuée il y a deux jours ; l'autre tourne peut-être encore.
  fs::last_write_time(root / "calibration-essais-1a",
                      fs::file_time_type::clock::now() - std::chrono::hours(48));

  CHECK(purge_bench_leftovers(root) == 2);
  CHECK_FALSE(fs::exists(root / "calibration-essais-1a"));
  CHECK(fs::exists(root / "calibration-essais-2b"));
  CHECK_FALSE(fs::exists(root / "bench-clip-v3.mkv"));
  CHECK(fs::exists(root / std::string(kBenchClipName)));
  CHECK(fs::exists(root / "calibration.toml"));
  CHECK(fs::exists(root / "bench-clip-notes.txt"));
  CHECK(purge_bench_leftovers(root) == 0);
}

TEST_CASE(
    "mises à jour : la clé active ne révoque pas la clé de secours, celle-ci peut révoquer l'active (B26-02)",
    "[updater]") {
  const std::vector<std::string> keys{std::string(kKeyA), std::string(kKeyB)};
  int calls = 0;
  // A tient le manifeste mais ne peut pas retirer B : la révocation est ignorée, la mise à jour se lit.
  auto checked =
      check_for_update(std::stop_token{}, release_of(kARevokesBText, kARevokesBSig, calls), keys, "0.1.0");
  REQUIRE(checked);
  CHECK(checked->revoked_keys.empty());
  CHECK(checked->newer);
  // B, la clé de secours, révoque A.
  checked =
      check_for_update(std::stop_token{}, release_of(kBRevokesAText, kBRevokesASig, calls), keys, "0.1.0");
  REQUIRE(checked);
  CHECK(checked->revoked_keys == std::vector<std::string>{std::string(kKeyA)});
  // Une clé que le lecteur ne connaît pas n'est pas retenue (C26-05) : elle ne signe rien ici, et la garder
  // couperait les mises à jour d'une version future qui la compilerait.
  checked = check_for_update(std::stop_token{}, release_of(kARevokesOtherText, kARevokesOtherSig, calls),
                             keys, "0.1.0");
  REQUIRE(checked);
  CHECK(checked->revoked_keys.empty());
  // Une clé qui se révoque elle-même n'est pas crue.
  const auto itself = check_for_update(std::stop_token{},
                                       release_of(kBRevokesSelfText, kBRevokesSelfSig, calls), keys, "0.1.0");
  REQUIRE_FALSE(itself);
  CHECK(itself.error().code == ajp::RefusalCode::HashMismatch);
  // Une fois A révoquée, ses manifestes ne se vérifient plus (le lecteur ne lui laisse que B).
  const std::vector<std::string> only_b{std::string(kKeyB)};
  const auto old_key = check_for_update(
      std::stop_token{}, release_of(kARevokesOtherText, kARevokesOtherSig, calls), only_b, "0.1.0");
  CHECK_FALSE(old_key);
}

TEST_CASE(
    "mises à jour : une signature venue d'une autre release se refait une fois, puis se refuse (B26-03)",
    "[updater]") {
  const std::vector<std::string> keys{std::string(kKeyA), std::string(kKeyB)};
  // Première paire : le manifeste de l'ancienne release avec la signature de la neuve ; la seconde est la
  // bonne.
  int calls = 0;
  const UpdateFetch changing = [&](std::string_view url, const fs::path& file, int,
                                   const std::stop_token&) -> std::expected<void, ajp::Refusal> {
    const bool second_round = calls >= 2;
    ++calls;
    write(file, std::string(url.ends_with(".sig") ? (second_round ? kBRevokesASig : kARevokesOtherSig)
                                                  : kBRevokesAText));
    return {};
  };
  const auto checked = check_for_update(std::stop_token{}, changing, keys, "0.1.0");
  REQUIRE(checked);
  CHECK(calls == 4);
  // Une signature absente du serveur (404) se lit comme fausse, après la seconde tentative.
  calls = 0;
  const UpdateFetch unsigned_release = [&](std::string_view url, const fs::path& file, int,
                                           const std::stop_token&) -> std::expected<void, ajp::Refusal> {
    ++calls;
    if (url.ends_with(".sig")) return std::unexpected(curl_failure(22, 404));
    write(file, std::string(kBRevokesAText));
    return {};
  };
  const auto missing = check_for_update(std::stop_token{}, unsigned_release, keys, "0.1.0");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == ajp::RefusalCode::HashMismatch);
  CHECK(calls == 4);
}

TEST_CASE("mises à jour : annulée ou sans signature, la vérification ne compte pas comme faite (B26-03)",
          "[updater]") {
  // Le manifeste arrive, la signature est coupée (fermeture du lecteur) : seule la tentative est notée.
  const UpdateFetch cancelled = [](std::string_view url, const fs::path& file, int,
                                   const std::stop_token&) -> std::expected<void, ajp::Refusal> {
    if (url.ends_with(".sig")) return ajp::refuse(ajp::RefusalCode::Cancelled, "annulé");
    write(file, std::string(kSignedManifest));
    return {};
  };
  UpdateState state{.last_check_unix = 1000};
  REQUIRE_FALSE(run_update_check(state, std::stop_token{}, cancelled, 9000));
  CHECK(state.last_check_unix == 1000);
  CHECK(state.last_attempt_unix == 9000);
  // Un manifeste reçu mais dont la signature n'est pas celle d'une clé crue : pas non plus une vérification
  // faite.
  state = UpdateState{.last_check_unix = 1000};
  std::vector<std::string> asked;
  REQUIRE_FALSE(run_update_check(state, std::stop_token{}, signed_release(asked), 9000));
  CHECK(state.last_check_unix == 1000);
}

TEST_CASE("mises à jour : une clé révoquée s'écrit en majuscules ou en minuscules (B26-04)", "[updater]") {
  const std::vector<std::string> keys = update_public_keys({});
  REQUIRE_FALSE(keys.empty());
  std::string upper = keys.front();
  for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  const std::vector<std::string> left = update_public_keys({upper});
  CHECK(std::ranges::find(left, keys.front()) == left.end());
  // Les clés rendues sont en minuscules, comme celles que lit un manifeste.
  for (const std::string& key : keys)
    CHECK(key == [&] {
      std::string lower = key;
      for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      return lower;
    }());
}

TEST_CASE("mises à jour : un update.toml illisible garde ses clés révoquées et sa copie (B26-10)",
          "[updater]") {
  const fs::path root = fs::temp_directory_path() / "ajp-update-corrupt";
  fs::remove_all(root);
  const std::string revoked(64, 'a');
  const fs::path file = root / "update.toml";
  // Un fichier tronqué en plein milieu de sa liste de clés (TOML invalide).
  write(file, "last_check_unix = 12\nrevoked_keys = [\"" + revoked + "\", \"" + std::string(64, 'B') +
                  "\", \"ab\n");
  UpdateState state = load_update_state(file);
  CHECK(state.last_check_unix == 0);
  CHECK(state.revoked_keys == std::vector<std::string>{revoked, std::string(64, 'b')});
  // La réécriture garde la copie de l'illisible et les clés.
  state.last_check_unix = 5;
  REQUIRE(save_update_state(file, state));
  CHECK(fs::exists(root / "update.toml.bak"));
  const UpdateState reread = load_update_state(file);
  CHECK(reread.last_check_unix == 5);
  CHECK(reread.revoked_keys == state.revoked_keys);
  fs::remove_all(root);
}

TEST_CASE("mises à jour : le verrou de l'état n'est pas tenu pendant le téléchargement (B26-09, B30-02)",
          "[updater]") {
  const fs::path root = fs::temp_directory_path() / "ajp-update-lock";
  fs::remove_all(root);
  const fs::path file = root / "update.toml";
  std::atomic<bool> downloading{false};
  std::atomic<bool> release{false};
  const UpdateFetch slow = [&](std::string_view, const fs::path&, int,
                               const std::stop_token&) -> std::expected<void, ajp::Refusal> {
    downloading = true;
    while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return std::unexpected(curl_failure(6, 0));  // pas de réseau
  };
  std::thread checking([&] { (void)check_updates_now(file, std::stop_token{}, slow); });
  while (!downloading) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  // Pendant que curl attend, une autre tâche (le bouton, la fermeture) lit et écrit l'état sans attendre.
  auto other = std::async(std::launch::async, [&] {
    return change_update_state(file,
                               [](UpdateState& state) { state.notes_url = "https://example.invalid/n"; });
  });
  const bool free_lock = other.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  release = true;
  checking.join();
  REQUIRE(free_lock);
  REQUIRE(other.get());
  // La vérification ne réécrit que ses champs : la note posée entre-temps reste, la tentative est notée.
  const UpdateState kept = load_update_state(file);
  CHECK(kept.notes_url == "https://example.invalid/n");
  CHECK(kept.last_attempt_unix > 0);
  CHECK_FALSE(kept.last_error.empty());
  fs::remove_all(root);
}

// Le bouton « Download page » ne mène que sous le dépôt des releases (C26-04).
TEST_CASE("mises à jour : la page de téléchargement reste dans le dépôt des releases (C26-04)", "[updater]") {
  const std::string inside = "https://github.com/o/r/releases/tag/v1.0.0";
  CHECK(download_page_url(inside, "o/r") == inside);
  const std::string page = releases_page("o/r");
  CHECK(download_page_url("https://example.org/install-me", "o/r") == page);
  CHECK(download_page_url("https://github.com/autre/depot/releases", "o/r") == page);
  CHECK(download_page_url("https://github.com/o/r/../../x/y", "o/r") == page);
  CHECK(download_page_url("https://github.com/o/r/", "o/r") == page);
  CHECK(download_page_url("http://github.com/o/r/releases", "o/r") == page);
  CHECK(download_page_url("", "o/r") == page);
}
