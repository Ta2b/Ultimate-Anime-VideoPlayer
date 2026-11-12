// La relance du lecteur (C18-10) : relaunch.txt écrit puis relu, sa péremption, le processus relancé qui le
// prend, et les arguments rejoués avec leur citation pour _spawnv.
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/options.h"
#include "app/play/player.h"
#include "app/play/relaunch_args.h"
#include "app/play/relaunch_target.h"

namespace ajp::app::play {
namespace {

std::filesystem::path fresh_dir(std::string_view name) {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / name;
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

TEST_CASE("relaunch.txt : aller-retour, instant et pause gardés, lu une fois", "[app][relaunch]") {
  const std::filesystem::path store = fresh_dir("ajp-relaunch-round") / "relaunch.txt";
  const RelaunchTarget written{.file = "/vidéos/Épisode 01.mkv",
                               .start_us = 754'000'000,
                               .paused = true,
                               .reason = RelaunchReason::GpuLost,
                               .written_s = 1000};
  REQUIRE(write_relaunch_target(store, written));
  const auto read = take_relaunch_target(store, 1010);
  REQUIRE(read);
  CHECK(read->file == written.file);
  CHECK(read->start_us == 754'000'000);
  CHECK(read->paused);
  CHECK(read->reason == RelaunchReason::GpuLost);
  CHECK_FALSE(std::filesystem::exists(store));
  CHECK_FALSE(take_relaunch_target(store, 1010));
}

TEST_CASE("C18-02 : une relance à l'ouverture n'impose pas 0:00, la reprise s'applique", "[app][relaunch]") {
  const std::filesystem::path store = fresh_dir("ajp-relaunch-nostart") / "relaunch.txt";
  REQUIRE(write_relaunch_target(store, RelaunchTarget{.file = "/v/a.mkv", .written_s = 50}));
  const auto read = take_relaunch_target(store, 60);
  REQUIRE(read);
  CHECK_FALSE(read->start_us.has_value());
  CHECK_FALSE(read->paused);
}

TEST_CASE("relaunch.txt : périmé, daté du futur, incomplet ou à clé inconnue", "[app][relaunch]") {
  const std::filesystem::path dir = fresh_dir("ajp-relaunch-stale");
  const std::filesystem::path store = dir / "relaunch.txt";
  REQUIRE(write_relaunch_target(store, RelaunchTarget{.file = "/v/a.mkv", .start_us = 1, .written_s = 100}));
  CHECK_FALSE(take_relaunch_target(store, 100 + kRelaunchFreshS + 1));  // d'une autre séance
  CHECK_FALSE(std::filesystem::exists(store));                          // et retiré quand même
  REQUIRE(write_relaunch_target(store, RelaunchTarget{.file = "/v/a.mkv", .start_us = 1, .written_s = 500}));
  CHECK_FALSE(take_relaunch_target(store, 100));  // une horloge qui recule ne le rend pas valable
  std::ofstream(store) << "file=/v/a.mkv\nstart_us=12\n";
  CHECK_FALSE(take_relaunch_target(store, 100));  // sans date
  std::ofstream(store) << "file=/v/a.mkv\nstart_us=12\nfutur=oui\nwritten_s=99\n";
  const auto read = take_relaunch_target(store, 100);  // une clé d'une version suivante est ignorée
  REQUIRE(read);
  CHECK(read->start_us == 12);
  CHECK_FALSE(write_relaunch_target(store, RelaunchTarget{.file = "/v/a\nb.mkv", .written_s = 1}));
}

TEST_CASE("relaunch.txt aux fins de ligne Windows (\\r\\n) : lu comme l'autre", "[app][relaunch]") {
  const std::filesystem::path store = fresh_dir("ajp-relaunch-crlf") / "relaunch.txt";
  std::ofstream(store, std::ios::binary) << "file=/v/a.mkv\r\nstart_us=12\r\npaused=1\r\nreason=gpu-lost\r\n"
                                            "written_s=99\r\n";
  const auto read = take_relaunch_target(store, 100);
  REQUIRE(read);
  CHECK(read->file == std::filesystem::path("/v/a.mkv"));
  CHECK(read->start_us == 12);
  CHECK(read->paused);
  CHECK(read->reason == RelaunchReason::GpuLost);
}

TEST_CASE(
    "take_relaunch : seul le processus relancé prend la vidéo, un lancement ordinaire retire le fichier",
    "[app][relaunch]") {
  const std::filesystem::path dir = fresh_dir("ajp-relaunch-take");
  const PlayerPaths paths{.data = dir};
  REQUIRE(write_relaunch_target(
      relaunch_target_file(dir),
      RelaunchTarget{.file = "/v/b.mkv", .start_us = 5, .written_s = relaunch_now_s()}));
  PlayOptions plain;
  plain.file = "/v/mien.mkv";
  CHECK_FALSE(take_relaunch(plain, paths, relaunch_now_s()));
  CHECK(plain.file == "/v/mien.mkv");
  CHECK_FALSE(std::filesystem::exists(relaunch_target_file(dir)));
  REQUIRE(write_relaunch_target(
      relaunch_target_file(dir),
      RelaunchTarget{.file = "/v/b.mkv", .start_us = 5, .written_s = relaunch_now_s()}));
  PlayOptions relaunched;
  relaunched.relaunched = RelaunchedFor::GpuLost;
  const auto taken = take_relaunch(relaunched, paths, relaunch_now_s());
  REQUIRE(taken);
  CHECK(relaunched.file == "/v/b.mkv");
}

TEST_CASE("Relance : les arguments rejoués, sans le --relaunched= d'avant", "[app][relaunch]") {
  const std::array<std::string_view, 3> launched{"C:\\Mes vidéos\\a.mkv", "--relaunched=engine-updated",
                                                 "--fullscreen"};
  const std::vector<std::string> kept = relaunch_arguments(launched, RelaunchedFor::GpuLost);
  REQUIRE(kept.size() == 3);
  CHECK(kept[0] == "C:\\Mes vidéos\\a.mkv");
  CHECK(kept[1] == "--fullscreen");
  CHECK(kept[2] == "--relaunched=gpu-lost");
}

TEST_CASE("Relance : la citation de _spawnv suit CommandLineToArgvW", "[app][relaunch]") {
  CHECK(windows_quoted("simple") == "simple");
  CHECK(windows_quoted("") == "\"\"");
  CHECK(windows_quoted("C:\\Mes vidéos\\a.mkv") == "\"C:\\Mes vidéos\\a.mkv\"");
  // Antislashs doublés devant le guillemet fermant, pas ailleurs.
  CHECK(windows_quoted("C:\\Mes dossiers\\") == "\"C:\\Mes dossiers\\\\\"");
  // Un guillemet dans l'argument : précédé de ses antislashs doublés et d'un antislash.
  CHECK(windows_quoted("a\\\"b c") == "\"a\\\\\\\"b c\"");
  CHECK(windows_quoted("tab\there") == "\"tab\there\"");
}

}  // namespace
}  // namespace ajp::app::play
