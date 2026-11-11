// La veille et les gels vus de la boucle (A29-02, A33-03), et l'économiseur d'écran (A33-02).
#include "app/play/sleep_watch.h"

#include <filesystem>
#include <system_error>

#include <catch2/catch_test_macros.hpp>

#include "app/play/relaunch_target.h"

namespace ajp::app::play {
namespace {

TEST_CASE("Veille : un trou sur l'une des deux horloges est vu, un tour ordinaire non", "[app][sleep]") {
  SleepWatch watch;
  CHECK_FALSE(watch.turn(1'000'000, 50'000'000));  // premier tour
  CHECK_FALSE(watch.turn(1'016'000, 50'016'000));  // une image
  CHECK_FALSE(watch.turn(3'000'000, 52'000'000));  // une fenêtre déplacée, 2 s
  // Linux : steady_clock arrêtée pendant la veille, CLOCK_BOOTTIME a compté 20 minutes.
  const auto linux_sleep = watch.turn(3'020'000, 1'252'020'000);
  REQUIRE(linux_sleep);
  CHECK(*linux_sleep >= 1'199'000'000);
  // Windows : QPC continue, les deux horloges sautent.
  const auto windows_sleep = watch.turn(1'203'020'000, 2'452'020'000);
  REQUIRE(windows_sleep);
  CHECK(*windows_sleep == 1'200'000'000);
  CHECK_FALSE(watch.turn(1'203'036'000, 2'452'036'000));
}

TEST_CASE("Veille ou gel : la barre de titre tenue sous Windows n'est pas une veille (B31-13)",
          "[app][sleep]") {
  // Windows : les deux horloges comptent la veille ; seule la fenêtre tenue (boucle modale) dit le gel.
  CHECK(gap_kind(6'000'000, 6'000'000, false, false) == GapKind::Sleep);
  CHECK(gap_kind(6'000'000, 6'000'000, true, false) == GapKind::Freeze);
  // Linux : steady s'arrête pendant la veille. Les deux horloges d'accord : un gel (SIGSTOP, débogueur).
  CHECK(gap_kind(3'000, 1'200'000'000, false, true) == GapKind::Sleep);
  CHECK(gap_kind(6'000'000, 6'000'000, false, true) == GapKind::Freeze);

  SleepWatch watch;
  CHECK_FALSE(watch.turn(1'000'000, 50'000'000));
  const auto held = watch.turn(7'500'000, 56'500'000);  // 6,5 s sans tour de boucle
  REQUIRE(held);
  // Un déplacement de la fenêtre vu 2 s après le début du trou : la boucle modale de Windows.
  CHECK(watch.last_kind(52'000'000) == GapKind::Freeze);
  // Rien pendant le trou, ou juste au tour d'avant (un événement ordinaire de la boucle) : pas de fenêtre
  // tenue.
  CHECK(watch.last_kind(0) == watch.last_kind(50'010'000));
}

TEST_CASE("Économiseur : coupé seulement quand une vidéo avance", "[app][sleep]") {
  CHECK(must_stay_awake(true, false, false));
  CHECK_FALSE(must_stay_awake(true, true, false));    // en pause
  CHECK_FALSE(must_stay_awake(true, false, true));    // fin de la vidéo
  CHECK_FALSE(must_stay_awake(false, false, false));  // accueil, assistant
}

TEST_CASE("Relance : la vidéo et l'instant relus une fois, pas un fichier d'une autre séance",
          "[app][relaunch]") {
  std::error_code error;
  const std::filesystem::path dir = std::filesystem::temp_directory_path(error) / "ajp-relaunch-test";
  std::filesystem::remove_all(dir, error);
  std::filesystem::create_directories(dir, error);
  const std::filesystem::path store = relaunch_target_file(dir);
  const RelaunchTarget target{.file = dir / "épisode 01.mkv",
                              .start_us = 754'000'000,
                              .paused = false,
                              .reason = RelaunchReason::GpuLost,
                              .written_s = 1'000};
  REQUIRE(write_relaunch_target(store, target));
  const auto read = take_relaunch_target(store, 1'010);
  REQUIRE(read);
  CHECK(read->file == target.file);
  CHECK(read->start_us == 754'000'000);
  CHECK(read->reason == RelaunchReason::GpuLost);
  CHECK_FALSE(take_relaunch_target(store, 1'011));  // effacé après lecture
  REQUIRE(write_relaunch_target(store, target));
  CHECK_FALSE(take_relaunch_target(store, 1'000 + kRelaunchFreshS + 1));  // trop vieux : ignoré
  CHECK_FALSE(std::filesystem::exists(store, error));
  std::filesystem::remove_all(dir, error);
}

}  // namespace
}  // namespace ajp::app::play
