#include "app/play/relaunch_process.h"

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

using ajp::app::play::start_relaunched;
using ajp::app::play::windows_quoted;

TEST_CASE("Relance : les arguments cités pour Windows comme CommandLineToArgvW les relit",
          "[app][relaunch]") {
  CHECK(windows_quoted("a.mkv") == "a.mkv");
  CHECK(windows_quoted("C:\\Mes vidéos\\a.mkv") == "\"C:\\Mes vidéos\\a.mkv\"");
  CHECK(windows_quoted("") == "\"\"");
  CHECK(windows_quoted("dit \"oui\"") == "\"dit \\\"oui\\\"\"");
  // Les barres obliques inverses avant le guillemet de fin se doublent.
  CHECK(windows_quoted("C:\\a b\\") == "\"C:\\a b\\\\\"");
}

#ifndef _WIN32
TEST_CASE("Relance après un GPU bloqué : un nouveau processus, la main revient (C34-04)", "[app][relaunch]") {
  // Sans remplacer ce processus : posix_spawn, pas execv, qui attendrait la fin de tous les threads.
  // Un nom par processus : deux lancements du test en même temps ne se mêlent pas.
  const std::filesystem::path marker =
      std::filesystem::temp_directory_path() / ("ajp-relaunch-test-" + std::to_string(::getpid()));
  std::filesystem::remove(marker);
  const std::string started =
      start_relaunched("/bin/sh", {"-c", "echo ok > '" + marker.string() + "'"}, false);
  CHECK(started.empty());
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  while (!std::filesystem::exists(marker) && std::chrono::steady_clock::now() < until)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CHECK(std::filesystem::exists(marker));
  std::filesystem::remove(marker);
  // Un exécutable absent : la cause, pas un succès.
  CHECK_FALSE(start_relaunched("/nexiste/pas/ajp", {}, false).empty());
}
#endif
