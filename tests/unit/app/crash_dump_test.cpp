// Rapport de plantage (P2.2) : sous Windows, un minidump lisible par WinDbg ; ailleurs, aucun.
#include "app/play/crash_dump.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#ifndef _WIN32
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Minidump : écrit sous Windows, absent ailleurs", "[app][crash]") {
  const std::filesystem::path file = std::filesystem::temp_directory_path() / "ajp-crash-dump-test.dmp";
  std::error_code error;
  std::filesystem::remove(file, error);
#ifdef _WIN32
  REQUIRE(ajp::app::play::write_crash_dump(file));
  CHECK(std::filesystem::file_size(file) > 0);
  std::filesystem::remove(file, error);
#else
  CHECK_FALSE(ajp::app::play::write_crash_dump(file));
  CHECK_FALSE(std::filesystem::exists(file));
#endif
}

#ifndef _WIN32
// Un signal fatal laisse sa ligne dans ajp.log (A39-04) : le processus de l'essai se termine par ce signal,
// le système garde la main (core), et le journal d'à côté dit « plantage : signal 8 ».
TEST_CASE("Plantage sous POSIX : une ligne dans ajp.log, puis la fin normale du signal", "[app][crash]") {
  const std::filesystem::path root = std::filesystem::temp_directory_path() / "ajp-crash-signal-test";
  std::error_code error;
  std::filesystem::remove_all(root, error);
  std::filesystem::create_directories(root / "crashes", error);
  {
    std::ofstream(root / "ajp.log") << "avant\n";
  }

  const pid_t child = fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    const rlimit no_core{0, 0};
    setrlimit(RLIMIT_CORE, &no_core);
    ajp::app::play::install_crash_dumps(root / "crashes");
    std::raise(SIGFPE);
    _exit(0);  // jamais atteint : le signal tue le processus
  }
  int status = 0;
  REQUIRE(waitpid(child, &status, 0) == child);
  CHECK(WIFSIGNALED(status));
  CHECK(WTERMSIG(status) == SIGFPE);
  std::ifstream log(root / "ajp.log");
  const std::string text((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
  CHECK(text.starts_with("avant\n"));
  CHECK(text.find("plantage : signal 8 (SIGFPE)") != std::string::npos);
  std::filesystem::remove_all(root, error);
}
#endif
