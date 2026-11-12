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

TEST_CASE("Plantage : l'étape en cours est retenue, bornée (B39-10)", "[app][crash]") {
  ajp::app::play::note_crash_stage("ouverture de ep.mkv");
  CHECK(ajp::app::play::crash_stage() == "ouverture de ep.mkv");
  ajp::app::play::note_crash_stage(std::string(500, 'x'));
  CHECK(ajp::app::play::crash_stage() == std::string(200, 'x'));
  ajp::app::play::note_crash_stage("");
  CHECK(ajp::app::play::crash_stage().empty());
}

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
    ajp::app::play::note_crash_stage("construction du moteur SPAN (tensorrt)");
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
  // B39-10 : l'étape en cours, que la file du journal aurait perdue.
  CHECK(text.find("dernière étape : construction du moteur SPAN (tensorrt)") != std::string::npos);
#ifdef __linux__
  // B39-05 : la pile nomme les modules sans leur dossier.
  const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", error);
  CHECK(text.find(self.parent_path().string() + "/") == std::string::npos);
  CHECK(text.find(self.filename().string() + "(+0x") != std::string::npos);
#endif
  std::filesystem::remove_all(root, error);
}
#endif
