// Rapport de plantage (P2.2) : sous Windows, un minidump lisible par WinDbg ; ailleurs, aucun.
#include "app/play/crash_dump.h"

#include <filesystem>

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
