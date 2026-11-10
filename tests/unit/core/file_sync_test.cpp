// L'écriture atomique (write_atomic) : contenu entier, temporaire retiré sur un refus, et sous Windows une
// destination tenue ouverte par un autre processus le temps d'un scan (A20-01).
#include "core/file_sync.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

std::string read_all(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

int entries(const fs::path& dir) {
  int count = 0;
  for (const auto& entry : fs::directory_iterator(dir)) {
    (void)entry;
    ++count;
  }
  return count;
}

}  // namespace

TEST_CASE("Écriture atomique : le nouveau contenu remplace l'ancien, sans temporaire laissé", "[core]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-write-atomic";
  fs::remove_all(dir);
  const fs::path file = dir / "sous" / "play.toml";
  REQUIRE(ajp::core::write_text_atomically(file, "a = 1\n"));
  REQUIRE(ajp::core::write_text_atomically(file, "a = 2\n"));
  CHECK(read_all(file) == "a = 2\n");
  CHECK(entries(file.parent_path()) == 1);
  fs::remove_all(dir);
}

TEST_CASE("Écriture atomique : un refus est rendu et le temporaire retiré", "[core]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-write-refused";
  fs::remove_all(dir);
  // La destination est un dossier non vide : le renommage est refusé, sur tous les systèmes.
  fs::create_directories(dir / "play.toml" / "dedans");
  const auto written = ajp::core::write_text_atomically(dir / "play.toml", "x");
  REQUIRE_FALSE(written);
  CHECK(written.error().code == ajp::RefusalCode::FileUnwritable);
  CHECK(entries(dir) == 1);
  fs::remove_all(dir);
}

#ifdef _WIN32
TEST_CASE(
    "Écriture atomique : une destination tenue sans partage de suppression est remplacée après sa "
    "fermeture (A20-01)",
    "[core]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-write-held";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path file = dir / "update.toml";
  REQUIRE(ajp::core::write_text_atomically(file, "avant"));
  // Comme un antivirus : lecture partagée en lecture seule, ni écriture ni suppression.
  HANDLE held = CreateFileW(file.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  REQUIRE(held != INVALID_HANDLE_VALUE);
  std::thread release([held] {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    CloseHandle(held);
  });
  const auto written = ajp::core::write_text_atomically(file, "après");
  release.join();
  REQUIRE(written);
  CHECK(read_all(file) == "après");
  fs::remove_all(dir);
}
#endif
