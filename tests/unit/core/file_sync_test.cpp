// L'écriture atomique (write_atomic) : contenu entier, temporaire retiré sur un refus, et sous Windows une
// destination tenue ouverte par un autre processus le temps d'un scan (A20-01), liens symboliques et
// temporaires abandonnés (A20-08).
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

#ifdef __linux__
TEST_CASE("Écriture atomique : un fichier qu'on ne sait pas vider n'est pas une erreur (B20-01)", "[core]") {
  // fsync sur un fichier spécial rend EINVAL, comme sur certains systèmes de fichiers FUSE.
  CHECK_FALSE(ajp::core::sync_file("/dev/null"));
  CHECK(ajp::core::sync_file("/chemin/absent") == std::errc::no_such_file_or_directory);
}
#endif

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

TEST_CASE("Écriture atomique : un lien symbolique reste un lien, le fichier nommé change (A20-08)",
          "[core]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-write-symlink";
  fs::remove_all(dir);
  fs::create_directories(dir / "dotfiles");
  const fs::path real = dir / "dotfiles" / "play.toml";
  REQUIRE(ajp::core::write_text_atomically(real, "a = 1\n"));
  std::error_code linked;
  fs::create_symlink(fs::path("dotfiles") / "play.toml", dir / "play.toml", linked);
  if (linked) SKIP("liens symboliques refusés ici : " << linked.message());
  REQUIRE(ajp::core::write_text_atomically(dir / "play.toml", "a = 2\n"));
  CHECK(fs::is_symlink(dir / "play.toml"));
  CHECK(read_all(real) == "a = 2\n");
  CHECK(entries(dir) == 2);
  fs::remove_all(dir);
}

TEST_CASE("Écriture atomique : les temporaires abandonnés depuis plus d'une heure sont retirés (A20-08)",
          "[core]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-write-orphans";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path old_temp = dir / ".play.toml.1-2.tmp";
  const fs::path fresh_temp = dir / ".play.toml.3-4.tmp";
  const fs::path other = dir / ".autre.toml.5-6.tmp";
  for (const fs::path& file : {old_temp, fresh_temp, other}) std::ofstream(file) << "x";
  const auto two_hours_ago = fs::file_time_type::clock::now() - std::chrono::hours(2);
  fs::last_write_time(old_temp, two_hours_ago);
  fs::last_write_time(other, two_hours_ago);
  REQUIRE(ajp::core::write_text_atomically(dir / "play.toml", "a = 1\n"));
  CHECK_FALSE(fs::exists(old_temp));
  CHECK(fs::exists(fresh_temp));  // un autre écrivain peut l'écrire en ce moment
  CHECK(fs::exists(other));       // les temporaires d'un autre fichier ne sont pas les siens
  fs::remove_all(dir);
}
