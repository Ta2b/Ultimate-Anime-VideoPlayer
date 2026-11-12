// Les briques du socle que seuls les tests GPU exerçaient (C20-10) : DynamicLibrary, text.h, les droits
// gardés par l'écriture atomique (C20-08) et la place rendue par une boîte dont le message lève (C20-07).
#include <filesystem>
#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "core/dynamic_library.h"
#include "core/file_sync.h"
#include "core/inbox.h"
#include "core/text.h"

namespace fs = std::filesystem;

TEST_CASE("DynamicLibrary : chemin relatif refusé, fichier absent refusé, symbole absent nommé", "[core]") {
  const auto relative = ajp::DynamicLibrary::open("libabsente.so");
  REQUIRE_FALSE(relative);
  CHECK(relative.error().code == ajp::RefusalCode::LibraryUnavailable);
  CHECK(relative.error().detail.find("chemin relatif") != std::string::npos);

  const auto absent = ajp::DynamicLibrary::open(fs::temp_directory_path() / "ajp-absente" / "libx.so");
  REQUIRE_FALSE(absent);
  CHECK(absent.error().code == ajp::RefusalCode::LibraryUnavailable);
#ifndef _WIN32
  auto libc = ajp::DynamicLibrary::open_system("libc.so.6");
  REQUIRE(libc);
  CHECK(libc->symbol<int(const char*)>("puts"));
  const auto missing = libc->symbol<void()>("ajp_symbole_absent");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().detail.find("ajp_symbole_absent") != std::string::npos);
  CHECK(missing.error().detail.find("libc.so.6") != std::string::npos);
  // Le déplacement garde la bibliothèque ouverte.
  ajp::DynamicLibrary moved = std::move(*libc);
  CHECK(moved.symbol<int(const char*)>("puts"));
#endif
}

TEST_CASE("text.h : trim et iequal", "[core]") {
  STATIC_REQUIRE(ajp::core::trim("  a b \t\n") == "a b");
  STATIC_REQUIRE(ajp::core::trim(" \t ").empty());
  STATIC_REQUIRE(ajp::core::iequal("Vulkan", "vULKAN"));
  STATIC_REQUIRE_FALSE(ajp::core::iequal("Vulkan", "Vulkan2"));
  CHECK(ajp::core::ascii_lower("ÉtÉ ABC") == "ÉtÉ abc");
}

#ifndef _WIN32
TEST_CASE("Écriture atomique : les droits de l'ancien fichier restent (C20-08)", "[core]") {
  const fs::path dir = fs::temp_directory_path() / "ajp-write-perms";
  fs::remove_all(dir);
  const fs::path file = dir / "play.toml";
  REQUIRE(ajp::core::write_text_atomically(file, "a = 1\n"));
  fs::permissions(file, fs::perms::owner_read | fs::perms::owner_write);
  REQUIRE(ajp::core::write_text_atomically(file, "a = 2\n"));
  CHECK(fs::status(file).permissions() == (fs::perms::owner_read | fs::perms::owner_write));
  fs::remove_all(dir);
}
#endif

namespace {

struct Fragile {
  bool fail = false;
  Fragile() = default;
  Fragile(const Fragile& other) : fail(other.fail) {
    if (other.fail) throw std::runtime_error("copie refusée");
  }
  Fragile& operator=(const Fragile&) = default;
};

}  // namespace

TEST_CASE("Inbox : un message qui lève ne garde pas sa place (C20-07)", "[core]") {
  ajp::Inbox<Fragile> inbox({1});
  Fragile bad;
  bad.fail = true;
  CHECK_THROWS(inbox.try_push(bad));
  CHECK(inbox.pending<Fragile>() == 0);
  const Fragile good;
  CHECK(inbox.try_push(good));
}
