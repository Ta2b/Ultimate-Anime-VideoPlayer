// Le disque plein dit tel quel (A31-11) : la place libre lue sous un dossier, ou son premier parent qui
// existe.
#include <cstdint>
#include <filesystem>
#include <limits>
#include <system_error>

#include <catch2/catch_test_macros.hpp>

#include "core/file_sync.h"

namespace ajp::core {
namespace {

TEST_CASE("Place libre : assez, rien ; pas assez, InsufficientSpace qui dit combien", "[core][files]") {
  std::error_code error;
  const std::filesystem::path dir = std::filesystem::temp_directory_path(error);
  CHECK_FALSE(lacking_space(dir, 0));
  const std::uintmax_t huge = std::numeric_limits<std::uintmax_t>::max() / 2;
  const auto full = lacking_space(dir, huge);
  REQUIRE(full);
  CHECK(full->code == RefusalCode::InsufficientSpace);
  CHECK(full->detail.find("disque plein") != std::string::npos);
  // Un dossier qui n'existe pas encore (le cache des moteurs avant sa première construction) : son parent.
  const auto missing = lacking_space(dir / "ajp-pas-encore" / "engines", huge);
  REQUIRE(missing);
  CHECK(missing->code == RefusalCode::InsufficientSpace);
}

TEST_CASE("Écriture atomique : un fichier écrit en entier sur un disque qui a la place", "[core][files]") {
  std::error_code error;
  const std::filesystem::path file = std::filesystem::temp_directory_path(error) / "ajp-file-sync-test.txt";
  REQUIRE(write_text_atomically(file, "contenu"));
  CHECK(std::filesystem::file_size(file, error) == 7);
  std::filesystem::remove(file, error);
}

}  // namespace
}  // namespace ajp::core
