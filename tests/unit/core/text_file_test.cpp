// Fichiers texte entiers (text_file.h) : lecture, fichier facultatif, et remplacement atomique.
#include "core/text_file.h"

#include <filesystem>
#include <iterator>
#include <optional>
#include <string>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using namespace ajp::core;
namespace fs = std::filesystem;

namespace {

fs::path scratch(const std::string& name) {
  const fs::path dir = fs::temp_directory_path() / ("ajp-text-file-" + name);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

}  // namespace

TEST_CASE("Fichiers texte : lecture entière et remplacement atomique", "[text_file]") {
  const fs::path dir = scratch("text-files");
  const fs::path path = dir / "neuf" / "fichier.toml";
  REQUIRE(write_text_atomically(path, "un"));
  REQUIRE(write_text_atomically(path, "deux\n"));
  auto text = read_text_file(path);
  REQUIRE(text);
  CHECK(*text == "deux\n");
  CHECK(std::distance(fs::directory_iterator(path.parent_path()), fs::directory_iterator{}) == 1);

  auto missing = read_text_file(dir / "absent");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::FileUnreadable);
  auto folder = read_text_file(dir);
  REQUIRE_FALSE(folder);
  CHECK(folder.error().code == RefusalCode::FileUnreadable);

  // Facultatif : absent n'est pas une erreur, présent se lit, un dossier reste refusé.
  auto absent = read_text_file_if_present(dir / "absent");
  REQUIRE(absent);
  CHECK_FALSE(*absent);
  auto present = read_text_file_if_present(path);
  REQUIRE(present);
  CHECK(*present == std::optional<std::string>{"deux\n"});
  CHECK_FALSE(read_text_file_if_present(dir));
  auto under_file = read_text_file_if_present(path / "sous-fichier");
  REQUIRE(under_file);
  CHECK_FALSE(*under_file);
  // Un dossier à la place du fichier : le renommage échoue, rien n'est laissé derrière.
  fs::create_directories(dir / "dossier");
  auto blocked = write_text_atomically(dir / "dossier", "x");
  REQUIRE_FALSE(blocked);
  CHECK(blocked.error().code == RefusalCode::FileUnwritable);
  CHECK(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 2);
}
