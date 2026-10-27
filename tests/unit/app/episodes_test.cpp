// L'épisode voisin : ordre naturel des noms, extensions vidéo seules, bouts de liste.
#include "app/play/episodes.h"

#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

TEST_CASE("Ordre naturel : les nombres comptent pour leur valeur, la casse non", "[app][episodes]") {
  CHECK(natural_compare("ep2.mkv", "ep10.mkv") < 0);
  CHECK(natural_compare("ep10.mkv", "ep2.mkv") > 0);
  CHECK(natural_compare("Ep02.mkv", "ep2.mkv") == 0);
  CHECK(natural_compare("a.mkv", "B.mkv") < 0);
  CHECK(natural_compare("ep1", "ep1b") < 0);
}

TEST_CASE("Épisode voisin : suivant, précédent, et rien au bout", "[app][episodes]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-episodes-test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  for (const char* name : {"Show - 2.mkv", "Show - 10.mkv", "Show - 1.MKV", "Show - 1.srt", "notes.txt"})
    std::ofstream(dir / name) << "x";

  CHECK(sibling_episode(dir / "Show - 1.MKV", 1) == dir / "Show - 2.mkv");
  CHECK(sibling_episode(dir / "Show - 2.mkv", 1) == dir / "Show - 10.mkv");
  CHECK(sibling_episode(dir / "Show - 10.mkv", -1) == dir / "Show - 2.mkv");
  CHECK_FALSE(sibling_episode(dir / "Show - 10.mkv", 1));
  CHECK_FALSE(sibling_episode(dir / "Show - 1.MKV", -1));
  CHECK_FALSE(sibling_episode(dir / "absent.mkv", 1));
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace ajp::app::play
