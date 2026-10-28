#include "app/play/external_subtitles.h"

#include <fstream>

#include <catch2/catch_test_macros.hpp>

using ajp::app::play::find_external_subtitles;

namespace {

void touch(const std::filesystem::path& path) {
  std::ofstream(path) << "x";
}

}  // namespace

TEST_CASE("Sous-titres externes : le nom de la vidéo, suivi de rien ou d'un point", "[subtitles]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-external-subs";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  touch(dir / "Episode 01.mkv");
  touch(dir / "Episode 01.srt");
  touch(dir / "Episode 01.fr.srt");
  touch(dir / "Episode 01.ASS");
  touch(dir / "Episode 010.srt");  // un autre épisode
  touch(dir / "Autre.srt");
  touch(dir / "Episode 01.txt");

  const auto found = find_external_subtitles(dir / "Episode 01.mkv");
  REQUIRE(found.size() == 3);
  CHECK(found[0].path.filename() == "Episode 01.ASS");
  CHECK(found[1].path.filename() == "Episode 01.fr.srt");
  CHECK(found[1].label.find("Français") != std::string::npos);
  CHECK(found[2].path.filename() == "Episode 01.srt");

  CHECK(find_external_subtitles(dir / "Absent.mkv").empty());
  CHECK(find_external_subtitles(dir / "nodir" / "x.mkv").empty());
  std::filesystem::remove_all(dir);
}
