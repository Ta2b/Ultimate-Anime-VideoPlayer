#include "app/play/external_subtitles.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

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
  CHECK(found[1].label.find("French") != std::string::npos);
  CHECK(found[2].path.filename() == "Episode 01.srt");

  CHECK(find_external_subtitles(dir / "Absent.mkv").empty());
  CHECK(find_external_subtitles(dir / "nodir" / "x.mkv").empty());
  std::filesystem::remove_all(dir);
}

TEST_CASE("Sous-titres externes : dossiers Subs/ et noms proches (P3.80)", "[subtitles]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-external-subs-near";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir / "Subs" / "[G] Show - 01 [1080p]");
  touch(dir / "[G] Show - 01 [1080p].mkv");
  touch(dir / "Show_-_01.fr.ass");
  touch(dir / "Subs" / "[G] Show - 01 [1080p]" / "2_English.srt");
  touch(dir / "Subs" / "Show - 02.srt");
  const auto found = find_external_subtitles(dir / "[G] Show - 01 [1080p].mkv");
  REQUIRE(found.size() == 2);
  CHECK(found[0].path.filename() == "2_English.srt");
  CHECK(found[0].language == "eng");
  CHECK(found[1].language == "fr");
  std::filesystem::remove_all(dir);
}

TEST_CASE("Sous-titres externes : la langue voulue passe avant la coupe à 16 (A17-11)", "[subtitles]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-external-subs-many";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir / "Subs" / "Episode 01");
  touch(dir / "Episode 01.mkv");
  // 30 fichiers : « 10_ » à « 19_ » se rangent avant « 2_ », le français (« 28_French ») après la coupe.
  for (int i = 1; i <= 30; ++i) {
    const std::string language = i == 28 ? "French" : i == 2 ? "English" : "Thai";
    touch(dir / "Subs" / "Episode 01" / (std::to_string(i) + "_" + language + ".srt"));
  }
  const auto unordered = find_external_subtitles(dir / "Episode 01.mkv");
  REQUIRE(unordered.size() == ajp::app::play::kMaxExternalSubtitles);
  CHECK(std::ranges::none_of(unordered, [](const auto& s) { return s.language == "fr"; }));

  const std::vector<std::string> wanted{"fre", "fr", "eng", "en"};
  const auto found = find_external_subtitles(dir / "Episode 01.mkv", wanted);
  REQUIRE(found.size() == ajp::app::play::kMaxExternalSubtitles);
  CHECK(found[0].path.filename() == "28_French.srt");
  CHECK(found[1].path.filename() == "2_English.srt");
  std::filesystem::remove_all(dir);
}
