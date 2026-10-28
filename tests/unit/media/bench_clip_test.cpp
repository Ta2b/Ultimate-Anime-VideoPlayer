#include "media/bench_clip.h"

#include <filesystem>

#include <catch2/catch_test_macros.hpp>

#include "media/demuxer.h"

using namespace ajp::media;

TEST_CASE("la séquence de l'évaluation s'écrit et se rouvre", "[media][bench_clip]") {
  const std::filesystem::path file = std::filesystem::temp_directory_path() / "ajp-bench-clip-test.mkv";
  std::filesystem::remove(file);
  const auto written =
      write_bench_clip(file, BenchClipSpec{.width = 320, .height = 180, .fps = 24, .seconds = 2});
  INFO((written ? std::string() : written.error().detail));
  REQUIRE(written.has_value());
  CHECK(std::filesystem::file_size(file) > 10'000);
  auto demuxer = Demuxer::open(file);
  REQUIRE(demuxer.has_value());
  bool video = false;
  for (const TrackInfo& track : demuxer->tracks()) video = video || track.kind == TrackKind::Video;
  CHECK(video);
  std::filesystem::remove(file);
}
