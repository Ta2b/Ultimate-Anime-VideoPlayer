// Le conteneur d'un export d'après l'extension, et l'échelle de temps vidéo d'un MP4 ou d'un MOV.
#include "export/container.h"

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

namespace ajp::exporter {

TEST_CASE("Conteneur : choisi par l'extension, quelle que soit la casse", "[export]") {
  CHECK(container_of("a/b.mkv") == Container::Matroska);
  CHECK(container_of("a/b.MP4") == Container::Mp4);
  CHECK(container_of("b.Mov") == Container::Mov);
  CHECK_FALSE(container_of("b.avi"));
  CHECK_FALSE(container_of("b"));
  CHECK(format_name(Container::Matroska) == "matroska");
  CHECK(label(Container::Mp4) == "MP4");
}

TEST_CASE("Conteneur : FFV1 et les polices jointes", "[export]") {
  CHECK(carries(Container::Matroska, Codec::Lossless));
  CHECK(carries(Container::Mp4, Codec::Lossless));
  CHECK_FALSE(carries(Container::Mov, Codec::Lossless));
  CHECK(carries(Container::Mov, Codec::Hevc));
  CHECK(carries_attachments(Container::Matroska));
  CHECK_FALSE(carries_attachments(Container::Mp4));
  CHECK_FALSE(carries_attachments(Container::Mov));
}

TEST_CASE("Conteneur : chaque période tombe sur un nombre entier de pas", "[export]") {
  CHECK(video_timescale(120'000, 1'001) == 120'000);  // 1001 pas par image
  CHECK(video_timescale(240, 1) == 15'360);           // 64 pas par image
  CHECK(video_timescale(48, 2) == 12'288);            // 24 i/s réduit, 512 pas
  CHECK(video_timescale(300'000, 1'001) == 300'000);
  CHECK(video_timescale(600'000, 1'001) == 0);  // trop fin : la microseconde
  CHECK(video_timescale(0, 1) == 0);
  // L'instant arrondi à la microseconde retombe sur son pas.
  const int timescale = video_timescale(120'000, 1'001);
  int off_grid = 0;
  for (std::int64_t n = 0; n < 100'000; ++n) {
    const std::int64_t us = (n * 1'001 * 1'000'000 + 60'000) / 120'000;  // arrondi
    const std::int64_t ticks = (us * timescale + 500'000) / 1'000'000;
    if (ticks != n * 1'001) ++off_grid;
  }
  CHECK(off_grid == 0);
}

}  // namespace ajp::exporter
