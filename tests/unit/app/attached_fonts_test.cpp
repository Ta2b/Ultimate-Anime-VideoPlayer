// Les polices jointes d'un fichier, sous un plafond total (A17-09).
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/mem.h>
}

#include "app/play/bridges.h"
#include "media/track.h"

using ajp::app::play::attached_fonts;
using ajp::media::CodecParameters;
using ajp::media::TrackInfo;
using ajp::media::TrackKind;

TEST_CASE("Polices jointes : dans l'ordre du fichier jusqu'au plafond total (A17-09)", "[subtitles]") {
  AVCodecParameters* raw = avcodec_parameters_alloc();
  REQUIRE(raw != nullptr);
  raw->extradata = static_cast<std::uint8_t*>(av_mallocz(4 + AV_INPUT_BUFFER_PADDING_SIZE));
  raw->extradata_size = 4;
  auto data = CodecParameters::copy_of(*raw);
  avcodec_parameters_free(&raw);
  REQUIRE(data.has_value());
  std::vector<TrackInfo> tracks;
  for (const char* name : {"a.ttf", "b.ttf", "c.ttf"}) {
    tracks.push_back(TrackInfo{
        .kind = TrackKind::Attachment, .filename = name, .mime_type = "font/ttf", .parameters = *data});
  }
  tracks.push_back(TrackInfo{.kind = TrackKind::Subtitle, .filename = "d.ttf", .parameters = *data});

  const auto all = attached_fonts(tracks);
  CHECK(all.fonts.size() == 3);
  CHECK(all.dropped == 0);

  const auto capped = attached_fonts(tracks, 10);  // deux polices de 4 octets tiennent, pas trois
  REQUIRE(capped.fonts.size() == 2);
  CHECK(capped.fonts[0].name == "a.ttf");
  CHECK(capped.fonts[1].name == "b.ttf");
  CHECK(capped.dropped == 1);
}
