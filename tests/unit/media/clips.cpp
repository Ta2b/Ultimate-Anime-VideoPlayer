#include "clips.h"

#include <catch2/catch_test_macros.hpp>

namespace ajp::media::test {

ClipSpec full_clip() {
  return ClipSpec{
      .name = "full.mkv",
      .audio = {{.language = "jpn", .title = "Japonais"},
                {.language = "eng", .title = "Anglais", .is_default = true}},
      .subtitles = {{.language = "fre",
                     .title = "Complets",
                     .is_default = true,
                     .events = {{.start_ms = 100, .duration_ms = 500, .text = "Bonjour"},
                                {.start_ms = 700, .duration_ms = 400, .text = "Au revoir"}}},
                    {.language = "fre",
                     .title = "Panneaux",
                     .forced = true,
                     .events = {{.start_ms = 200, .duration_ms = 300, .text = "Panneau"}}}},
      .attachments = {{.filename = "Police.ttf",
                       .mime_type = "application/x-truetype-font",
                       .data = kFontBytes}},
      .chapters = {{.start_ms = 0, .end_ms = 400, .title = "Ouverture"},
                   {.start_ms = 400, .end_ms = 800, .title = "Partie A"},
                   {.start_ms = 800, .end_ms = 1200, .title = "Fin"}},
  };
}

ClipSpec seek_clip() {
  return ClipSpec{
      .name = "seek.mkv",
      .codec = VideoCodec::Mpeg4,
      .gop = 10,
      .audio = {{.language = "jpn", .is_default = true}},
      .subtitles = {{.language = "fre",
                     .is_default = true,
                     .events = {{.start_ms = 0, .duration_ms = 300, .text = "Un"},
                                {.start_ms = 500, .duration_ms = 300, .text = "Deux"},
                                {.start_ms = 900, .duration_ms = 200, .text = "Trois"}}}},
  };
}

ClipSpec hdr_clip() {
  return ClipSpec{
      .name = "hdr.mkv",
      .pixel_format = AV_PIX_FMT_YUV420P10LE,
      .frames = 5,
      .color = {.primaries = AVCOL_PRI_BT2020,
                .transfer = AVCOL_TRC_SMPTE2084,
                .matrix = AVCOL_SPC_BT2020_NCL,
                .range = AVCOL_RANGE_MPEG,
                .siting = AVCHROMA_LOC_TOPLEFT,
                .hdr10 = true},
  };
}

ClipSpec bframes_clip() {
  return ClipSpec{
      .name = "bframes.mp4",
      .container = "mp4",
      .codec = VideoCodec::Mpeg4,
      .frames = 24,
      .frame_rate = {24000, 1001},
      .gop = 12,
      .b_frames = 2,
  };
}

ClipSpec late_video_clip() {
  return ClipSpec{
      .name = "late_video.mkv",
      .codec = VideoCodec::Mpeg4,
      .gop = 10,
      .video_delay = 5,
      .audio = {{.language = "jpn", .is_default = true}},
  };
}

ClipSpec gpu_clip(const std::string& encoder) {
  return ClipSpec{
      .name = encoder + ".mkv",
      .width = 256,
      .height = 144,
      .frames = 36,
      .gop = 12,
      .encoder = encoder,
  };
}

std::filesystem::path require_gpu_clip(const ClipSpec& spec) {
  if (const auto why = encoder_unavailable(spec)) SKIP(*why);
  return require_clip(spec);
}

std::filesystem::path require_clip(const ClipSpec& spec) {
  auto path = write_clip(spec);
  INFO(spec.name << " : " << (path ? std::string("écrit") : path.error()));
  REQUIRE(path.has_value());
  return *path;
}

}  // namespace ajp::media::test
