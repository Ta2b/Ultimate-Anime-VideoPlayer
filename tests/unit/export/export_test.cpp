// Export FFV1 Matroska de bout en bout : sans perte, métadonnées HDR10, pistes recopiées.
#include <cstdlib>
#include <cstring>

#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavutil/mastering_display_metadata.h>
}

#include "export/export.h"
#include "support.h"

namespace ajp::exporter {
namespace {

using namespace test;

constexpr FrameRate kRate{24000, 1001};

const AVPacketSideData* side_data(const AVCodecParameters& par, AVPacketSideDataType type) {
  return av_packet_side_data_get(par.coded_side_data, par.nb_coded_side_data, type);
}

std::string tag(const AVDictionary* dict, const char* key) {
  const AVDictionaryEntry* entry = av_dict_get(dict, key, nullptr, 0);
  return entry == nullptr ? std::string{} : std::string(entry->value);
}

}  // namespace

TEST_CASE("export FFV1 RGB 16 bits relu sans perte", "[export][ffv1]") {
  const auto dir = work_dir("ffv1_lossless");
  const auto frames = pattern_frames(64, 36, 6, kRate);
  VectorSource source(frames);
  std::vector<ExportProgress> progress;
  ExportJob job = ffv1_job(dir / "out.mkv", 64, 36, source);
  job.on_progress = [&](const ExportProgress& p) { progress.push_back(p); };

  auto summary = run_export(job);
  REQUIRE(summary.has_value());
  CHECK(summary->frames == 6);
  CHECK(std::filesystem::exists(dir / "out.mkv"));
  CHECK_FALSE(std::filesystem::exists(dir / "out.mkv.part"));
  CHECK_FALSE(std::filesystem::exists(dir / "out.mkv.video.part"));

  REQUIRE(progress.size() == 7);
  CHECK(progress[5].frames_done == 6);
  CHECK(progress[5].frames_total == 6);
  CHECK(progress[6].finishing);

  std::size_t index = 0;
  decode_video(dir / "out.mkv", [&](const AVFrame& frame) {
    REQUIRE(index < frames.size());
    REQUIRE(frame.format == AV_PIX_FMT_GBRP16);
    CHECK(frame.color_primaries == AVCOL_PRI_BT2020);
    CHECK(frame.color_trc == AVCOL_TRC_SMPTE2084);
    CHECK(frame.colorspace == AVCOL_SPC_RGB);
    const auto& expected = frames[index].pixels;
    std::size_t differences = 0;
    for (int y = 0; y < frame.height; ++y) {
      // GBRP : plan 0 = G, 1 = B, 2 = R.
      const auto* g = reinterpret_cast<const std::uint16_t*>(frame.data[0] + y * frame.linesize[0]);
      const auto* b = reinterpret_cast<const std::uint16_t*>(frame.data[1] + y * frame.linesize[1]);
      const auto* r = reinterpret_cast<const std::uint16_t*>(frame.data[2] + y * frame.linesize[2]);
      for (int x = 0; x < frame.width; ++x) {
        const std::size_t p = (static_cast<std::size_t>(y) * 64 + static_cast<std::size_t>(x)) * 4;
        differences += static_cast<std::size_t>(r[x] != expected[p]) +
                       static_cast<std::size_t>(g[x] != expected[p + 1]) +
                       static_cast<std::size_t>(b[x] != expected[p + 2]);
      }
    }
    CHECK(differences == 0);
    ++index;
  });
  CHECK(index == frames.size());
}

// Le YUV de l'export suit ses étiquettes : matrice BT.2020 non constante (Kr 0,2627, Kb 0,0593),
// plage limitée sur 10 bits. Une matrice BT.709 (Y 250 pour le rouge) ou BT.601 (Y 326), ou la
// pleine échelle, donneraient d'autres valeurs que ces références calculées à la main.
TEST_CASE("export YUV 4:2:0 10 bits : matrice BT.2020 et plage limitée", "[export][ffv1]") {
  struct Case {
    std::uint16_t r, g, b;
    int y, cb, cr;
  };
  // Y = 64 + 876 Y', Cb = 512 + 896 (B' - Y') / 1,8814, Cr = 512 + 896 (R' - Y') / 1,4746.
  const Case cases[] = {
      {.r = 65535, .g = 0, .b = 0, .y = 294, .cb = 387, .cr = 960},
      {.r = 0, .g = 0, .b = 65535, .y = 116, .cb = 960, .cr = 476},
      {.r = 65535, .g = 65535, .b = 65535, .y = 940, .cb = 512, .cr = 512},
      {.r = 0, .g = 0, .b = 0, .y = 64, .cb = 512, .cr = 512},
  };
  for (const Case& c : cases) {
    const auto dir = work_dir("yuv_matrix");
    std::vector<OwnedFrame> frames{flat_frame(32, 16, c.r, c.g, c.b)};
    VectorSource source(frames);
    ExportJob job = ffv1_job(dir / "out.mkv", 32, 16, source);
    job.encoder.format = PixelFormat::Yuv420p10;
    REQUIRE(run_export(job).has_value());

    int decoded = 0;
    decode_video(dir / "out.mkv", [&](const AVFrame& frame) {
      ++decoded;
      REQUIRE(frame.format == AV_PIX_FMT_YUV420P10);
      CHECK(frame.colorspace == AVCOL_SPC_BT2020_NCL);
      CHECK(frame.color_range == AVCOL_RANGE_MPEG);
      const auto sample = [&](int plane, int x, int y) {
        return static_cast<int>(
            reinterpret_cast<const std::uint16_t*>(frame.data[plane] + y * frame.linesize[plane])[x]);
      };
      INFO("RGB " << c.r << " " << c.g << " " << c.b << " -> " << sample(0, 16, 8) << " " << sample(1, 8, 4)
                  << " " << sample(2, 8, 4));
      CHECK(std::abs(sample(0, 16, 8) - c.y) <= 1);
      CHECK(std::abs(sample(1, 8, 4) - c.cb) <= 1);
      CHECK(std::abs(sample(2, 8, 4) - c.cr) <= 1);
    });
    CHECK(decoded == 1);
  }
}

TEST_CASE("métadonnées HDR10 relues dans les données annexes du flux", "[export][hdr10]") {
  const auto dir = work_dir("hdr10_side_data");
  auto frames = pattern_frames(32, 32, 3, kRate);
  frames[1].pixels[0] = 65535;  // un pixel à 10000 cd/m² : MaxCLL connu
  VectorSource source(frames);
  ExportJob job = ffv1_job(dir / "out.mkv", 32, 32, source);
  job.encoder.format = PixelFormat::Yuv420p10;
  job.mastering.max_nits = 600.0;
  job.tags = {{"AJP_PLAN", "0123abcd"}};

  LightMeter expected;
  for (const auto& f : frames) expected.add(f.view());
  auto summary = run_export(job);
  REQUIRE(summary.has_value());
  CHECK(summary->light == expected.level());
  CHECK(summary->light.max_cll == 10000);

  auto input = open_input((dir / "out.mkv").string());
  REQUIRE(input.has_value());
  REQUIRE((*input)->nb_streams == 1);
  const AVCodecParameters& par = *(*input)->streams[0]->codecpar;
  CHECK(par.codec_id == AV_CODEC_ID_FFV1);
  CHECK(par.format == AV_PIX_FMT_YUV420P10);
  CHECK(par.color_primaries == AVCOL_PRI_BT2020);
  CHECK(par.color_trc == AVCOL_TRC_SMPTE2084);
  CHECK(par.color_space == AVCOL_SPC_BT2020_NCL);
  CHECK(par.color_range == AVCOL_RANGE_MPEG);
  CHECK(tag((*input)->metadata, "AJP_PLAN") == "0123abcd");

  const auto* mdcv = side_data(par, AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
  REQUIRE(mdcv != nullptr);
  const auto* mastering = reinterpret_cast<const AVMasteringDisplayMetadata*>(mdcv->data);
  CHECK(mastering->has_luminance);
  CHECK(av_q2d(mastering->max_luminance) == 600.0);
  CHECK(av_q2d(mastering->min_luminance) == 0.0001);
  CHECK(av_q2d(mastering->display_primaries[0][0]) == 0.708);
  CHECK(av_q2d(mastering->white_point[1]) == 0.329);

  const auto* clli = side_data(par, AV_PKT_DATA_CONTENT_LIGHT_LEVEL);
  REQUIRE(clli != nullptr);
  const auto* light = reinterpret_cast<const AVContentLightMetadata*>(clli->data);
  CHECK(light->MaxCLL == summary->light.max_cll);
  CHECK(light->MaxFALL == summary->light.max_fall);
}

TEST_CASE("MP4 : mdcv et clli relus", "[export][hdr10]") {
  const auto dir = work_dir("hdr10_mp4");
  const auto frames = pattern_frames(32, 32, 3, kRate);
  VectorSource source(frames);
  ExportJob job = ffv1_job(dir / "out.mp4", 32, 32, source);
  job.container = Container::Mp4;
  job.encoder.format = PixelFormat::Yuv420p10;
  auto summary = run_export(job);
  REQUIRE(summary.has_value());

  auto input = open_input((dir / "out.mp4").string());
  REQUIRE(input.has_value());
  const AVCodecParameters& par = *(*input)->streams[0]->codecpar;
  CHECK(par.color_trc == AVCOL_TRC_SMPTE2084);
  const auto* mdcv = side_data(par, AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
  REQUIRE(mdcv != nullptr);
  CHECK(av_q2d(reinterpret_cast<const AVMasteringDisplayMetadata*>(mdcv->data)->max_luminance) == 1000.0);
  const auto* clli = side_data(par, AV_PKT_DATA_CONTENT_LIGHT_LEVEL);
  REQUIRE(clli != nullptr);
  CHECK(reinterpret_cast<const AVContentLightMetadata*>(clli->data)->MaxCLL == summary->light.max_cll);
  std::size_t decoded = 0;
  decode_video(dir / "out.mp4", [&](const AVFrame&) { ++decoded; });
  CHECK(decoded == 3);
}

TEST_CASE("audio, sous-titres, chapitres et polices recopiés de la source", "[export][remux]") {
  const auto dir = work_dir("remux_copy");
  write_source_media(dir / "source.mkv");
  const auto frames = pattern_frames(32, 18, 6, kRate);  // 0 à 250,25 ms
  VectorSource source(frames);
  ExportJob job = ffv1_job(dir / "out.mkv", 32, 18, source);
  job.copy_from = dir / "source.mkv";
  REQUIRE(run_export(job).has_value());

  auto input = open_input((dir / "out.mkv").string());
  REQUIRE(input.has_value());
  const AVFormatContext& out = **input;
  CHECK(tag(out.metadata, "title") == "Source synthétique");
  REQUIRE(out.nb_chapters == 2);
  CHECK(tag(out.chapters[0]->metadata, "title") == "Début");
  CHECK(av_rescale_q(out.chapters[1]->start, out.chapters[1]->time_base, kMicroseconds) ==
        SourceLayout::kChapterSplitUs);
  CHECK(av_rescale_q(out.chapters[1]->end, out.chapters[1]->time_base, kMicroseconds) ==
        250'250);  // fin de fenêtre

  const auto dumps = dump_streams(dir / "out.mkv");
  REQUIRE(dumps.size() == 4);
  CHECK(dumps[0].codec == AV_CODEC_ID_FFV1);
  CHECK(dumps[0].pts_us.size() == 6);
  CHECK(dumps[0].pts_us.front() == 0);
  CHECK(dumps[1].codec == AV_CODEC_ID_PCM_S16LE);
  CHECK(tag(out.streams[1]->metadata, "language") == "jpn");
  // Paquets de 21,33 ms commencés avant la fin de la dernière image (250,25 ms) : 12.
  REQUIRE(dumps[1].data.size() == 12);
  for (int k = 0; k < 12; ++k) CHECK(dumps[1].data[static_cast<std::size_t>(k)] == audio_payload(k));
  CHECK(dumps[2].codec == AV_CODEC_ID_SUBRIP);
  REQUIRE(dumps[2].data.size() == 1);  // le second sous-titre tombe après la fenêtre
  CHECK(std::string(dumps[2].data[0].begin(), dumps[2].data[0].end()) == "Ligne une");
  CHECK(dumps[2].pts_us[0] == SourceLayout::kSubtitle1Us);
  // 100 à 500 ms dans une fenêtre qui finit à 250,25 ms : coupé à la fin de la dernière image.
  CHECK(std::abs(dumps[2].duration_us[0] - (250'250 - SourceLayout::kSubtitle1Us)) <= 1000);
  CHECK(dumps[3].type == AVMEDIA_TYPE_ATTACHMENT);
  CHECK(tag(out.streams[3]->metadata, "filename") == SourceLayout::kFontName);
  const AVCodecParameters& font = *out.streams[3]->codecpar;
  REQUIRE(font.extradata_size == static_cast<int>(font_payload().size()));
  CHECK(std::memcmp(font.extradata, font_payload().data(), font_payload().size()) == 0);
}

TEST_CASE("horodatages recalés sur la première image exportée", "[export][remux]") {
  const auto dir = work_dir("remux_offset");
  write_source_media(dir / "source.mkv");
  // Les images 6 à 11 de la source (250,25 ms à 500,5 ms), sans sous-titres ni pièces jointes.
  const auto frames = pattern_frames(32, 18, 6, kRate, 6);
  VectorSource source(frames);
  ExportJob job = ffv1_job(dir / "out.mkv", 32, 18, source);
  job.copy_from = dir / "source.mkv";
  job.copy.subtitles = false;
  job.copy.attachments = false;
  REQUIRE(run_export(job).has_value());

  const auto dumps = dump_streams(dir / "out.mkv");
  REQUIRE(dumps.size() == 2);
  REQUIRE_FALSE(dumps[1].pts_us.empty());
  // L'écart entre l'audio et la vidéo de la source se garde, à la milliseconde de Matroska.
  const std::int64_t video_start = frame_pts(6, kRate);
  const std::int64_t audio_start =
      av_rescale(11, std::int64_t{1'000'000} * 1024, 48000);  // paquet chevauchant
  const std::int64_t gap = (dumps[1].pts_us.front() - dumps[0].pts_us.front()) - (audio_start - video_start);
  CHECK(gap >= -1000);
  CHECK(gap <= 1000);
  CHECK(dumps[1].data.front() == audio_payload(11));
}

TEST_CASE("sous-titre commencé avant la fenêtre : coupé à la première image", "[export][remux]") {
  const auto dir = work_dir("remux_subtitle_clip");
  write_source_media(dir / "source.mkv");
  // Fenêtre de 250,25 à 542,2 ms : le premier sous-titre (100 à 500 ms) la chevauche, le second
  // (600 ms) tombe après.
  const auto frames = pattern_frames(32, 18, 7, kRate, 6);
  VectorSource source(frames);
  ExportJob job = ffv1_job(dir / "out.mkv", 32, 18, source);
  job.copy_from = dir / "source.mkv";
  job.copy.audio = false;
  job.copy.attachments = false;
  REQUIRE(run_export(job).has_value());

  const auto dumps = dump_streams(dir / "out.mkv");
  REQUIRE(dumps.size() == 2);
  // Sans décalage de toute la sortie : la vidéo commence à 0 et le sous-titre aussi.
  REQUIRE_FALSE(dumps[0].pts_us.empty());
  CHECK(dumps[0].pts_us.front() == 0);
  REQUIRE(dumps[1].codec == AV_CODEC_ID_SUBRIP);
  REQUIRE(dumps[1].pts_us.size() == 1);
  CHECK(dumps[1].pts_us[0] == 0);
  const std::int64_t visible =
      SourceLayout::kSubtitle1Us + SourceLayout::kSubtitleDurationUs - frame_pts(6, kRate);
  CHECK(std::abs(dumps[1].duration_us[0] - visible) <= 1000);
  CHECK(std::string(dumps[1].data[0].begin(), dumps[1].data[0].end()) == "Ligne une");
}

}  // namespace ajp::exporter
