// Refus et annulation de run_export : aucun fichier ne reste derrière un export qui échoue.
#include <stop_token>

#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
}

#include "export/export.h"
#include "support.h"

namespace ajp::exporter {
namespace {

using namespace test;

constexpr FrameRate kRate{24000, 1001};

bool directory_empty_except(const std::filesystem::path& dir, const std::filesystem::path& keep) {
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path() != keep) return false;
  }
  return true;
}

RefusalCode refusal_of(const ExportJob& job) {
  auto result = run_export(job);
  REQUIRE_FALSE(result.has_value());
  return result.error().code;
}

}  // namespace

TEST_CASE("refus du travail avant toute ouverture", "[export][refus]") {
  const auto dir = work_dir("refusals");
  const auto frames = pattern_frames(32, 18, 2, kRate);
  VectorSource source(frames);

  ExportJob unknown = ffv1_job(dir / "a.mkv", 32, 18, source);
  unknown.encoder.encoder = "libx265";
  CHECK(refusal_of(unknown) == RefusalCode::EncoderUnavailable);

  ExportJob odd = ffv1_job(dir / "b.mkv", 33, 18, source);
  odd.encoder.format = PixelFormat::Yuv420p10;
  CHECK(refusal_of(odd) == RefusalCode::FieldOutOfRange);

  ExportJob quality = ffv1_job(dir / "c.mkv", 32, 18, source);
  quality.encoder = EncoderSettings{
      .encoder = "hevc_nvenc", .format = PixelFormat::P010, .rate = RateControl::ConstantQp, .quality = 70};
  CHECK(refusal_of(quality) == RefusalCode::FieldOutOfRange);

  ExportJob rate = ffv1_job(dir / "d.mkv", 32, 18, source);
  rate.frame_rate = FrameRate{24, 0};
  CHECK(refusal_of(rate) == RefusalCode::FieldOutOfRange);

  ExportJob no_source = ffv1_job(dir / "e.mkv", 32, 18, source);
  no_source.source = nullptr;
  CHECK(refusal_of(no_source) == RefusalCode::FieldMissing);

  ExportJob no_output = ffv1_job({}, 32, 18, source);
  CHECK(refusal_of(no_output) == RefusalCode::FieldMissing);

  ExportJob missing = ffv1_job(dir / "f.mkv", 32, 18, source);
  missing.copy_from = dir / "absent.mkv";
  CHECK(refusal_of(missing) == RefusalCode::MediaUnreadable);

  CHECK(std::filesystem::is_empty(dir));
}

TEST_CASE("encodeur absent de la construction de FFmpeg refusé à l'ouverture", "[export][refus]") {
  const auto dir = work_dir("refusal_build");
  if (avcodec_find_encoder_by_name("libsvtav1") != nullptr)
    SKIP("libsvtav1 est dans cette construction de FFmpeg");
  const auto frames = pattern_frames(32, 18, 2, kRate);
  VectorSource source(frames);
  ExportJob job = ffv1_job(dir / "svt.mkv", 32, 18, source);
  job.encoder = EncoderSettings{.encoder = "libsvtav1",
                                .format = PixelFormat::Yuv420p10,
                                .rate = RateControl::ConstantQuality,
                                .quality = 30};
  auto result = run_export(job);
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().code == RefusalCode::EncoderUnavailable);
  CHECK(result.error().detail.find("absent de cette construction") != std::string::npos);
  CHECK(std::filesystem::is_empty(dir));
}

TEST_CASE("piste que le conteneur ne porte pas : refus avant l'encodage", "[export][refus]") {
  const auto dir = work_dir("refusal_container");
  write_source_media(dir / "source.mkv");
  const auto frames = pattern_frames(32, 18, 2, kRate);
  VectorSource source(frames);

  // MP4 ne porte ni SubRip ni pièce jointe : la piste est nommée dans le refus.
  ExportJob subs_mp4 = ffv1_job(dir / "a.mp4", 32, 18, source);
  subs_mp4.container = Container::Mp4;
  subs_mp4.copy_from = dir / "source.mkv";
  auto result = run_export(subs_mp4);
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().code == RefusalCode::FormatUnsupported);
  CHECK(result.error().detail.find("piste source") != std::string::npos);

  ExportJob fonts_mp4 = subs_mp4;
  fonts_mp4.copy.subtitles = false;
  fonts_mp4.copy.audio = false;
  result = run_export(fonts_mp4);
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().code == RefusalCode::FormatUnsupported);
  CHECK(result.error().detail.find("pièce jointe") != std::string::npos);
  CHECK(directory_empty_except(dir, dir / "source.mkv"));
}

TEST_CASE("images hors contrat refusées", "[export][refus]") {
  const auto dir = work_dir("refusal_frames");

  auto frames = pattern_frames(32, 18, 3, kRate);
  frames[2].pts_us = frames[1].pts_us;  // pts qui ne croît pas
  VectorSource repeat(frames);
  CHECK(refusal_of(ffv1_job(dir / "a.mkv", 32, 18, repeat)) == RefusalCode::InputMismatch);

  const auto wrong = pattern_frames(30, 18, 2, kRate);
  VectorSource wrong_size(wrong);
  CHECK(refusal_of(ffv1_job(dir / "b.mkv", 32, 18, wrong_size)) == RefusalCode::InputMismatch);

  auto unknown = pattern_frames(32, 18, 1, kRate);
  unknown[0].pts_us = kNoPts;
  VectorSource no_pts(unknown);
  CHECK(refusal_of(ffv1_job(dir / "c.mkv", 32, 18, no_pts)) == RefusalCode::InputMismatch);

  const std::vector<OwnedFrame> none;
  VectorSource empty(none);
  CHECK(refusal_of(ffv1_job(dir / "d.mkv", 32, 18, empty)) == RefusalCode::InputMismatch);

  CHECK(std::filesystem::is_empty(dir));
}

TEST_CASE("voie appareil décrite mais pas encore branchée : refus nommé", "[export][refus]") {
  class DeviceSource final : public ExportFrameSource {
   public:
    std::expected<std::optional<SourceFrame>, Refusal> next() override {
      return SourceFrame{DeviceFrame{.width = 32, .height = 18, .pts_us = 0}};
    }
  } source;
  const auto dir = work_dir("refusal_device");
  auto result = run_export(ffv1_job(dir / "a.mkv", 32, 18, source));
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().code == RefusalCode::FormatUnsupported);
  CHECK(std::filesystem::is_empty(dir));
}

TEST_CASE("annulation pendant l'encodage : refus Cancelled, aucun fichier", "[export][annulation]") {
  const auto dir = work_dir("cancel");
  write_source_media(dir / "source.mkv");
  const auto frames = pattern_frames(32, 18, 8, kRate);
  VectorSource source(frames);
  std::stop_source stop;
  source.before_next = [&](std::size_t index) {
    if (index == 4) stop.request_stop();
  };
  ExportJob job = ffv1_job(dir / "out.mkv", 32, 18, source);
  job.copy_from = dir / "source.mkv";
  job.stop = stop.get_token();
  int published = 0;
  job.on_progress = [&](const ExportProgress&) { ++published; };

  auto result = run_export(job);
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().code == RefusalCode::Cancelled);
  CHECK(published == 4);
  CHECK(directory_empty_except(dir, dir / "source.mkv"));
}

TEST_CASE("refus de la source rendu tel quel", "[export][refus]") {
  class FailingSource final : public ExportFrameSource {
   public:
    std::expected<std::optional<SourceFrame>, Refusal> next() override {
      return std::unexpected(Refusal{RefusalCode::DecodeFailed, "source : image illisible"});
    }
  } source;
  const auto dir = work_dir("refusal_source");
  auto result = run_export(ffv1_job(dir / "a.mkv", 32, 18, source));
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().code == RefusalCode::DecodeFailed);
  CHECK(result.error().detail == "source : image illisible");
  CHECK(std::filesystem::is_empty(dir));
}

}  // namespace ajp::exporter
