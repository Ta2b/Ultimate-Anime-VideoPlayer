// Encodeurs matériels réels : NVENC HEVC et AV1 10 bits sur la RTX ([nvidia]), encodeurs Vulkan
// ([vulkan]). Sans l'appareil ou le pilote, l'ouverture rend EncoderUnavailable et le test se
// SKIP avec la raison ; jamais de réussite silencieuse.
#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>

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

// Quelques images rejouées en boucle, pts croissants : une longue séquence sans la garder en mémoire.
class CyclingSource final : public ExportFrameSource {
 public:
  CyclingSource(int width, int height, int count) : count_(count) {
    for (int i = 0; i < 4; ++i) frames_.push_back(pattern_frame(width, height, i, 0));
  }
  std::expected<std::optional<SourceFrame>, Refusal> next() override {
    if (index_ >= count_) return std::nullopt;
    HostFrame frame = frames_[static_cast<std::size_t>(index_ % 4)].view();
    frame.pts_us = frame_pts(index_++, kRate);
    return SourceFrame{frame};
  }
  std::optional<std::int64_t> total_frames() const override { return count_; }

 private:
  std::vector<OwnedFrame> frames_;
  int count_;
  int index_ = 0;
};

ExportJob hardware_job(const std::filesystem::path& output, int width, int height, EncoderSettings settings,
                       ExportFrameSource& source) {
  return ExportJob{.output = output,
                   .width = width,
                   .height = height,
                   .encoder = std::move(settings),
                   .mastering = Hdr10Mastering{.max_nits = 1000.0},
                   .source = &source};
}

// Lance l'export ; SKIP quand l'encodeur ne s'ouvre pas sur cette machine.
ExportSummary run_or_skip(const ExportJob& job) {
  auto summary = run_export(job);
  if (!summary && summary.error().code == RefusalCode::EncoderUnavailable) {
    SKIP(std::format("{} indisponible ici : {}", job.encoder.encoder, summary.error().detail));
  }
  if (!summary) FAIL(std::format("{} : {}", name(summary.error().code), summary.error().detail));
  return *summary;
}

void check_hdr10_stream(const std::filesystem::path& path, AVCodecID codec, int profile,
                        const ExportSummary& summary, int frames) {
  auto input = open_input(path.string());
  REQUIRE(input.has_value());
  const AVCodecParameters& par = *(*input)->streams[0]->codecpar;
  CHECK(par.codec_id == codec);
  CHECK(par.profile == profile);
  CHECK(par.format == AV_PIX_FMT_YUV420P10);
  CHECK(par.color_primaries == AVCOL_PRI_BT2020);
  CHECK(par.color_trc == AVCOL_TRC_SMPTE2084);
  CHECK(par.color_space == AVCOL_SPC_BT2020_NCL);
  CHECK(par.color_range == AVCOL_RANGE_MPEG);
  const auto* mdcv = av_packet_side_data_get(par.coded_side_data, par.nb_coded_side_data,
                                             AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
  REQUIRE(mdcv != nullptr);
  CHECK(av_q2d(reinterpret_cast<const AVMasteringDisplayMetadata*>(mdcv->data)->max_luminance) == 1000.0);
  const auto* clli =
      av_packet_side_data_get(par.coded_side_data, par.nb_coded_side_data, AV_PKT_DATA_CONTENT_LIGHT_LEVEL);
  REQUIRE(clli != nullptr);
  CHECK(reinterpret_cast<const AVContentLightMetadata*>(clli->data)->MaxCLL == summary.light.max_cll);
  CHECK(reinterpret_cast<const AVContentLightMetadata*>(clli->data)->MaxFALL == summary.light.max_fall);

  int decoded = 0;
  int with_mastering = 0;
  decode_video(path, [&](const AVFrame& frame) {
    ++decoded;
    with_mastering +=
        static_cast<int>(av_frame_get_side_data(&frame, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA) != nullptr);
  });
  CHECK(decoded == frames);
  CHECK(with_mastering > 0);  // le mastering est aussi dans le flux (SEI ou OBU de métadonnées)
}

EncoderSettings nvenc(const char* encoder) {
  return EncoderSettings{.encoder = encoder,
                         .format = PixelFormat::P010,
                         .rate = RateControl::ConstantQuality,
                         .quality = 24,
                         .preset = 4,
                         .keyframe_interval = 48};
}

}  // namespace

TEST_CASE("NVENC HEVC 10 bits réel : étiquettes, HDR10 et débit", "[export][nvidia]") {
  const auto dir = work_dir("nvenc_hevc");
  CyclingSource source(1920, 1080, 96);
  const auto summary = run_or_skip(hardware_job(dir / "out.mkv", 1920, 1080, nvenc("hevc_nvenc"), source));
  CHECK(summary.frames == 96);
  WARN(std::format("hevc_nvenc 1920x1080 P010 : {:.1f} im/s (conversion libswscale comprise)",
                   summary.frames_per_second));
  check_hdr10_stream(dir / "out.mkv", AV_CODEC_ID_HEVC, AV_PROFILE_HEVC_MAIN_10, summary, 96);
}

TEST_CASE("NVENC AV1 10 bits réel en MP4", "[export][nvidia]") {
  const auto dir = work_dir("nvenc_av1");
  CyclingSource source(1280, 720, 48);
  ExportJob job = hardware_job(dir / "out.mp4", 1280, 720, nvenc("av1_nvenc"), source);
  job.container = Container::Mp4;
  const auto summary = run_or_skip(job);
  WARN(std::format("av1_nvenc 1280x720 P010 : {:.1f} im/s", summary.frames_per_second));
  check_hdr10_stream(dir / "out.mp4", AV_CODEC_ID_AV1, AV_PROFILE_AV1_MAIN, summary, 48);
}

TEST_CASE("encodeur Vulkan HEVC 10 bits, s'il est présent", "[export][vulkan]") {
  const auto dir = work_dir("vulkan_hevc");
  CyclingSource source(640, 360, 24);
  const EncoderSettings settings{
      .encoder = "hevc_vulkan", .format = PixelFormat::P010, .rate = RateControl::ConstantQp, .quality = 24};
  const auto summary = run_or_skip(hardware_job(dir / "out.mkv", 640, 360, settings, source));
  CHECK(summary.frames == 24);
  check_hdr10_stream(dir / "out.mkv", AV_CODEC_ID_HEVC, AV_PROFILE_HEVC_MAIN_10, summary, 24);
}

// Mesure du débit NVENC, hors suite par défaut : ajp-gpu ajp_export_tests "[.bench]".
TEST_CASE("banc NVENC HEVC 1080p : échauffement, cinq passes, médiane", "[.bench][nvidia]") {
  const auto dir = work_dir("nvenc_bench");
  std::vector<double> fps;
  for (int pass = 0; pass < 6; ++pass) {
    CyclingSource source(1920, 1080, 240);
    const auto summary = run_or_skip(hardware_job(dir / "out.mkv", 1920, 1080, nvenc("hevc_nvenc"), source));
    if (pass > 0) fps.push_back(summary.frames_per_second);  // la passe 0 échauffe
  }
  std::ranges::sort(fps);
  const double mean = std::accumulate(fps.begin(), fps.end(), 0.0) / static_cast<double>(fps.size());
  double variance = 0.0;
  for (double f : fps) variance += (f - mean) * (f - mean);
  WARN(std::format(
      "hevc_nvenc 1080p P010, 240 images : médiane {:.1f} im/s, écart type {:.1f}, min {:.1f}, max {:.1f}",
      fps[fps.size() / 2], std::sqrt(variance / static_cast<double>(fps.size())), fps.front(), fps.back()));
}

}  // namespace ajp::exporter
