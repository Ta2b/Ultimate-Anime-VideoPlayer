// L'export par puits CUDA (export_sink.h) sans GPU : le puits sur des images hôte NV12 tenues comme
// celles du pool de NVENC, ce qui l'empêche, et l'étage gpu qui écrit ses sorties en YUV dans les
// images des slots au lieu du ring.
#include "app/play/export_sink.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

#include "app/play/live_settings.h"
#include "app/play/options.h"
#include "stage_rig.h"

namespace ajp::app::play {
namespace {

constexpr int kWidth = 64;
constexpr int kHeight = 36;

// Un pool factice : des images hôte NV12 de la taille donnée, dont il garde une référence pour
// savoir lesquelles sont encore tenues ailleurs.
struct HostPool {
  int width = kWidth;
  int height = kHeight;
  std::vector<AVBufferRef*> made;
  ~HostPool() {
    for (AVBufferRef*& buffer : made) av_buffer_unref(&buffer);
  }
  FrameSource source() {
    return [this]() -> std::expected<AVFrame*, Refusal> {
      AVFrame* frame = av_frame_alloc();
      frame->format = AV_PIX_FMT_NV12;
      frame->width = width;
      frame->height = height;
      if (av_frame_get_buffer(frame, 0) < 0) {
        av_frame_free(&frame);
        return std::unexpected(Refusal{RefusalCode::GpuMemoryExhausted, "pool factice"});
      }
      made.push_back(av_buffer_ref(frame->buf[0]));
      return frame;
    };
  }
  // Images encore tenues hors du pool factice.
  std::size_t held() const {
    std::size_t n = 0;
    for (AVBufferRef* buffer : made) n += av_buffer_get_ref_count(buffer) > 1 ? 1U : 0U;
    return n;
  }
};

ExportSinkSetup setup_for(HostPool& pool, std::uint32_t slots = 4) {
  return sink_setup(pool.source(), {.width = kWidth, .height = kHeight}, exporter::Layout::Nv12,
                    ColorInfo{.matrix = ColorMatrix::Bt709, .siting = ChromaSiting::TopLeft}, slots);
}

}  // namespace

TEST_CASE("Puits CUDA : ce qui l'empêche, et le flux qu'il écrit", "[app][export][sink]") {
  const ExportSize size{.width = 3840, .height = 2160};
  CHECK_FALSE(sink_blocker(true, true, exporter::Layout::P010, ColorMatrix::Bt709, size, size));
  CHECK_FALSE(sink_blocker(true, true, exporter::Layout::Nv12, ColorMatrix::Bt2020Ncl, size, size));
  CHECK(sink_blocker(false, true, exporter::Layout::P010, ColorMatrix::Bt709, size, size));
  CHECK(sink_blocker(true, false, exporter::Layout::P010, ColorMatrix::Bt709, size, size));
  CHECK(sink_blocker(true, true, exporter::Layout::Yuv420p10, ColorMatrix::Bt709, size, size));
  CHECK(sink_blocker(true, true, exporter::Layout::P010, ColorMatrix::Rgb, size, size));
  const auto scaled = sink_blocker(true, true, exporter::Layout::P010, ColorMatrix::Bt709,
                                   {.width = 1920, .height = 1080}, size);
  REQUIRE(scaled);
  CHECK(*scaled == "taille encodée 1920x1080 autre que la sortie de la chaîne 3840x2160");

  const ExportSinkSetup s = sink_setup(
      {}, size, exporter::Layout::P010,
      ColorInfo{.matrix = ColorMatrix::Bt2020Ncl, .range = ColorRange::Full, .siting = ChromaSiting::TopLeft},
      8);
  CHECK(s.format == kernels::Format::P010);
  CHECK(s.matrix == kernels::Matrix::Bt2020Ncl);
  CHECK(s.range == kernels::Range::Full);
  CHECK(s.siting == kernels::Siting::TopLeft);
  CHECK(s.slots == 8);
}

TEST_CASE("Puits CUDA : une image par slot, prise finie, remplacée par une neuve", "[app][export][sink]") {
  HostPool pool;
  auto sink = ExportSink::open(setup_for(pool));
  REQUIRE(sink);
  REQUIRE((*sink)->slots().size() == 4);
  CHECK(pool.held() == 4);
  for (std::uint32_t i = 0; i < 4; ++i) {
    const RingSlotDesc& desc = (*sink)->slots()[i];
    CHECK(desc.index == i);
    CHECK(desc.width == kWidth);
    CHECK(desc.height == kHeight);
    CHECK(desc.image == 0);
  }
  // Le thread gpu reçoit l'image du slot et son signal.
  const gpu::YuvTarget target = (*sink)->target(2, 5);
  CHECK(target.luma != 0);
  CHECK(target.chroma != 0);
  CHECK(target.luma_pitch >= kWidth);
  CHECK(target.width == kWidth);
  CHECK(target.format == kernels::Format::Nv12);
  CHECK(target.siting == kernels::Siting::TopLeft);
  CHECK(target.dither == kernels::Dither::Bayer8);
  REQUIRE(target.done != nullptr);
  CHECK(target.done_value == 5);
  const RingFrame frame{.slot = 2, .generation = 1, .pts_us = 40'000, .ready_value = 5};
  CHECK_FALSE((*sink)->ready(frame));
  target.done->store(5);  // le rappel du backend
  CHECK((*sink)->ready(frame));
  // L'image finie part, une neuve prend sa place : le slot repart sans attendre l'encodeur.
  auto taken = (*sink)->take(2);
  REQUIRE(taken);
  CHECK(reinterpret_cast<std::uintptr_t>((*taken)->data[0]) == target.luma);
  CHECK(pool.held() == 5);
  CHECK((*sink)->target(2, 7).luma != target.luma);
  (*sink)->target(2, 7).done->store(7);
  av_frame_free(&*taken);  // l'écriture l'a encodée
  CHECK(pool.held() == 4);
  sink->reset();  // toutes les écritures promises sont signalées : les images rentrent
  CHECK(pool.held() == 0);
}

TEST_CASE("Puits CUDA : refus de taille, de format et de nombre de slots", "[app][export][sink]") {
  HostPool pool;
  ExportSinkSetup none = setup_for(pool, 0);
  CHECK(ExportSink::open(std::move(none)).error().code == RefusalCode::FieldOutOfRange);
  ExportSinkSetup p016 = setup_for(pool);
  p016.format = kernels::Format::P016;
  CHECK(ExportSink::open(std::move(p016)).error().code == RefusalCode::FormatUnsupported);
  // Des images du pool plus petites que l'export : refusées, et rendues.
  HostPool small;
  small.width = kWidth / 2;
  auto refused = ExportSink::open(setup_for(small));
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::ShapeInvalid);
  CHECK(small.held() == 0);
}

TEST_CASE("Étage gpu : avec le puits de l'export, chaque sortie part en YUV dans l'image de son slot",
          "[app][export][sink][stage]") {
  test::StageRig rig;
  test::FakeEngine span{"span", {"input"}, {"output"}};
  test::FakeEngine motion{"rife_motion", {"input"}, {"flow", "mask"}};
  rig.backend.offer_ring_targets(true);  // le puits l'emporte sur l'écriture directe du ring
  rig.backend.offer_yuv_targets(true);
  HostPool pool;
  pool.width = kWidth * kSpanFactor;
  pool.height = kHeight * kSpanFactor;
  auto sink = ExportSink::open(sink_setup(pool.source(), {.width = pool.width, .height = pool.height},
                                          exporter::Layout::Nv12, ColorInfo{}, 4));
  REQUIRE(sink);
  auto live = std::make_shared<LiveSettings>();
  live->set_rife_factor(Factor{.num = 2, .den = 1});
  UpscaleSetup setup{.backend = &rig.backend,
                     .engine = &span,
                     .rife_motion = &motion,
                     .live = live,
                     .source_width = kWidth,
                     .source_height = kHeight,
                     .warp_after_span = true,
                     .slots = (*sink)->slots(),
                     .yuv_slots = *sink};
  auto started = UpscaleStage::start(rig.init(), std::move(setup));
  REQUIRE(started);
  rig.stage = std::move(*started);
  for (std::uint32_t id = 1; id <= 3; ++id)
    REQUIRE(rig.push(test::host_frame(id, kWidth, kHeight, std::int64_t{id} * 40'000)));
  REQUIRE(rig.collect(5));  // quatre slots : chacun rendu dès son image reçue
  std::vector<RingFrame> frames = rig.sink.frames();
  rig.stage.reset();
  const std::vector<test::YuvWritten> written = rig.backend.yuv_written();
  REQUIRE(written.size() == frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    INFO("sortie " << i);
    CHECK(written[i].done_value == frames[i].ready_value);
    CHECK((*sink)->ready(frames[i]));
  }
  CHECK(rig.backend.written().empty());  // ni write_ring, ni warp dans le ring
  CHECK(rig.backend.count("warp_blend_ring") == 0);
  CHECK(rig.backend.count("warp_blend") == 2);
  CHECK(rig.backend.count("write_yuv") == frames.size());
  CHECK(rig.reported().empty());
}

TEST_CASE("Étage gpu : le puits de l'export exige write_yuv du backend", "[app][export][sink][stage]") {
  test::StageRig rig;
  test::FakeEngine span{"span", {"input"}, {"output"}};
  HostPool pool;
  pool.width = kWidth * kSpanFactor;
  pool.height = kHeight * kSpanFactor;
  auto sink = ExportSink::open(sink_setup(pool.source(), {.width = pool.width, .height = pool.height},
                                          exporter::Layout::Nv12, ColorInfo{}, 4));
  REQUIRE(sink);
  UpscaleSetup setup{.backend = &rig.backend,
                     .engine = &span,
                     .source_width = kWidth,
                     .source_height = kHeight,
                     .slots = (*sink)->slots(),
                     .yuv_slots = *sink};
  auto started = UpscaleStage::start(rig.init(), std::move(setup));
  REQUIRE_FALSE(started);
  CHECK(started.error().code == RefusalCode::FormatUnsupported);
}

}  // namespace ajp::app::play
