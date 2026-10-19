// Exécution de bout en bout sur le backend CPU : tailles, valeurs, suite des pts, REPEATED,
// coupes, identités dérivées et absence d'allocation en régime.
#include "gpu/executor.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/cadence.h"
#include "gpu/cpu/host_tensor.h"
#include "gpu/level_scheduler.h"
#include "kernels/ref/color.h"
#include "kernels/ref/rife.h"
#include "pipeline.h"

using ajp::DecodedFrame;
using ajp::RingFrame;
using ajp::SourceId;
using namespace ajp::gpu;
namespace core = ajp::core;
namespace kernels = ajp::kernels;
namespace ref = ajp::kernels::ref;

namespace {

kernels::RgbImage nearest_x2(const kernels::RgbImage& in) {
  kernels::RgbImage out(2 * in.width(), 2 * in.height());
  for (std::size_t c = 0; c < 3; ++c) {
    for (int y = 0; y < out.height(); ++y) {
      for (int x = 0; x < out.width(); ++x) out.planes[c].at(x, y) = in.planes[c].at(x / 2, y / 2);
    }
  }
  return out;
}

kernels::YuvImage yuv(const DecodedFrame& frame) {
  auto image = cpu::read_frame(frame);
  REQUIRE(image);
  return *image;
}

// Une source montrée telle quelle : ingestion de la chaîne puis agrandissement.
std::vector<std::uint16_t> expected_source(const DecodedFrame& frame) {
  auto rgb = ref::yuv_to_rgb(yuv(frame), {});
  REQUIRE(rgb);
  return *ref::to_rgba16(nearest_x2(*rgb));
}

// L'instant t prédit par le RIFE des tests (a + t (b − a)) sur les entrées préparées de RIFE.
std::vector<std::uint16_t> expected_blend(const DecodedFrame& a, const DecodedFrame& b, float t) {
  auto pa = ref::rife_prepare(yuv(a), kernels::Range::Limited);
  auto pb = ref::rife_prepare(yuv(b), kernels::Range::Limited);
  REQUIRE(pa);
  REQUIRE(pb);
  kernels::RgbImage blend = *pa;
  for (std::size_t c = 0; c < 3; ++c) {
    for (std::size_t i = 0; i < blend.planes[c].data.size(); ++i) {
      const float d = pb->planes[c].data[i] - pa->planes[c].data[i];
      blend.planes[c].data[i] = pa->planes[c].data[i] + d * t;
    }
  }
  auto crop = ref::rife_crop(blend, kernels::rife_geometry(a.width, a.height));
  REQUIRE(crop);
  return *ref::to_rgba16(nearest_x2(*crop));
}

// Égalité au bit de deux images RGBA16, sans imprimer des milliers de valeurs en cas d'écart.
bool same(const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& want) {
  if (got.size() != want.size()) return false;
  const auto [g, w] = std::ranges::mismatch(got, want);
  if (g == got.end()) return true;
  UNSCOPED_INFO("premier écart à " << (g - got.begin()) << " : " << *g << " au lieu de " << *w);
  return false;
}

std::vector<DecodedFrame> frames(std::initializer_list<std::uint8_t> lumas) {
  std::vector<DecodedFrame> result;
  std::uint32_t id = 0;
  for (const std::uint8_t luma : lumas) result.push_back(test::make_frame(id++, luma));
  return result;
}

core::Plan quality(core::Ratio factor) {
  return test::with_level(test::upscale_plan(), factor, core::Policy::Quality, core::Schedule::BeforeUpscale);
}

core::Plan fast(core::Plan plan, core::Ratio factor = {2, 1}) {
  return test::with_level(std::move(plan), factor, core::Policy::Fast, core::Schedule::OutputRes,
                          core::Pacing::Source, "motion");
}

}  // namespace

TEST_CASE("Exécution S0 : chaque source agrandie, à son pts, aux valeurs de la référence", "[executor]") {
  test::Pipeline pipeline;
  pipeline.install(test::upscale_plan());
  const auto sources = frames({40, 90, 140});
  for (const DecodedFrame& frame : sources) pipeline.push(frame);

  REQUIRE(pipeline.sink.frames.size() == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    const RingFrame& out = pipeline.sink.frames[i];
    CHECK(out.pts_us == sources[i].pts_us);
    CHECK(out.left == sources[i].id);
    CHECK(out.right == sources[i].id);
    CHECK(out.t_num == 0);
    CHECK_FALSE(out.repeated);
    CHECK(out.ready_value % 2 == 1);
    CHECK(same(pipeline.pixels[i], expected_source(sources[i])));
  }
  CHECK(pipeline.pixels[0].size() ==
        std::size_t{4} * 2 * test::kWidth * 2 * test::kHeight);  // 128 × 72 RGBA16
  CHECK(pipeline.refusals.empty());
  CHECK(pipeline.worker.stats().written == 3);
  CHECK(pipeline.worker.stats().host_waits == 0);
}

TEST_CASE("Exécution S1 : la suite des pts de x2, x3 et 5/2 est celle de l'accumulateur", "[executor]") {
  for (const core::Ratio factor : {core::Ratio{2, 1}, core::Ratio{3, 1}, core::Ratio{5, 2}}) {
    test::Pipeline pipeline;
    pipeline.install(quality(factor));
    const auto sources = frames({40, 44, 48, 52, 56, 60});
    auto cadence = core::CadenceAccumulator::create(factor);
    REQUIRE(cadence);
    std::vector<std::int64_t> expected;
    for (const DecodedFrame& frame : sources) {
      pipeline.push(frame);
      const core::CadenceStep step = cadence->push(core::stamp_of(frame));
      for (const core::OutputInstant& instant : step.instants()) expected.push_back(*instant.pts_us);
    }
    std::vector<std::int64_t> pts;
    for (const RingFrame& out : pipeline.sink.frames) pts.push_back(out.pts_us);
    CHECK(pts == expected);
    CHECK(pipeline.refusals.empty());
    const auto interpolated =
        std::ranges::count_if(pipeline.sink.frames, [](const RingFrame& f) { return f.t_num != 0; });
    CHECK(static_cast<std::uint64_t>(interpolated) == pipeline.worker.stats().interpolated);
    CHECK(interpolated > 0);
  }
}

TEST_CASE("Exécution S1 : l'instant prédit par RIFE, recadré puis agrandi", "[executor]") {
  test::Pipeline pipeline;
  pipeline.install(quality({2, 1}));
  const auto sources = frames({40, 100});  // un écart de 60 niveaux : sous le seuil de coupe
  for (const DecodedFrame& frame : sources) pipeline.push(frame);
  REQUIRE(pipeline.sink.frames.size() == 3);
  const RingFrame& middle = pipeline.sink.frames[1];
  CHECK_FALSE(middle.repeated);
  CHECK_FALSE(middle.scene_cut);
  CHECK(middle.left == SourceId{0});
  CHECK(middle.right == SourceId{1});
  CHECK(middle.t_num == 1);
  CHECK(middle.t_den == 2);
  CHECK(pipeline.refusals.empty());  // ABI 3 : un refus de moteur arrive au rapporteur
  CHECK(same(pipeline.pixels[1], expected_blend(sources[0], sources[1], 0.5F)));
  CHECK(same(pipeline.pixels[2], expected_source(sources[1])));
}

TEST_CASE("Exécution S3 Fast : un dessin tenu est REPEATED, le mouvement est déformé à la sortie",
          "[executor]") {
  test::Pipeline pipeline;
  pipeline.install(fast(test::upscale_plan()));
  // En deux : chaque dessin tient deux sources identiques.
  const auto sources = frames({50, 50, 60, 60, 70});
  for (const DecodedFrame& frame : sources) pipeline.push(frame);
  const auto& out = pipeline.sink.frames;
  REQUIRE(out.size() == 9);
  CHECK(out[1].repeated);  // paire (0, 1) : le même dessin
  CHECK(out[1].left == SourceId{0});
  CHECK(same(pipeline.pixels[1], pipeline.pixels[0]));  // rien de recalculé
  CHECK_FALSE(out[3].repeated);                         // paire (1, 2) : nouveau dessin, warp à t = 1/2
  CHECK(out[3].t_num == 1);
  CHECK(out[5].repeated);
  CHECK(pipeline.worker.stats().repeated == 2);
  CHECK(pipeline.worker.stats().scene_cuts == 0);
  // Mouvement nul et masque 0,5 : la loi de mélange à t = 1/2 rend la moyenne des deux images.
  CHECK(pipeline.pixels[3] != pipeline.pixels[2]);
  CHECK(pipeline.refusals.empty());
}

TEST_CASE("Exécution : une coupe de scène n'est jamais interpolée", "[executor]") {
  test::Pipeline pipeline;
  pipeline.install(quality({2, 1}));
  const auto sources = frames({50, 54, 230, 226});
  for (const DecodedFrame& frame : sources) pipeline.push(frame);
  const auto& out = pipeline.sink.frames;
  REQUIRE(out.size() == 7);
  CHECK_FALSE(out[1].scene_cut);
  CHECK(out[3].scene_cut);  // paire (1, 2)
  CHECK(out[3].repeated);
  CHECK(out[3].left == SourceId{1});
  CHECK(out[3].right == SourceId{2});
  CHECK(same(pipeline.pixels[3], expected_source(sources[1])));  // l'image de gauche reste
  CHECK_FALSE(out[5].scene_cut);
  CHECK(pipeline.worker.stats().scene_cuts == 1);
}

TEST_CASE("Exécution : deux niveaux, la sortie du premier devient la source du second", "[executor]") {
  test::Pipeline pipeline;
  pipeline.install(fast(quality({2, 1})));
  const auto sources = frames({40, 60, 80, 100});
  for (const DecodedFrame& frame : sources) pipeline.push(frame);
  const auto& out = pipeline.sink.frames;
  // x2 puis x2 : une sortie tous les 10 ms, la première source comprise.
  REQUIRE(out.size() >= 9);
  for (std::size_t i = 0; i < out.size(); ++i) CHECK(out[i].pts_us == std::int64_t(i) * 10'000);
  for (const RingFrame& frame : out) {
    CHECK(is_derived(frame.left));  // les entrées du second niveau sont des sorties du premier
    CHECK(((frame.left.value >> 28) & 0x7U) == 0);
  }
  CHECK(std::ranges::any_of(out, [](const RingFrame& f) { return f.t_num == 1 && f.t_den == 2; }));
  CHECK(pipeline.refusals.empty());
}

TEST_CASE("Exécution : en régime, plus aucun tenseur n'est créé", "[executor]") {
  test::Pipeline pipeline;
  Executor executor(pipeline.program(fast(quality({2, 1}))));
  std::size_t warmed = 0;
  for (std::uint32_t id = 0; id < 24; ++id) {
    REQUIRE(executor.push(test::make_frame(id, static_cast<std::uint8_t>(40 + (id % 6) * 8))));
    while (executor.has_output()) executor.pop();
    if (id == 8) warmed = executor.tensors_created();
  }
  CHECK(warmed > 0);
  CHECK(executor.tensors_created() == warmed);
  CHECK(executor.counters().sources == 24);
}
