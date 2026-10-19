// Travailleur du thread gpu, pas à pas : contre-pression du ring, Flush, Install en cours de
// flux, refus.
#include "gpu/worker.h"

#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "pipeline.h"

using ajp::DecodedFrame;
using ajp::RefusalCode;
using ajp::RingFrame;
using ajp::SourceId;
using namespace ajp::gpu;
namespace core = ajp::core;

namespace {

core::Plan quality_x2() {
  return test::with_level(test::upscale_plan(), {2, 1}, core::Policy::Quality, core::Schedule::BeforeUpscale);
}

std::vector<std::int64_t> pts_of(const std::vector<RingFrame>& frames) {
  std::vector<std::int64_t> pts;
  for (const RingFrame& frame : frames) pts.push_back(frame.pts_us);
  return pts;
}

}  // namespace

TEST_CASE("Travailleur : sans slot libre, il garde les images décodées et attend SlotFree", "[worker]") {
  test::Pipeline pipeline(2);
  pipeline.auto_release = false;
  pipeline.install(test::upscale_plan());
  for (std::uint32_t id = 0; id < 4; ++id) REQUIRE(pipeline.worker.frame(test::make_frame(id, 50)));
  pipeline.worker.pump();
  // Deux slots écrits, la troisième image calculée attend un slot : la quatrième n'est pas prise.
  CHECK(pipeline.sink.frames.size() == 2);
  CHECK(pipeline.worker.take_consumed() == 3);
  REQUIRE(pipeline.worker.output_ring() != nullptr);
  CHECK(pipeline.worker.output_ring()->count(SlotState::Shown) == 2);

  pipeline.release();  // main rend ses slots : tout repart
  CHECK(pipeline.sink.frames.size() == 4);
  CHECK(pipeline.worker.take_consumed() == 1);
  CHECK(pts_of(pipeline.sink.frames) == std::vector<std::int64_t>{0, 40'000, 80'000, 120'000});
}

TEST_CASE("Travailleur : main plein, l'image écrite attend dans son slot", "[worker]") {
  test::Pipeline pipeline(3);
  pipeline.sink = test::CollectingSink(1);  // sans copie des slots : on_frame vide
  pipeline.auto_release = false;
  pipeline.install(test::upscale_plan());
  for (std::uint32_t id = 0; id < 3; ++id) REQUIRE(pipeline.worker.frame(test::make_frame(id, 50)));
  pipeline.worker.pump();
  CHECK(pipeline.sink.frames.size() == 1);
  CHECK(pipeline.worker.output_ring()->count(SlotState::Ready) == 2);
  pipeline.release();
  CHECK(pipeline.sink.frames.size() == 3);
}

TEST_CASE("Travailleur : Flush jette l'époque passée sans attendre", "[worker]") {
  test::Pipeline pipeline(2);
  pipeline.auto_release = false;
  pipeline.install(quality_x2());
  pipeline.push(test::make_frame(0, 40));
  pipeline.push(test::make_frame(1, 60));
  REQUIRE(pipeline.worker.frame(test::make_frame(2, 80)));  // encore dans la file au Flush
  CHECK(pipeline.sink.frames.size() == 2);                  // la troisième sortie attend un slot

  pipeline.worker.flush(ajp::Flush{.epoch = 1});
  CHECK(pipeline.worker.take_consumed() == 3);  // deux traitées, une jetée
  CHECK(pipeline.worker.stats().dropped >= 2);
  pipeline.release();  // main rend aussi les images de l'époque 0, qu'il jette
  pipeline.auto_release = true;
  pipeline.push(test::make_frame(10, 40, 1, 400'000));
  pipeline.push(test::make_frame(11, 60, 1, 440'000));

  const auto& out = pipeline.sink.frames;
  REQUIRE(out.size() == 5);
  const RingFrame& restart = out[2];
  CHECK(restart.epoch == 1);
  CHECK(restart.left == SourceId{10});
  CHECK(restart.t_num == 0);  // la nouvelle origine, sans paire avec l'époque passée
  CHECK(out[3].pts_us == 420'000);
  CHECK(std::ranges::all_of(out.begin() + 2, out.end(), [](const RingFrame& f) { return f.epoch == 1; }));
  CHECK(pipeline.refusals.empty());
}

TEST_CASE("Travailleur : un Install s'applique à la paire suivante, sans Flush", "[worker]") {
  test::Pipeline pipeline;
  pipeline.install(test::upscale_plan());
  const std::uint64_t first = pipeline.worker.stats().plan_hash;
  for (std::uint32_t id = 0; id < 3; ++id)
    pipeline.push(test::make_frame(id, static_cast<std::uint8_t>(40 + 10 * id)));
  pipeline.install(quality_x2());
  for (std::uint32_t id = 3; id < 5; ++id)
    pipeline.push(test::make_frame(id, static_cast<std::uint8_t>(40 + 10 * id)));

  // La dernière source de l'ancien Program devient l'origine du nouveau : rien n'est montré deux
  // fois, et la paire (2, 3) est déjà interpolée.
  CHECK(pts_of(pipeline.sink.frames) ==
        std::vector<std::int64_t>{0, 40'000, 80'000, 100'000, 120'000, 140'000, 160'000});
  CHECK(pipeline.sink.frames[3].left == SourceId{2});
  CHECK(pipeline.sink.frames[3].right == SourceId{3});
  CHECK(pipeline.worker.stats().plan_hash != first);
  CHECK(pipeline.refusals.empty());

  SECTION("un ring d'une autre génération remplace l'ancien") {
    pipeline.install(test::upscale_plan(), 2);
    pipeline.push(test::make_frame(5, 90));
    CHECK(pipeline.sink.frames.back().generation == 2);
    CHECK(pipeline.sink.frames.back().pts_us == 200'000);
    // Un SlotFree de l'ancien ring est ignoré.
    pipeline.worker.slot_free({.slot = 0, .semaphore_value = 2, .generation = 1});
    CHECK(pipeline.refusals.empty());
  }
}

TEST_CASE("Travailleur : refus rapportés, flux qui continue", "[worker]") {
  test::Pipeline pipeline;
  pipeline.install(test::upscale_plan());
  pipeline.push(test::make_frame(0, 40, 0, 0, 32, 18));  // le Plan attend une source 64 × 36
  REQUIRE(pipeline.refusals.size() == 1);
  CHECK(pipeline.refusals[0].code == RefusalCode::ShapeInvalid);
  CHECK(pipeline.worker.stats().dropped == 1);
  pipeline.push(test::make_frame(1, 40));
  CHECK(pipeline.sink.frames.size() == 1);

  struct Other final : ajp::ProgramBase {
    std::uint64_t plan_hash() const override { return 7; }
  };
  pipeline.worker.install(ajp::Install{std::make_shared<Other>()});
  CHECK(pipeline.refusals.back().code == RefusalCode::FieldType);  // seule une ProgramRequest s'installe
  pipeline.worker.flush(ajp::Flush{.epoch = 0});
  pipeline.worker.slot_free({.slot = 0, .semaphore_value = 99, .generation = 1});
  CHECK(pipeline.refusals.back().code == RefusalCode::SequenceOrder);
}

TEST_CASE("Travailleur : fin du flux, les paires en attente d'anticipation partent", "[worker]") {
  test::Pipeline pipeline;
  pipeline.install(test::with_level(test::upscale_plan(), {2, 1}, core::Policy::Fast,
                                    core::Schedule::OutputRes, core::Pacing::Smooth, "motion"));
  for (std::uint32_t id = 0; id < 4; ++id)
    pipeline.push(test::make_frame(id, static_cast<std::uint8_t>(40 + 10 * id)));
  CHECK(pipeline.sink.frames.size() == 1);  // Smooth attend 4 sources au-delà de la paire
  pipeline.worker.end_of_stream();
  pipeline.worker.pump();
  pipeline.release();
  CHECK(pipeline.sink.frames.size() == 7);
  CHECK(std::ranges::is_sorted(pts_of(pipeline.sink.frames)));
}

TEST_CASE("Travailleur : un ring d'une autre taille attend le Program qui lui va", "[worker]") {
  test::Pipeline pipeline;
  pipeline.install(test::upscale_plan());
  pipeline.push(test::make_frame(0, 40));
  pipeline.push(test::make_frame(1, 50));
  REQUIRE(pipeline.sink.frames.size() == 2);

  // render recrée le ring à la taille du Plan suivant (sans agrandissement) avant l'Install.
  core::Plan small;
  small.output_width = test::kWidth;
  small.output_height = test::kHeight;
  small.hash = core::compute_hash(small);
  pipeline.rings[2] = std::make_unique<test::HostRing>(4, test::kWidth, test::kHeight, 2);
  pipeline.worker.ring(ajp::gpu::RingGeneration{pipeline.rings[2]->slots()});
  pipeline.push(test::make_frame(2, 60));
  CHECK(pipeline.sink.frames.size() == 2);  // l'ancien Program n'a plus de ring à sa taille
  CHECK(pipeline.worker.take_consumed() == 2);

  pipeline.worker.install(ajp::Install{pipeline.request(small)});
  pipeline.worker.pump();
  pipeline.release();
  REQUIRE(pipeline.sink.frames.size() == 3);
  CHECK(pipeline.sink.frames[2].generation == 2);
  CHECK(pipeline.sink.frames[2].pts_us == 80'000);
  CHECK(pipeline.pixels[2].size() == std::size_t{4} * test::kWidth * test::kHeight);

  pipeline.worker.ring(ajp::gpu::RingGeneration{pipeline.rings[1]->slots()});  // génération passée
  REQUIRE_FALSE(pipeline.refusals.empty());
  CHECK(pipeline.refusals.back().code == RefusalCode::SequenceOrder);
}
