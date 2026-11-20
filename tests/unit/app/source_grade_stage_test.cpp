// Le grade avant le warp (source_grade.h) dans l'étage gpu, sur le banc sans GPU (upscale_rig.h) : un grade
// par SPAN, aucun pour un dessin tenu, les images marquées gradées pour l'écran (RingFrame::graded), et les
// tuiles de SPAN (upscale_tiles.h) qui partent de la sortie d'avant le grade, ou sur place sans lui.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "app/play/links.h"
#include "app/play/source_grade.h"
#include "core/clock.h"
#include "kernels/spec/span_tiles.h"
#include "upscale_rig.h"

namespace kirari::app::play {
using namespace upscale_test;
namespace {

enum class Answer { Graded, Nothing, Refused };

TEST_CASE("Grade avant le warp : l'instantané retrouve le masque de l'image montrée par son pts", "[app]") {
  SourceGradeShare share;
  CHECK_FALSE(share.effects(1'000).on);  // rien de noté
  for (std::int64_t i = 0; i < 8; ++i) {
    share.note_effects(EffectSummary{.on = true, .sure = 0.1 * static_cast<double>(i), .pts_us = i * 41'708});
  }
  // Le grade va devant l'écran : l'image montrée retrouve le sien, pas celui de la dernière source.
  CHECK(share.effects(3 * 41'708).pts_us == 3 * 41'708);
  CHECK(share.effects(3 * 41'708).sure == 0.1 * 3.0);
  // Une image inconnue (interpolée, trop ancienne) : le plus récent, que son pts trahit.
  CHECK(share.effects(5).pts_us == 7 * 41'708);
  CHECK(share.effects(kNoPts).pts_us == 7 * 41'708);
}

class CountingGrader final : public SourceGrader {
 public:
  explicit CountingGrader(Answer answer) : answer_(answer) {}
  std::expected<bool, Refusal> grade(gpu::KernelOps& /*ops*/, const gpu::Tensor& rgb, bool /*debanded*/,
                                     const GradedSource& source) override {
    ++calls;
    grade_pts.push_back(source.pts_us);
    graded.push_back(rgb.handle);  // lu par le test une fois l'étage arrêté
    if (answer_ == Answer::Refused)
      return std::unexpected(Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "essai"});
    return answer_ == Answer::Graded;
  }
  std::atomic<int> calls{0};
  std::vector<std::int64_t> grade_pts;
  std::vector<std::uintptr_t> graded;  // les tenseurs gradés sur place

 private:
  Answer answer_;
};

// RIFE mouvement ×2, warp après SPAN, six sources dont 3 tient le dessin de 2 : SPAN sur 1, 2, 4, 5 et 6.
// Sans SPAN (span faux) : les slots à la taille de travail.
std::vector<RingFrame> play(Rig& rig, std::shared_ptr<SourceGrader> grader, bool warp_after_span = true,
                            bool span = true) {
  rig.backend.score(SourceId{2}, SourceId{3}, 0.0, 10);
  UpscaleSetup s = rig.setup(true);
  if (!span) {
    s.engine = nullptr;
    s.slots = rig.slots(4, kWidth, kHeight);
  }
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.warp_after_span = warp_after_span;
  s.grader = std::move(grader);
  rig.start(std::move(s));
  rig.feed(1, 6);
  REQUIRE(rig.collect(10));  // six sources, une image entre chaque paire hors du dessin tenu
  rig.stage.reset();
  return rig.sink.frames();
}

// Une horloge sans valeur : la lecture (les tuiles ne servent qu'en lecture), rien n'est en retard.
struct IdleClock final : MasterClock {
  ClockSnapshot snapshot() const override { return {}; }
};

// L'écart de chaque bloc 16 × 16 de l'image de travail de 320 × 180 (20 × 12) : un dessin qui change dans le
// bloc du coin haut gauche, rien ailleurs ; chaque paire se calcule alors par tuiles.
std::vector<std::uint32_t> corner_blocks() {
  std::vector<std::uint32_t> blocks(20 * 12, 0);
  blocks[0] = 1000;
  return blocks;
}

// La grille du SPAN natif sur l'image de travail de 320 × 180 : tuiles de 64 × 16.
constexpr kernels::SpanTileGrid kTileGrid{.tile_w = 64,
                                          .tile_h = 16,
                                          .cols = 5,
                                          .rows = 12,
                                          .reach_cols = 1,
                                          .reach_rows = 1,
                                          .shift_step = 2,
                                          .scale = 2};

}  // namespace

TEST_CASE("Étage gpu : grade avant le warp, un grade par SPAN et toutes les images gradées", "[app][stage]") {
  Rig rig;
  auto grader = std::make_shared<CountingGrader>(Answer::Graded);
  const std::vector<RingFrame> frames = play(rig, grader);
  CHECK(grader->calls == static_cast<int>(rig.backend.count("engine span")));
  CHECK(grader->calls == 5);  // le dessin tenu reprend le tampon gradé de 2
  for (const RingFrame& frame : frames) CHECK(frame.graded);
}

TEST_CASE("Étage gpu : Smooth grade le futur dessin à son propre pts", "[app][stage][smooth]") {
  const std::uint32_t hold = GENERATE(2U, 4U);
  Rig rig;
  rig.live->set_keep_poses(false);
  for (std::uint32_t k = 1; k < hold; ++k) rig.backend.score(SourceId{k}, SourceId{k + 1}, 0.0, 10);
  for (std::uint32_t k = 3; k <= hold; ++k) rig.backend.score(SourceId{2}, SourceId{k}, 0.0, 10);
  auto grader = std::make_shared<CountingGrader>(Answer::Graded);
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.warp_after_span = true;
  s.smooth_drawings = true;
  s.grader = grader;
  rig.start(std::move(s));
  rig.feed(1, hold + 2);
  REQUIRE(rig.stage->post(StreamEnd{.epoch = 0}));
  REQUIRE(rig.collect(1 + 2 * (hold - 1) + 2 + 2));
  rig.stage.reset();
  CHECK(grader->grade_pts ==
        std::vector<std::int64_t>{kPeriodUs, (hold + 1) * kPeriodUs, (hold + 2) * kPeriodUs});
}

TEST_CASE("Étage gpu : SPAN de B sans slot, le grade et les passes RIFE gardent leur place", "[app][stage]") {
  const bool ahead = GENERATE(false, true);
  INFO("SPAN de B en avance : " << ahead);
  Rig rig;
  rig.live->set_keep_poses(false);
  rig.backend.motion_max(40.0F);
  auto grader = std::make_shared<CountingGrader>(Answer::Graded);
  UpscaleSetup s = rig.setup(true, Factor{.num = 4, .den = 1}, 1);
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.rife_at_slot = true;
  s.span_b_ahead = ahead;
  s.warp_after_span = true;
  s.grader = grader;
  rig.start(std::move(s));

  rig.feed(1, 2);
  REQUIRE(rig.collect(1, false));  // le seul slot est pris par A
  REQUIRE(logged(rig, "rife_input"));
  if (ahead) {
    REQUIRE(logged(rig, "engine span", 2));
  } else {
    CHECK(rig.backend.count("engine span") == 1);
  }
  CHECK(rig.backend.count("engine rife_motion") == 0);
  CHECK(grader->calls == 1);  // B sera gradée à l'endroit habituel
  REQUIRE(rig.collect(5));    // rend le slot de A, laisse sortir B et les trois instants intermédiaires
  rig.stage.reset();
  CHECK(pts_of(rig.sink.frames()) == std::vector<std::int64_t>{40'000, 50'000, 60'000, 70'000, 80'000});
  using Pair = std::pair<std::uint32_t, std::uint32_t>;
  CHECK(rig.backend.warp_pairs() == std::vector<Pair>{{1, 2}, {1, 2}, {1, 2}});
  CHECK(rig.backend.count("engine span") == 2);
  CHECK(rig.backend.count("engine rife_motion") == 3);
  CHECK(rig.backend.rife_ts() == std::vector<float>{0.5F, 0.25F, 0.75F});
  CHECK(grader->calls == 2);
  for (const RingFrame& frame : rig.sink.frames()) CHECK(frame.graded);
}

TEST_CASE("Étage gpu : un refus du SPAN de B sans slot est attribué à SPAN", "[app][stage]") {
  using namespace std::chrono_literals;
  Rig rig;
  UpscaleSetup s = rig.setup(true, Factor{.num = 2, .den = 1}, 1);
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.rife_at_slot = true;
  s.span_b_ahead = true;
  s.warp_after_span = true;
  rig.start(std::move(s));
  rig.feed(1, 1);
  REQUIRE(rig.collect(1, false));
  rig.backend.refuse_engine("span");
  rig.feed(2, 2);
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (rig.reported().empty() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(1ms);
  rig.stage.reset();
  const auto reported = rig.reported();
  REQUIRE(reported.size() == 1);
  CHECK(blamed_stage_key(reported[0].key) == kSpanRefusalKey);
  CHECK(rig.sink.frames().size() == 1);  // aucun slot de plus n'a été promis
  CHECK(rig.backend.count("engine rife_motion") == 0);
}

TEST_CASE("Étage gpu : grade avant le warp, rien à grader, l'écran garde le grade", "[app][stage]") {
  Rig rig;
  auto grader = std::make_shared<CountingGrader>(Answer::Nothing);
  const std::vector<RingFrame> frames = play(rig, grader);
  CHECK(grader->calls == 5);
  for (const RingFrame& frame : frames) CHECK_FALSE(frame.graded);
}

TEST_CASE("Étage gpu : grade avant le warp refusé, l'étage cesse de grader et la lecture continue",
          "[app][stage]") {
  Rig rig;
  auto grader = std::make_shared<CountingGrader>(Answer::Refused);
  const std::vector<RingFrame> frames = play(rig, grader);
  CHECK(grader->calls == 1);
  CHECK(rig.reported().empty());
  for (const RingFrame& frame : frames) CHECK_FALSE(frame.graded);
}

TEST_CASE("Étage gpu : grade avant le warp sans SPAN, chaque source copiée puis gradée", "[app][stage]") {
  Rig rig;
  auto grader = std::make_shared<CountingGrader>(Answer::Graded);
  const std::vector<RingFrame> frames = play(rig, grader, false, false);
  CHECK(grader->calls == 6);  // une copie par source, dessin tenu compris : RIFE lit les sources, pas elles
  for (const RingFrame& frame : frames) CHECK(frame.graded);
}

TEST_CASE("Étage gpu : grade avant le warp sans warp après SPAN, l'écran grade", "[app][stage]") {
  Rig rig;
  auto grader = std::make_shared<CountingGrader>(Answer::Graded);
  const std::vector<RingFrame> frames = play(rig, grader, false);
  for (const RingFrame& frame : frames) CHECK_FALSE(frame.graded);
}

// L'essai du PC du 19 novembre : SPAN natif en 4K, warp après SPAN, grade avant le warp ; aucune image par
// tuiles, le grade réécrivant sur place la sortie d'avant. Elle est maintenant copiée avant le grade, et les
// tuiles partent de la copie : dès la troisième source (la première n'a pas d'avant, la deuxième suit un
// grade sans copie), jamais d'un tampon gradé. La plaque de décor allumée ne les coupe plus (elle réécrit la
// sortie comme le grade, docs/calques.md) ; sans carte, elle ne tourne pas.
TEST_CASE("Étage gpu : grade avant le warp, les tuiles de SPAN partent de la sortie d'avant le grade",
          "[app][stage]") {
  const bool plate = GENERATE(false, true);
  INFO("plaque " << plate);
  Rig rig;
  IdleClock clock;
  rig.span.offer_tiles(kTileGrid);
  rig.backend.offer_blocks(corner_blocks(), 20);
  auto grader = std::make_shared<CountingGrader>(Answer::Graded);
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.warp_after_span = true;
  s.grader = grader;
  s.span_reuse.tiles = true;
  s.clock = &clock;
  s.layers = {.plate = plate};
  rig.start(std::move(s), &clock);
  rig.feed(1, 6);
  REQUIRE(rig.collect(11));  // six sources et une image entre chaque paire
  rig.stage.reset();
  CHECK(grader->calls == 6);
  // Sources 3 à 6, chacune avec la mesure de sa paire.
  const std::vector<std::uintptr_t>& from = rig.span.tiles_from();
  CHECK(from.size() == 4);
  for (const std::uintptr_t handle : from) {
    CHECK(handle != 0U);
    CHECK(std::ranges::find(grader->graded, handle) == grader->graded.end());
  }
}

// SPAN seul, sans interpolation : l'écran grade, et les tuiles se font sur place, dans la sortie qui tient
// déjà celle d'avant (aucun tenseur d'avant donné au moteur).
TEST_CASE("Étage gpu : SPAN seul, les tuiles se font sur place", "[app][stage]") {
  Rig rig;
  IdleClock clock;
  rig.span.offer_tiles(kTileGrid);
  rig.backend.offer_blocks(corner_blocks(), 20);
  UpscaleSetup s = rig.setup(false);
  s.grader = std::make_shared<CountingGrader>(Answer::Graded);
  s.span_reuse.tiles = true;
  s.clock = &clock;
  rig.start(std::move(s), &clock);
  rig.feed(1, 6);
  REQUIRE(rig.collect(6));
  rig.stage.reset();
  const std::vector<std::uintptr_t>& from = rig.span.tiles_from();
  CHECK(from.size() == 5);  // sources 2 à 6
  for (const std::uintptr_t handle : from) CHECK(handle == 0U);
}

}  // namespace kirari::app::play
