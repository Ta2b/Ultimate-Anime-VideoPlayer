// Le grade avant le warp (source_grade.h) dans l'étage gpu, sur le banc sans GPU (upscale_rig.h) : un grade
// par SPAN, aucun pour un dessin tenu, les images marquées gradées pour l'écran (RingFrame::graded), et les
// tuiles de SPAN (upscale_tiles.h) qui partent de la sortie d'avant le grade, ou sur place sans lui.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/source_grade.h"
#include "core/clock.h"
#include "kernels/spec/span_tiles.h"
#include "upscale_rig.h"

namespace kirari::app::play {
using namespace upscale_test;
namespace {

enum class Answer { Graded, Nothing, Refused };

class CountingGrader final : public SourceGrader {
 public:
  explicit CountingGrader(Answer answer) : answer_(answer) {}
  std::expected<bool, Refusal> grade(gpu::KernelOps& /*ops*/, const gpu::Tensor& rgb, bool /*debanded*/,
                                     const GradedSource& /*source*/) override {
    ++calls;
    graded.push_back(rgb.handle);  // lu par le test une fois l'étage arrêté
    if (answer_ == Answer::Refused)
      return std::unexpected(Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "essai"});
    return answer_ == Answer::Graded;
  }
  std::atomic<int> calls{0};
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

// Passage 9 du PC : SPAN natif en 4K, warp après SPAN, grade avant le warp, carte de la scène ; aucune image
// par tuiles, le grade réécrivant sur place la sortie d'avant. Elle est maintenant copiée avant le grade, et
// les tuiles partent de la copie : dès la troisième source (la première n'a pas d'avant, la deuxième suit un
// grade sans copie), jamais d'un tampon gradé.
TEST_CASE("Étage gpu : grade avant le warp, les tuiles de SPAN partent de la sortie d'avant le grade",
          "[app][stage]") {
  Rig rig;
  IdleClock clock;
  rig.span.offer_tiles(kTileGrid);
  rig.backend.offer_still_scene(true);
  auto grader = std::make_shared<CountingGrader>(Answer::Graded);
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.warp_after_span = true;
  s.grader = grader;
  s.scene_map = true;
  s.clock = &clock;
  rig.start(std::move(s), &clock);
  rig.feed(1, 6);
  REQUIRE(rig.collect(11));  // six sources et une image entre chaque paire
  rig.stage.reset();
  CHECK(grader->calls == 6);
  // Sources 3 à 6 quand chaque carte arrive à l'heure ; une source poussée après l'arrivée de sa précédente
  // (le banc ne tient pas l'étage) n'a pas de carte et se calcule entière.
  const std::vector<std::uintptr_t>& from = rig.span.tiles_from();
  REQUIRE_FALSE(from.empty());
  CHECK(from.size() <= 4);
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
  rig.backend.offer_still_scene(true);
  UpscaleSetup s = rig.setup(false);
  s.grader = std::make_shared<CountingGrader>(Answer::Graded);
  s.scene_map = true;
  s.clock = &clock;
  rig.start(std::move(s), &clock);
  rig.feed(1, 6);
  REQUIRE(rig.collect(6));
  rig.stage.reset();
  const std::vector<std::uintptr_t>& from = rig.span.tiles_from();
  REQUIRE_FALSE(from.empty());
  for (const std::uintptr_t handle : from) CHECK(handle == 0U);
}

}  // namespace kirari::app::play
