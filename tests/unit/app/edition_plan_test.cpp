// A14-03 : une édition ordonnée de Matroska jouée dans son ordre, par sauts d'une partie à la suivante.
#include "app/play/edition_plan.h"

#include <string>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/session.h"
#include "doubles.h"

namespace ajp::app::play {
namespace {

using test::FakeClock;
using test::FakeScreen;
using test::FakeStages;
using test::ring_frame;

// Un récapitulatif à sauter : l'édition joue 2 à 3 s, puis 0 à 1 s, puis 5 à 6 s et 6 à 8 s qui la prolonge.
const std::vector<PlayPart> kParts{{.start_us = 2'000'000, .end_us = 3'000'000},
                                   {.start_us = 0, .end_us = 1'000'000},
                                   {.start_us = 5'000'000, .end_us = 6'000'000},
                                   {.start_us = 6'000'000, .end_us = 8'000'000}};

std::int64_t jump_of(const EditionPlan::Step& step) {
  const auto* jump = std::get_if<EditionPlan::Jump>(&step);
  REQUIRE(jump != nullptr);
  return jump->to_us;
}

bool nothing(const EditionPlan::Step& step) {
  return std::holds_alternative<std::monostate>(step);
}

}  // namespace

TEST_CASE("Édition ordonnée : les parties dans l'ordre de l'édition, puis la fin", "[app][chapters]") {
  EditionPlan plan(kParts);
  REQUIRE(plan.active());
  CHECK(jump_of(plan.shown(0, 40'000)) == 2'000'000);  // le fichier commence avant la première partie
  CHECK(nothing(plan.shown(0, 40'000)));               // une fois, en attendant l'image du saut
  CHECK(nothing(plan.shown(2'000'000, 40'000)));
  CHECK(nothing(plan.shown(2'920'000, 40'000)));
  CHECK(jump_of(plan.shown(2'960'000, 40'000)) == 0);  // l'image suivante sortirait de la partie
  CHECK(nothing(plan.shown(0, 40'000)));
  CHECK(jump_of(plan.shown(960'000, 40'000)) == 5'000'000);
  CHECK(nothing(plan.shown(5'000'000, 40'000)));
  CHECK(nothing(plan.shown(5'960'000, 40'000)));  // 6 à 8 s prolonge 5 à 6 s : sans seek
  CHECK(nothing(plan.shown(6'000'000, 40'000)));
  CHECK(std::holds_alternative<EditionPlan::End>(plan.shown(7'960'000, 40'000)));
}

TEST_CASE("Édition ordonnée : un seek va dans la partie qui contient sa cible, ou à la suivante",
          "[app][chapters]") {
  EditionPlan plan(kParts);
  CHECK(plan.seek(500'000) == 500'000);  // dans la deuxième partie
  CHECK(nothing(plan.shown(500'000, 40'000)));
  CHECK(jump_of(plan.shown(960'000, 40'000)) == 5'000'000);
  CHECK(plan.seek(4'000'000) == 5'000'000);       // hors de l'édition : la partie suivante dans le fichier
  CHECK_FALSE(plan.seek(9'000'000).has_value());  // après la dernière : fin
  CHECK(EditionPlan().seek(9'000'000) == 9'000'000);  // sans édition ordonnée : le fichier
  CHECK(nothing(EditionPlan().shown(9'000'000, 40'000)));
}

TEST_CASE("Édition ordonnée : une partie jusqu'à la fin du fichier finit avec lui", "[app][chapters]") {
  EditionPlan plan({{.start_us = 1'000'000, .end_us = 2'000'000}, {.start_us = 0}});
  CHECK(jump_of(plan.shown(0, 0)) == 1'000'000);
  CHECK(nothing(plan.shown(1'000'000, 0)));
  CHECK(jump_of(plan.shown(2'000'000, 0)) == 0);
  CHECK(nothing(plan.shown(0, 0)));
  CHECK(nothing(plan.shown(60'000'000, 0)));  // la fin du flux finit la lecture
}

TEST_CASE("Édition ordonnée : la session saute au début de l'édition et finit après la dernière partie",
          "[app][chapters][session]") {
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session(SessionConfig{.duration_us = 10'000'000, .parts = kParts}, stages, screen, clock);
  session.begin(0);
  session.handle(ring_frame(0, 0));
  REQUIRE(session.tick(screen.now_us));
  CHECK(std::ranges::contains(stages.log, std::string("seek 2000000")));
  CHECK(session.epoch() == 1);
  session.handle(ring_frame(1, 2'000'000, 1));
  REQUIRE(session.tick(screen.now_us));
  REQUIRE(screen.shown.back().frame);
  CHECK(screen.shown.back().frame->pts_us == 2'000'000);
  CHECK(session.epoch() == 1);  // dans la partie : pas de saut
  CHECK_FALSE(session.finished());
  session.command(cmd::Seek{.target_us = 9'000'000});  // après la dernière partie
  CHECK(session.finished());
}

}  // namespace ajp::app::play
