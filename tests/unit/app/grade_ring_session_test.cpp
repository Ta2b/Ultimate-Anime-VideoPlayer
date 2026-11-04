// Le grade de l'anneau dans la Session (chantier 2.10) : en pause, l'image montrée gradée par une version
// dépassée (Screen::stale_grade) est récrite par un seek exact ; en lecture, rien.
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/session.h"
#include "doubles.h"
#include "french_ui.h"

namespace ajp::app::play {
namespace {

using test::FakeClock;
using test::FakeScreen;
using test::FakeStages;
using test::ring_frame;

struct Rig {
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session;

  explicit Rig(SessionConfig config = {}) : session(std::move(config), stages, screen, clock) {}
  void tick() {
    auto shown = session.tick(screen.now_us);
    REQUIRE(shown);
  }
  std::int64_t shown_pts() const {
    REQUIRE(!screen.shown.empty());
    REQUIRE(screen.shown.back().frame);
    return screen.shown.back().frame->pts_us;
  }
};

TEST_CASE(
    "Grade de l'anneau : en pause, l'image gradée par une version dépassée est récrite par un seek exact",
    "[app][session][grade_ring]") {
  ajp::app::play::FrenchUi french;
  Rig rig(SessionConfig{.duration_us = 60'000'000});
  RingFrame graded = ring_frame(0, 1'000'000);
  graded.grade = 3;
  rig.session.handle(RingFrame{graded});
  rig.tick();
  rig.session.command(cmd::Pause{});
  rig.stages.log.clear();
  rig.screen.now_us += 1'000'000;
  rig.tick();
  CHECK(rig.stages.log.empty());  // la version est la bonne : rien

  rig.screen.stale_versions = {3};
  rig.screen.now_us += 1'000'000;
  rig.tick();
  CHECK(rig.stages.log == std::vector<std::string>{"pause", "flush 1", "seek 1000000"});
  CHECK(rig.session.paused());
  // Le seek attend sa première image : pas d'autre écriture d'ici là.
  rig.screen.now_us += 1'000'000;
  rig.tick();
  CHECK(rig.stages.log.size() == 3);
  RingFrame again = ring_frame(1, 1'000'000, 1);
  again.grade = 4;
  rig.session.handle(RingFrame{again});
  rig.screen.now_us += 1'000'000;
  rig.tick();
  CHECK(rig.shown_pts() == 1'000'000);
  CHECK(rig.stages.log.size() == 3);  // la nouvelle version n'est pas dépassée
}

TEST_CASE("Grade de l'anneau : en lecture, une version dépassée ne demande rien",
          "[app][session][grade_ring]") {
  Rig rig(SessionConfig{.duration_us = 60'000'000});
  rig.screen.stale_versions = {3};
  RingFrame graded = ring_frame(0, 1'000'000);
  graded.grade = 3;
  rig.session.handle(RingFrame{graded});
  rig.tick();
  rig.stages.log.clear();
  rig.screen.now_us += 1'000'000;
  rig.tick();
  for (const std::string& entry : rig.stages.log) CHECK_FALSE(entry.starts_with("seek"));
}

}  // namespace
}  // namespace ajp::app::play
