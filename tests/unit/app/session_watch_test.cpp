// La session avec le veilleur d'écriture (Screen::watch, written_at, wakes_on_write ; ring_watcher.h).
#include <cstdint>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/frame_latency.h"
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
  explicit Rig(SessionConfig config) : session(std::move(config), stages, screen, clock) {}
  void run_clock(std::int64_t media_us) {
    clock.value = {.media_us = media_us, .host_us = screen.now_us, .rate = 1.0, .epoch = 0, .valid = true};
  }
  void tick() { REQUIRE(session.tick(screen.now_us)); }
};

}  // namespace

TEST_CASE("Veilleur d'écriture : retard à l'heure exacte, plus de sondage de la tête", "[app][session]") {
  ajp::app::play::FrenchUi french;
  FrameLatency latency;
  Rig rig(SessionConfig{.latency = &latency});
  rig.screen.now_us = 1'000'000;
  rig.screen.unwritten = {0};
  RingFrame first = ring_frame(0, 40'000);
  first.submitted_us = 1'000'000;
  rig.session.handle(first);
  CHECK(rig.screen.watched == std::vector<std::uint32_t>{0});
  rig.tick();
  // Le veilleur l'a vue écrite à 1,012 s ; main ne la voit qu'à 1,030 s.
  rig.screen.now_us = 1'030'000;
  rig.screen.write(0);
  rig.screen.written_us = {{0, 1'012'000}};
  rig.tick();
  CHECK(latency.us() == 12'000);
  CHECK(rig.screen.written_us.empty());
  // La tête suivante pas encore écrite : sans veilleur, main sonde chaque milliseconde ; avec, il dort.
  rig.run_clock(40'000);
  rig.screen.unwritten = {1};
  rig.session.handle(ring_frame(1, 81'708));
  CHECK(rig.session.idle_wait_us(rig.screen.now_us) == 1'000);
  rig.screen.waking = true;
  CHECK(rig.session.idle_wait_us(rig.screen.now_us) == 4'000);
}

}  // namespace ajp::app::play
