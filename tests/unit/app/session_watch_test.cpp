// La session et le retour de l'écran : veilleur d'écriture (Screen::watch, written_at, wakes_on_write ;
// ring_watcher.h) et fréquence mesurée (Screen::vblank_grid).
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
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

struct OutputRate final : StatsSource {
  double fps = 164.9;
  std::optional<std::string> rife() const override { return std::nullopt; }
  std::optional<std::string> memory() const override { return std::nullopt; }
  double output_fps() const override { return fps; }
};

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

TEST_CASE("Synchro sur l'écran : la fréquence mesurée remplace l'annoncée", "[app][session][present]") {
  ajp::app::play::FrenchUi french;
  OutputRate rate;
  Rig rig(
      SessionConfig{.display_period_us = 6061, .display_hz = 165.0, .display_sync = true, .extra = &rate});
  rig.tick();
  REQUIRE(rig.stages.speeds.size() == 1);
  CHECK(rig.stages.speeds[0] == Catch::Approx(165.0 / 164.9).epsilon(1e-9));
  // Le retour de présentation mesure 164,78 Hz : la vitesse se cale sur l'écran réel.
  rig.screen.grid = core::VblankGrid{.phase_us = 0.0, .period_us = 1e6 / 164.78};
  rig.session.set_display(6061, 165.0);
  REQUIRE(rig.stages.speeds.size() == 2);
  CHECK(rig.stages.speeds[1] == Catch::Approx(164.78 / 164.9).epsilon(1e-6));
  // Un relevé à 0,001 Hz près : rien de redemandé.
  rig.screen.grid->period_us = 1e6 / 164.781;
  rig.session.set_display(6061, 165.0);
  CHECK(rig.stages.speeds.size() == 2);
  // Une grille d'un autre mode (plus de 2 % d'écart) ne compte pas.
  rig.screen.grid->period_us = 1e6 / 120.0;
  rig.session.set_display(6061, 165.0);
  REQUIRE(rig.stages.speeds.size() == 3);
  CHECK(rig.stages.speeds[2] == Catch::Approx(165.0 / 164.9).epsilon(1e-9));
}

}  // namespace ajp::app::play
