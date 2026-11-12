// La session et l'horloge : reprise après la pause, fin de lecture, pas en arrière depuis une image
// interpolée, vblank d'apparition (audit 3, C13-02, C13-03, C13-04, C13-07). Chaque cas échouait avant sa
// correction.
#include <algorithm>
#include <cstdint>
#include <format>
#include <optional>
#include <string>

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

constexpr std::int64_t kSourceUs = 41'708;  // 23,976 i/s

struct Rates final : StatsSource {
  std::optional<std::string> rife() const override { return std::nullopt; }
  std::optional<std::string> memory() override { return std::nullopt; }
  std::int64_t output_period_us() const override { return kSourceUs / 2; }
  std::int64_t source_period_us() const override { return kSourceUs; }
};

}  // namespace

TEST_CASE("C13-02 : la reprise relance l'horloge où elle s'est figée, pas au pts de l'image",
          "[app][session]") {
  FrenchUi french;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session(SessionConfig{.display_period_us = 16'667}, stages, screen, clock);
  for (std::uint32_t k = 0; k < 16; ++k) session.handle(ring_frame(k, k * kSourceUs));
  REQUIRE(session.tick(screen.now_us));  // amorce
  clock.value = {.media_us = 400'000, .host_us = screen.now_us, .rate = 1.0, .valid = true};
  REQUIRE(session.tick(screen.now_us));
  const std::int64_t shown = screen.shown.back().frame->pts_us;
  REQUIRE(shown != 400'000);  // l'image montrée est choisie en avance sur l'horloge
  session.command(cmd::Pause{});
  clock.value = {.media_us = 400'000, .host_us = screen.now_us, .rate = 0.0, .valid = true};  // figée
  session.command(cmd::Play{});
  CHECK(stages.log.back() == "play 0 400000");
}

TEST_CASE("C13-03 : la fin attend que l'horloge passe la durée du fichier (son plus long, carton tenu)",
          "[app][session]") {
  FrenchUi french;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session(SessionConfig{.duration_us = 5'000'000, .display_period_us = 16'667}, stages, screen,
                  clock);
  session.handle(ring_frame(0, 0));
  session.handle(ring_frame(1, kSourceUs));  // la dernière image, tenue jusqu'à 5 s (fin du son)
  session.handle(StreamEnded{.epoch = 0});
  std::int64_t media = 0;
  while (!session.finished() && media < 6'000'000) {
    clock.value = {.media_us = media, .host_us = screen.now_us, .rate = 1.0, .valid = true};
    (void)session.tick(screen.now_us);
    screen.now_us += 10'000;
    media += 10'000;
  }
  CHECK(media >= 5'000'000);
  CHECK(media < 5'100'000);
}

TEST_CASE("C13-03 : horloge arrêtée, la fin tombe après la grâce d'une seconde", "[app][session]") {
  FrenchUi french;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session(SessionConfig{.duration_us = 60'000'000, .display_period_us = 16'667}, stages, screen,
                  clock);
  session.handle(ring_frame(0, 0));
  session.handle(StreamEnded{.epoch = 0});
  clock.value = {.media_us = 0, .host_us = 0, .rate = 1.0, .valid = true};
  (void)session.tick(screen.now_us);
  clock.value.rate = 0.0;  // l'horloge ne court plus (audio fini, rien ne mène)
  while (!session.finished() && screen.now_us < 3'000'000) {
    screen.now_us += 10'000;
    (void)session.tick(screen.now_us);
  }
  CHECK(session.finished());
  CHECK(screen.now_us < 1'100'000);
}

TEST_CASE("C13-07 : image précédente depuis une interpolée, la source de gauche", "[app][session]") {
  FrenchUi french;
  Rates rates;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session(SessionConfig{.extra = &rates}, stages, screen, clock);
  for (int k = 0; k < 6; ++k) {
    RingFrame f = ring_frame(static_cast<std::uint32_t>(k), k * kSourceUs / 2);
    if (k % 2 != 0) {
      f.t_num = 1;
      f.t_den = 2;
    }
    session.handle(f);
  }
  REQUIRE(session.tick(screen.now_us));
  // L'horloge court jusqu'à l'interpolée k = 3, entre les sources 1 (k = 2) et 2 (k = 4).
  clock.value = {.media_us = 3 * kSourceUs / 2 - 1'000, .host_us = screen.now_us, .rate = 1.0, .valid = true};
  REQUIRE(session.tick(screen.now_us));
  REQUIRE(screen.shown.back().frame->t_num == 1);
  REQUIRE(screen.shown.back().frame->pts_us == 3 * kSourceUs / 2);
  session.command(cmd::Pause{});
  clock.value.rate = 0.0;
  stages.log.clear();
  session.command(cmd::StepFrame{.direction = -1});
  // Le milieu de la source 1, montrée juste avant, et non celui de la source 0.
  CHECK(std::ranges::find(stages.log, std::format("seek {}", kSourceUs + kSourceUs / 2)) != stages.log.end());
}

TEST_CASE("C13-04 : par numéro de vblank, l'image paraît à son vblank, pas un avant",
          "[app][session][present]") {
  FrenchUi french;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  constexpr std::int64_t kPeriod = 16'667;
  Session session(SessionConfig{.display_period_us = kPeriod, .display_hz = 60.0, .vsync_index = true},
                  stages, screen, clock);
  screen.grid = core::VblankGrid{.phase_us = 0.0, .period_us = static_cast<double>(kPeriod)};
  // Horloge maître = horloge hôte. L'image 1 est due au vblank 10 ; l'image 0 la précède.
  session.handle(ring_frame(0, 9 * kPeriod));
  session.handle(ring_frame(1, 10 * kPeriod));
  REQUIRE(session.tick(screen.now_us));  // amorce : l'image 0
  // Juste avant le vblank 9 : une présentation faite maintenant paraît au vblank 9. L'image du vblank 10 ne
  // doit pas encore partir ; l'ancien calcul comptait deux fois l'attente moyenne du vsync et l'envoyait.
  screen.now_us = 9 * kPeriod - 2'000;
  clock.value = {.media_us = screen.now_us, .host_us = screen.now_us, .rate = 1.0, .valid = true};
  (void)session.tick(screen.now_us);
  REQUIRE(screen.shown.back().frame);
  CHECK(screen.shown.back().frame->slot == 0);
  // Juste après le vblank 9 : la présentation paraît au vblank 10, l'image 1 part.
  screen.now_us = 9 * kPeriod + 2'000;
  clock.value = {.media_us = screen.now_us, .host_us = screen.now_us, .rate = 1.0, .valid = true};
  session.request_present();
  (void)session.tick(screen.now_us);
  REQUIRE(screen.shown.back().frame);
  CHECK(screen.shown.back().frame->slot == 1);
}

}  // namespace ajp::app::play
