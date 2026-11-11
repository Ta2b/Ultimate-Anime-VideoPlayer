// Session de lecture (chantier D04, défauts B13-01 à B13-04) : chaque cas échouait avant sa correction.
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
  std::int64_t source_us = kSourceUs;
  std::int64_t output_us = kSourceUs / 2;
  std::optional<std::string> rife() const override { return std::nullopt; }
  std::optional<std::string> memory() override { return std::nullopt; }
  std::int64_t output_period_us() const override { return output_us; }
  std::int64_t source_period_us() const override { return source_us; }
};

// Une image de sortie de RIFE ×2 : t = 1/2 entre deux sources, ou une source (t = 0).
RingFrame output_frame(std::uint32_t slot, int k) {
  RingFrame f = ring_frame(slot, k * kSourceUs / 2);
  if (k % 2 != 0) {
    f.t_num = 1;
    f.t_den = 2;
  }
  return f;
}

}  // namespace

TEST_CASE("B13-01 : image par image en interpolation, le pas montre la source suivante", "[app][session]") {
  FrenchUi french;
  Rates rates;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session(SessionConfig{.extra = &rates}, stages, screen, clock);
  for (int k = 0; k < 8; ++k) session.handle(output_frame(static_cast<std::uint32_t>(k), k));
  REQUIRE(session.tick(screen.now_us));  // amorce : la source 0
  // L'horloge court : l'image à l'écran devient l'interpolée k = 1.
  clock.value = {.media_us = 5'000, .host_us = screen.now_us, .rate = 1.0, .valid = true};
  REQUIRE(session.tick(screen.now_us));
  REQUIRE(screen.shown.back().frame);
  REQUIRE(screen.shown.back().frame->pts_us == kSourceUs / 2);
  REQUIRE(screen.shown.back().frame->t_num == 1);

  session.command(cmd::Pause{});
  clock.value.rate = 0.0;
  session.command(cmd::StepFrame{.direction = 1});
  REQUIRE(session.tick(screen.now_us));
  REQUIRE(screen.shown.back().frame);
  const RingFrame stepped = *screen.shown.back().frame;
  CHECK(stepped.t_num == 0);                   // une source, pas une image de RIFE
  CHECK(stepped.pts_us == 2 * kSourceUs / 2);  // la suivante, k = 2
  // Un pas de plus : la source d'après (k = 4), l'interpolée k = 3 passée.
  session.command(cmd::StepFrame{.direction = 1});
  REQUIRE(session.tick(screen.now_us));
  CHECK(screen.shown.back().frame->t_num == 0);
  CHECK(screen.shown.back().frame->pts_us == 4 * kSourceUs / 2);
}

TEST_CASE("B13-01 : d'une image interpolée, le seek exact vise la source suivante", "[app][session]") {
  FrenchUi french;
  Rates rates;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session(SessionConfig{.extra = &rates}, stages, screen, clock);
  session.handle(output_frame(0, 0));
  session.handle(output_frame(1, 1));  // la source suivante n'est pas arrivée : le pas passe par un seek
  REQUIRE(session.tick(screen.now_us));
  clock.value = {.media_us = 5'000, .host_us = screen.now_us, .rate = 1.0, .valid = true};
  REQUIRE(session.tick(screen.now_us));
  REQUIRE(screen.shown.back().frame->t_num == 1);
  session.command(cmd::Pause{});
  clock.value.rate = 0.0;
  stages.log.clear();
  session.command(cmd::StepFrame{.direction = 1});
  // Le milieu de la source suivante (k = 2), pas celui de la source d'après.
  CHECK(std::ranges::find(stages.log, std::format("seek {}", kSourceUs + kSourceUs / 2)) != stages.log.end());
}

TEST_CASE("B13-02 : FIFO pleine (acquisitions bloquées), le délai visible compte l'image en file",
          "[app][session]") {
  FrenchUi french;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  screen.period_us = 6'944;
  Session session(SessionConfig{.display_period_us = 6'944, .swapchain_depth = 2}, stages, screen, clock);
  session.handle(ring_frame(0, 0));
  REQUIRE(session.tick(screen.now_us));
  screen.acquire_us = 6'000;  // chaque acquisition attend le vsync : la file de l'écran est pleine
  screen.queue_wait_us = 0;   // swap_buffers ne bloque pas
  for (int i = 0; i < 60; ++i) {
    session.request_present();
    REQUIRE(session.tick(screen.now_us));
  }
  // (depth − 1) périodes en file, puis une demi-période jusqu'au vsync.
  CHECK(session.visible_delay_us() >= 6'944 + 6'944 / 2);
  // Une composition lourde (swap_buffers lent) sans attente d'acquisition : la file n'est pas pleine.
  screen.acquire_us = 0;
  screen.queue_wait_us = 6'000;
  for (int i = 0; i < 60; ++i) {
    session.request_present();
    REQUIRE(session.tick(screen.now_us));
  }
  CHECK(session.visible_delay_us() < 6'944);
}

TEST_CASE("B13-03 : le délai visible ne bascule pas à chaque présentation autour du seuil",
          "[app][session]") {
  FrenchUi french;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  screen.period_us = 6'944;
  Session session(SessionConfig{.display_period_us = 6'944, .swapchain_depth = 2}, stages, screen, clock);
  session.handle(ring_frame(0, 0));
  REQUIRE(session.tick(screen.now_us));
  int flips = 0;
  std::int64_t last = session.visible_delay_us();
  for (int i = 0; i < 400; ++i) {
    // Une acquisition sur deux attend l'écran (seuil : un quart de période, 1,736 ms).
    screen.acquire_us = i % 2 == 0 ? 2'000 : 1'000;
    session.request_present();
    REQUIRE(session.tick(screen.now_us));
    const std::int64_t now = session.visible_delay_us();
    if (now != last) ++flips;
    last = now;
  }
  CHECK(flips <= 2);
}

TEST_CASE("B13-04 : les trous de sous-titres se comptent par présentation", "[app][session]") {
  FrenchUi french;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session(SessionConfig{.subtitles = true, .video_width = 1920, .video_height = 1080}, stages, screen,
                  clock);
  session.handle(ring_frame(0, 0));
  session.handle(subs::SubtitleFrame{.pts_us = 500'000, .epoch = 0});  // réponse d'une image à venir
  REQUIRE(session.tick(screen.now_us));
  const std::size_t presents = screen.shown.size();
  for (int i = 0; i < 100; ++i) {
    screen.now_us += 1'000;
    REQUIRE(session.tick(screen.now_us));  // rien de neuf, rien de présenté
  }
  CHECK(screen.shown.size() == presents);
  CHECK(session.stats().subtitle_gaps <= static_cast<std::int64_t>(presents));
}

}  // namespace ajp::app::play
