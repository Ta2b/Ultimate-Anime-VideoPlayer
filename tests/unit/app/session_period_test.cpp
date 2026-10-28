// Période de l'écran mesurée (SDL ne la donne pas) : la présentation est à la demande, les
// intervalles mesurés suivent donc la cadence des images et non celle de l'écran.
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/session.h"
#include "doubles.h"

namespace ajp::app::play {
namespace {

using test::ring_frame;

TEST_CASE("Période mesurée : une image à 25 i/s sur 200 Hz part à son heure, pas une image trop tôt",
          "[app][session]") {
  test::FakeStages stages;
  test::FakeScreen screen;
  test::FakeClock clock;
  screen.period_us = 5'000;  // chaque présentation avance l'écran de 5 ms
  Session session(SessionConfig{}, stages, screen, clock);
  session.handle(ring_frame(0, 0));
  REQUIRE(session.tick(screen.now_us));  // amorce : 0 à l'écran
  const std::int64_t start_us = screen.now_us;
  clock.value = {.media_us = 0, .host_us = start_us, .rate = 1.0, .valid = true};

  constexpr std::int64_t kFrameUs = 40'000;
  std::int64_t next_pts = kFrameUs;
  std::int64_t last_pts = 0;
  std::vector<std::int64_t> early_us;     // avance de chaque image neuve sur l'horloge à l'écran
  std::vector<std::int64_t> interval_us;  // entre deux images neuves, à l'écran
  std::int64_t last_shown_us = screen.now_us;
  while (last_pts < 40 * kFrameUs) {
    const std::int64_t media_us = screen.now_us - start_us;
    // gpu garde trois images d'avance, comme le ring.
    while (next_pts <= media_us + 3 * kFrameUs) {
      session.handle(ring_frame(static_cast<std::uint32_t>(next_pts / kFrameUs % 8), next_pts));
      next_pts += kFrameUs;
    }
    const std::size_t before = screen.shown.size();
    REQUIRE(session.tick(screen.now_us));
    if (screen.shown.size() == before) {
      screen.now_us += 1'000;  // rien de présenté : la boucle repasse une milliseconde plus tard
      continue;
    }
    REQUIRE(screen.shown.back().frame);
    const std::int64_t pts = screen.shown.back().frame->pts_us;
    if (pts == last_pts) continue;
    early_us.push_back(pts - (screen.now_us - start_us));
    interval_us.push_back(screen.now_us - last_shown_us);
    last_shown_us = screen.now_us;
    last_pts = pts;
  }
  REQUIRE(early_us.size() >= 39);
  // La médiane des intervalles de présentation valait 40 ms : l'image suivante devenait due une
  // image trop tôt et les images partaient par paires, 5 ms puis 75 ms.
  for (const std::int64_t early : early_us) CHECK(early < kFrameUs / 2);
  for (std::size_t i = 1; i < interval_us.size(); ++i) {
    CHECK(interval_us[i] >= 35'000);
    CHECK(interval_us[i] <= 45'000);
  }
}

}  // namespace
}  // namespace ajp::app::play
