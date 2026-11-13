// Période de l'écran mesurée (SDL ne la donne pas) : la présentation est à la demande, les
// intervalles mesurés suivent donc la cadence des images et non celle de l'écran.
#include <cstdint>
#include <optional>
#include <string>
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
struct OutputRate final : StatsSource {
  std::int64_t period_us = 0;
  std::optional<std::string> rife() const override { return std::nullopt; }
  std::optional<std::string> memory() override { return std::nullopt; }
  std::int64_t output_period_us() const override { return period_us; }
};

TEST_CASE("Cadence fixe : la composition n'est gardée que si l'image peut se représenter", "[app][session]") {
  const auto redraws = [](std::int64_t output_us, bool steady, std::int64_t floor_us) {
    test::FakeStages stages;
    test::FakeScreen screen;
    test::FakeClock clock;
    screen.period_us = 6'000;
    OutputRate rate;
    rate.period_us = output_us;
    Session session(SessionConfig{.display_period_us = 6'000, .extra = &rate}, stages, screen, clock);
    session.set_steady_cadence(steady);
    session.set_vrr_floor(floor_us);
    session.handle(ring_frame(0, 0));
    REQUIRE(session.tick(screen.now_us));
    REQUIRE(!screen.shown.empty());
    return screen.shown.back().redraws;
  };
  CHECK_FALSE(redraws(6'000, false, 0));       // ni cadence fixe ni plancher : jamais représentée
  CHECK_FALSE(redraws(6'000, true, 0));        // interpolation au rythme de l'écran : chaque image est neuve
  CHECK(redraws(41'708, true, 0));             // 24 i/s sur 165 Hz : représentée à chaque période
  CHECK(redraws(0, true, 0));                  // cadence inconnue : gardée
  CHECK_FALSE(redraws(6'000, false, 20'833));  // VRR, sortie au-dessus du plancher de 48 Hz
  CHECK(redraws(41'708, false, 20'833));       // VRR, 24 i/s sous le plancher
}

TEST_CASE("Cadence fixe : une image représentée garde la composition une seconde", "[app][session]") {
  test::FakeStages stages;
  test::FakeScreen screen;
  test::FakeClock clock;
  screen.period_us = 6'000;
  OutputRate rate;
  rate.period_us = 6'000;
  Session session(SessionConfig{.display_period_us = 6'000, .extra = &rate}, stages, screen, clock);
  session.set_steady_cadence(true);
  session.handle(ring_frame(0, 0));
  REQUIRE(session.tick(screen.now_us));
  CHECK_FALSE(screen.shown.back().redraws);
  screen.now_us += 6'000;
  REQUIRE(session.tick(screen.now_us));  // rien de neuf, la période passée : représentée
  REQUIRE(screen.shown.size() == 2);
  screen.now_us += 6'000;
  REQUIRE(session.tick(screen.now_us));
  CHECK(screen.shown.back().redraws);  // une image manquée : la suivante garde sa composition
  screen.now_us += 1'100'000;
  REQUIRE(session.tick(screen.now_us));
  CHECK_FALSE(screen.shown.back().redraws);  // plus d'une seconde sans représentation : oubliée
  screen.now_us += 6'000;
  REQUIRE(session.tick(screen.now_us));
  CHECK(screen.shown.back().redraws);  // la représentation d'avant la fait garder de nouveau
}

}  // namespace ajp::app::play
