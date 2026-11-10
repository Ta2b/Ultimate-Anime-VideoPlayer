// L'interface ne présente pas seule juste avant une image neuve (A24-02, app/play/overlay_hold.h) : la règle
// pure, puis Session avec un faux écran dont chaque présentation prend un vblank.
#include "app/play/overlay_hold.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

TEST_CASE("Interface : gardée près de l'image suivante, libre entre deux images", "[app][overlay]") {
  constexpr std::int64_t kFrame = 41'708;  // 23,976 i/s
  constexpr std::int64_t k144 = 6'944;
  // 144 Hz : libre jusqu'à deux périodes d'écran avant l'image suivante.
  CHECK_FALSE(overlay_holds(false, 1'000'000, 1'000'000 + 5'000, kFrame, k144));
  CHECK_FALSE(overlay_holds(false, 1'000'000, 1'000'000 + kFrame - 2 * k144, kFrame, k144));
  CHECK(overlay_holds(false, 1'000'000, 1'000'000 + kFrame - 2 * k144 + 1, kFrame, k144));
  CHECK(overlay_holds(false, 1'000'000, 1'000'000 + kFrame + 3'000, kFrame, k144));  // image en retard
  // Interpolation à la fréquence de l'écran : toujours gardée, l'interface suit les images.
  CHECK(overlay_holds(false, 1'000'000, 1'000'000 + 1, k144, k144));
  // Pause, aucune image encore, ou plus d'image depuis kOverlayHoldUs : elle part tout de suite.
  CHECK_FALSE(overlay_holds(true, 1'000'000, 1'000'000 + kFrame - 1, kFrame, k144));
  CHECK_FALSE(overlay_holds(false, 0, 1'000'000, kFrame, k144));
  CHECK_FALSE(overlay_holds(false, 1'000'000, 1'000'000 + kOverlayHoldUs, kFrame, k144));
  // Périodes inconnues : gardée tant que la lecture coule.
  CHECK(overlay_holds(false, 1'000'000, 1'000'000 + 5'000, 0, k144));
  CHECK(overlay_holds(false, 1'000'000, 1'000'000 + 5'000, kFrame, 0));
}

TEST_CASE("Interface : un changement gardé part avec l'image, une seule fois", "[app][overlay]") {
  OverlayHold hold;
  CHECK_FALSE(hold.note(false, true, true, false).ui);  // gardé
  CHECK(hold.pending());
  CHECK_FALSE(hold.note(true, false, true, false).osd);                     // l'OSD aussi
  const OverlayHold::Due with_frame = hold.note(false, false, true, true);  // l'image neuve les emporte
  CHECK(with_frame.ui);
  CHECK(with_frame.osd);
  CHECK_FALSE(hold.pending());
  CHECK_FALSE(hold.note(false, false, false, false).ui);  // rien de plus après
  CHECK(hold.note(false, true, false, false).ui);         // sans garde : tout de suite
}

// 23,976 i/s pendant 3 s, tours de 1 ms, l'interface change toutes les 100 ms (la barre visible), 2 ms avant
// l'échéance d'une image sur deux à peu près. Chaque présentation du faux écran prend 16,667 ms (FIFO).
struct Flow {
  std::size_t alone = 0;  // présentations de l'interface seule, sans image neuve
  std::int64_t spread_us =
      0;  // l'écart entre l'image neuve sortie le plus tôt et le plus tard, face à son pts
};
Flow play_with_bar() {
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session(SessionConfig{}, stages, screen, clock);
  constexpr std::int64_t kFrame = 41'708;
  const auto* ui = reinterpret_cast<const ImDrawData*>(&screen);  // jamais lue par la doublure
  session.handle(ring_frame(0, 0));
  REQUIRE(session.tick(screen.now_us, UiFrame{.draw = ui, .changed = true}));
  clock.value = {.media_us = 0, .host_us = screen.now_us, .rate = 1.0, .epoch = 0, .valid = true};
  const std::int64_t start = screen.now_us;
  Flow flow;
  std::int64_t earliest = INT64_MAX;
  std::int64_t latest = INT64_MIN;
  std::int64_t fed = 0;
  std::int64_t next_ui = start + 100'000;
  while (screen.now_us - start < 3'000'000) {
    const std::int64_t media = screen.now_us - start;
    while ((fed + 1) * kFrame < media + 60'000) {
      ++fed;
      session.handle(ring_frame(static_cast<std::uint32_t>(fed % 6), fed * kFrame));
    }
    const bool changed = screen.now_us >= next_ui;
    if (changed) next_ui += 100'000;
    const std::size_t before = screen.shown.size();
    const std::int64_t at = screen.now_us;
    auto shown = session.tick(at, UiFrame{.draw = ui, .changed = changed});
    REQUIRE(shown);
    if (screen.shown.size() != before) {
      if (*shown && (*shown)->fresh) {
        const std::int64_t offset = (*shown)->host_us - start - (*shown)->pts_us;
        earliest = std::min(earliest, offset);
        latest = std::max(latest, offset);
      } else {
        ++flow.alone;
      }
    } else {
      screen.now_us += 1'000;
    }
  }
  flow.spread_us = latest - earliest;
  return flow;
}

TEST_CASE("Interface : la barre visible ne décale aucune image neuve", "[app][overlay][session]") {
  FrenchUi french;
  const Flow flow = play_with_bar();
  INFO("présentations de l'interface seule " << flow.alone << ", écart " << flow.spread_us << " µs");
  CHECK(flow.alone == 0);
  // Chaque image part au tour où elle est due (tours de 1 ms) : aucune ne glisse d'un vblank (16,667 ms).
  CHECK(flow.spread_us <= 2'000);
}

TEST_CASE("L'interface demande une présentation quand elle change, et une pour s'effacer",
          "[app][overlay][session]") {
  FrenchUi french;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session(SessionConfig{}, stages, screen, clock);
  session.handle(ring_frame(0, 0));
  REQUIRE(session.tick(screen.now_us));
  screen.now_us += kOverlayHoldUs;  // plus d'image neuve : l'interface présente seule
  const std::size_t presents = screen.shown.size();
  const auto* ui = reinterpret_cast<const ImDrawData*>(&screen);  // jamais lue par la doublure
  REQUIRE(session.tick(screen.now_us, UiFrame{.draw = ui, .changed = true}));
  CHECK(screen.shown.size() == presents + 1);
  CHECK(screen.shown.back().ui);
  REQUIRE(session.tick(screen.now_us, UiFrame{.draw = ui, .changed = false}));
  CHECK(screen.shown.size() == presents + 1);  // inchangée : rien
  REQUIRE(session.tick(screen.now_us, UiFrame{}));
  CHECK(screen.shown.size() == presents + 2);  // effacée : une présentation sans elle
  CHECK_FALSE(screen.shown.back().ui);
}

TEST_CASE("Interface : en pause, le changement part tout de suite", "[app][overlay][session]") {
  FrenchUi french;
  FakeStages stages;
  FakeScreen screen;
  FakeClock clock;
  Session session(SessionConfig{}, stages, screen, clock);
  const auto* ui = reinterpret_cast<const ImDrawData*>(&screen);
  session.handle(ring_frame(0, 0));
  REQUIRE(session.tick(screen.now_us));
  session.command(cmd::TogglePause{});
  REQUIRE(session.tick(screen.now_us));
  const std::size_t presents = screen.shown.size();
  REQUIRE(session.tick(screen.now_us, UiFrame{.draw = ui, .changed = true}));
  CHECK(screen.shown.size() == presents + 1);
  CHECK(screen.shown.back().ui);
}

}  // namespace
}  // namespace ajp::app::play
