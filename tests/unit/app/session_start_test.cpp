// Le départ de la lecture : l'horloge et --seconds partent quand la première image est rendue, pas quand
// elle est choisie. Sous lavapipe sans cache, ce premier rendu compile les shaders (plus d'une seconde) :
// partie avant, l'horloge courait pendant ce temps et la lecture en sautait les images.
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/session.h"
#include "doubles.h"

namespace ajp::app::play {
namespace {

using test::ring_frame;

TEST_CASE("Départ : l'horloge part après le premier rendu, --seconds compte de là et quitte",
          "[app][session]") {
  test::FakeStages stages;
  test::FakeScreen screen;
  test::FakeClock clock;
  Session session(SessionConfig{.stop_after_us = 1'000'000}, stages, screen, clock);
  session.handle(ring_frame(0, 0));
  screen.period_us = 1'200'000;  // le premier rendu dure 1,2 s
  REQUIRE(session.tick(screen.now_us));
  REQUIRE(stages.log.size() == 1);
  CHECK(stages.log.front() == "play 0 0");
  const std::int64_t shown_us = screen.now_us;
  clock.value = {.media_us = 0, .host_us = shown_us, .rate = 1.0, .valid = true};

  // 1,2 s après le choix de l'image, mais à peine après son rendu : la lecture continue.
  screen.period_us = 16'667;
  screen.now_us = shown_us + 900'000;
  REQUIRE(session.tick(screen.now_us));
  CHECK_FALSE(session.finished());

  // Une seconde après le rendu : fini, et c'est un arrêt demandé, pas une fin de fichier qui
  // enchaînerait l'épisode suivant.
  screen.now_us = shown_us + 1'000'000;
  REQUIRE(session.tick(screen.now_us));
  CHECK(session.finished());
  CHECK(session.quit_asked());
}

TEST_CASE("Départ : en pause avant le premier rendu, l'horloge attend la reprise", "[app][session]") {
  test::FakeStages stages;
  test::FakeScreen screen;
  test::FakeClock clock;
  Session session(SessionConfig{}, stages, screen, clock);
  session.handle(ring_frame(0, 0));
  session.command(cmd::Pause{});
  REQUIRE(session.tick(screen.now_us));
  CHECK(stages.log == std::vector<std::string>{"pause"});
  session.command(cmd::Play{});
  CHECK(stages.log == std::vector<std::string>{"pause", "play 0 0"});
}

}  // namespace
}  // namespace ajp::app::play
