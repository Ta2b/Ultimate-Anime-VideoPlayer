// La place des sous-titres (session_subtitles.cpp) : la taille de sortie et la barre du lecteur vont dans
// les demandes à subs, et l'image à l'écran redemande les siens quand l'une des deux change.
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "app/play/controls.h"
#include "app/play/session.h"
#include "doubles.h"
#include "french_ui.h"

namespace kirari::app::play {
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
  void tick(const UiFrame& ui = {}) { REQUIRE(session.tick(screen.now_us, ui)); }
  // Une image à l'écran, sa réponse de sous-titres reçue : la situation d'une pause.
  void show_first() {
    session.handle(ring_frame(0, 0));
    session.handle(subs::SubtitleFrame{.pts_us = 0, .epoch = 0});
    tick();
    REQUIRE_FALSE(stages.requests.empty());
  }
};

}  // namespace

TEST_CASE("Montée des dialogues : la part de la hauteur que prend la barre", "[app][session]") {
  CHECK(line_lift(0, 1080) == 0.0F);
  CHECK(line_lift(190, 0) == 0.0F);
  CHECK(line_lift(108, 1080) == Catch::Approx(0.1F));
  CHECK(line_lift(900, 1080) == subs::kMaxLineLift);  // jamais plus de la moitié
}

TEST_CASE("Sous-titres : la fenêtre change de taille, l'image à l'écran redemande les siens (A24-20)",
          "[app][session]") {
  FrenchUi french;
  Rig rig(SessionConfig{.subtitles = true, .video_width = 1920, .video_height = 1080});
  rig.show_first();
  CHECK(rig.stages.requests.back().target.width == 1920);
  const std::size_t before = rig.stages.requests.size();
  rig.tick();  // rien n'a changé : aucune demande de plus
  CHECK(rig.stages.requests.size() == before);

  rig.screen.output = {2560, 1440};  // plein écran, sans image neuve (pause)
  rig.tick();
  REQUIRE(rig.stages.requests.size() == before + 1);
  CHECK(rig.stages.requests.back().pts_us == 0);
  CHECK(rig.stages.requests.back().target.width == 2560);
  CHECK(rig.stages.requests.back().target.height == 1440);
  rig.tick();  // une seule demande par changement
  CHECK(rig.stages.requests.size() == before + 1);
}

TEST_CASE("Sous-titres : la barre paraît, libass monte les dialogues ; elle part, ils redescendent (A24-19)",
          "[app][session]") {
  FrenchUi french;
  Rig rig(SessionConfig{.subtitles = true, .video_width = 1920, .video_height = 1080});
  rig.show_first();
  CHECK(rig.stages.requests.back().target.line_lift == 0.0F);
  const std::size_t before = rig.stages.requests.size();

  const auto* drawn = reinterpret_cast<const ImDrawData*>(&rig.screen);  // jamais lue par la doublure
  rig.tick(UiFrame{.draw = drawn, .changed = true, .bottom = 108});
  REQUIRE(rig.stages.requests.size() == before + 1);
  CHECK(rig.stages.requests.back().pts_us == 0);
  CHECK(rig.stages.requests.back().target.line_lift == Catch::Approx(0.1F));
  rig.session.handle(subs::SubtitleFrame{.pts_us = 0, .epoch = 0, .line_lift = 0.1F});
  rig.tick(UiFrame{.draw = drawn, .changed = false, .bottom = 108});
  CHECK(rig.stages.requests.size() == before + 1);

  rig.tick();  // la barre est partie
  REQUIRE(rig.stages.requests.size() == before + 2);
  CHECK(rig.stages.requests.back().target.line_lift == 0.0F);
}

// Après un saut, plus aucun sous-titre, même aux répliques suivantes. Pendant la
// préroulée, l'image à l'écran est encore celle d'avant le saut ; la barre qui paraît (la flèche l'affiche)
// redemandait ses sous-titres à l'époque passée. subs jette une demande d'époque passée sans répondre :
// chacune restait comptée, et au bout de kMaxSubtitleRequests plus aucune demande ne partait.
TEST_CASE("Sous-titres : la barre qui bouge pendant la préroulée d'un saut ne bloque pas les demandes",
          "[app][session]") {
  FrenchUi french;
  Rig rig(SessionConfig{.subtitles = true, .video_width = 1920, .video_height = 1080});
  rig.show_first();
  const std::size_t sought = rig.stages.requests.size();
  rig.session.command(cmd::Seek{.target_us = 5'000'000});
  rig.session.handle(ring_frame(1, 5'000'000, 1));  // la première image du saut, sans réponse encore
  REQUIRE(rig.stages.requests.back().epoch == 1);

  // La barre glisse : sa hauteur change à chaque tour, avant que la première image ne soit montrée.
  const auto* drawn = reinterpret_cast<const ImDrawData*>(&rig.screen);  // jamais lue par la doublure
  for (int bottom = 20; bottom <= 120; bottom += 20)
    rig.tick(UiFrame{.draw = drawn, .changed = true, .bottom = bottom});
  for (std::size_t i = sought; i < rig.stages.requests.size(); ++i) {
    INFO("demande " << rig.stages.requests[i].pts_us);
    CHECK(rig.stages.requests[i].epoch == 1);  // une demande d'avant le saut resterait sans réponse
  }

  // Les réponses de l'époque 1 arrivent ; les images suivantes demandent toujours leurs sous-titres.
  for (const subs::RenderRequest& request : std::vector(rig.stages.requests)) {
    if (request.epoch == 1) rig.session.handle(subs::SubtitleFrame{.pts_us = request.pts_us, .epoch = 1});
  }
  for (std::int64_t i = 1; i <= 6; ++i) {
    const std::size_t before = rig.stages.requests.size();
    const std::int64_t pts = 5'000'000 + (i * 41'708);
    rig.session.handle(ring_frame(static_cast<std::uint32_t>(1 + (i % 3)), pts, 1));
    INFO("image " << i);
    REQUIRE(rig.stages.requests.size() == before + 1);
    CHECK(rig.stages.requests.back().pts_us == pts);
    rig.session.handle(subs::SubtitleFrame{.pts_us = pts, .epoch = 1});
    rig.tick();
  }
}

}  // namespace kirari::app::play
