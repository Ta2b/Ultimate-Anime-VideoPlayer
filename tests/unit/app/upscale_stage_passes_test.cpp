// L'étage gpu, ce qu'il paie et ce qu'il écrit dans deux cas du Rapide (audit 09) : RIFE mouvement sans
// « garder les poses » à un seul instant hors du milieu (B09-01), et DistilDRBA dont la sortie chargée
// d'écrire b passe en retard pendant qu'elle attend un slot (B09-03).
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/live_settings.h"
#include "upscale_rig.h"

namespace ajp::app::play {
using namespace upscale_test;
using namespace std::chrono_literals;

namespace {

// Une horloge que le test déplace pendant que le thread gpu la lit (comme late_fate_stage_test.cpp).
struct JumpClock final : MasterClock {
  mutable std::mutex mutex;
  ClockSnapshot value;
  ClockSnapshot snapshot() const override {
    const std::scoped_lock lock(mutex);
    return value;
  }
  void jump(std::int64_t media_us) {
    const std::scoped_lock lock(mutex);
    value.media_us = media_us;
    value.host_us = host_now_us();
  }
};

}  // namespace

TEST_CASE(
    "Étage gpu : RIFE mouvement sans poses, un instant hors du milieu, une seule passe par paire (B09-01)",
    "[app][stage]") {
  Rig rig;
  rig.live->set_keep_poses(false);
  UpscaleSetup s = rig.setup(true, Factor{.num = 3, .den = 2});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  rig.start(std::move(s));
  rig.feed(1, 5);  // quatre paires : t = 2/3, puis 1/3 et la source, deux fois
  REQUIRE(rig.collect(7));
  rig.stage.reset();
  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "1-2", "2-3", "3", "3-4", "4-5", "5"});
  CHECK(rig.backend.count("warp_blend") == 4);
  // Avant : 8 (la passe du milieu, que rien ne lisait, puis celle de l'instant).
  CHECK(rig.backend.count("engine rife_motion") == 4);
}

// Rapide à pas constant N = 2 : (2, 3) est un dessin tenu, l'instant t = 1/2 de la paire (1, 2) doit écrire
// b. Il est jugé Write, attend le seul slot, puis passe en retard pendant l'attente : Late. La source 2,
// encore à l'heure, doit alors s'écrire ; avant, elle sortait Same et 2 n'était jamais montrée.
TEST_CASE(
    "Étage gpu : DistilDRBA, b tenue passée en retard en attendant un slot, la source 2 s'écrit (B09-03)",
    "[app][stage]") {
  Rig rig;
  JumpClock clock;
  clock.value =
      ClockSnapshot{.media_us = 0, .host_us = host_now_us(), .rate = 1.0, .epoch = 0, .valid = true};
  rig.live->set_free(true, 20'000, host_now_us());
  rig.live->set_steady(true, {.vrr = true, .hz = 50.0}, 25.0, {.a = 500.0, .b = 500.0});
  REQUIRE(rig.live->steady_active());
  rig.backend.score(SourceId{2}, SourceId{3}, 0.0, 10);  // (b, après) tenu
  UpscaleSetup s = rig.setup(true, Factor{.num = 2, .den = 1}, 1);
  s.rife = nullptr;
  s.distil_drba = &rig.drba;
  s.skip_held = true;
  s.clock = &clock;
  rig.start(std::move(s), &clock);
  rig.feed(1, 2);
  REQUIRE(rig.collect(1, false));  // la source 1 tient le seul slot
  rig.feed(3, 3);                  // la paire (1, 2) se prépare à l'heure, b attend le slot
  // Les mesures de (2, 3) relues : la paire (1, 2) est décidée. Que l'horloge saute avant ou après le premier
  // jugement de b, la source 2 doit s'écrire.
  REQUIRE(logged(rig, "scores_take"));
  clock.jump(70'000);  // l'instant 1/2 (60 000) est passé, la source 2 (80 000) ne l'est pas
  rig.release_all();
  rig.feed(4, 5);
  (void)rig.collect(4, true, 1500ms);
  rig.stage.reset();
  const std::vector<std::string> shape = shape_of(rig.sink.frames());
  REQUIRE(shape.size() >= 2);
  CHECK(shape[1] == "2");
}

}  // namespace ajp::app::play
