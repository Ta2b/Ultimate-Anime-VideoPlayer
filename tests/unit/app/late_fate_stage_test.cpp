// Producteur « une image à la fois » (chantier 7.6, UpscaleSetup::late_fate) : chaque instant est comparé à
// l'horloge prédite juste avant son calcul. Ici, l'horloge saute pendant que la paire attend un emplacement :
// avec late_fate, ses instants passés ne sont ni calculés ni écrits ; sans, ils sortent en retard.
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

// Une horloge que le test déplace pendant que le thread gpu la lit.
struct MovingClock final : MasterClock {
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

struct Played {
  std::vector<std::string> shape;
  std::size_t rife = 0;
  std::size_t span = 0;
  std::size_t drba = 0;
};

Played play(bool late_fate, bool drba = false) {
  Rig rig;
  MovingClock clock;
  clock.value =
      ClockSnapshot{.media_us = 0, .host_us = host_now_us(), .rate = 1.0, .epoch = 0, .valid = true};
  rig.live->set_free(true, 6'060, host_now_us());
  UpscaleSetup s = rig.setup(true, Factor{.num = 2, .den = 1}, 1);
  s.clock = &clock;
  s.late_fate = late_fate;
  if (drba) {
    s.rife = nullptr;
    s.distil_drba = &rig.drba_motion;
    s.drba_motion = true;
    s.warp_after_span = true;
  }
  rig.start(std::move(s), &clock);
  rig.feed(1, 1);
  if (drba) rig.feed(2, 2);              // DistilDRBA attend l'image d'après pour montrer la source 1
  REQUIRE(rig.collect(1, false));        // le seul emplacement est tenu
  rig.feed(drba ? 3 : 2, drba ? 3 : 2);  // la paire (1, 2) est préparée à l'heure, puis attend l'emplacement
  // La paire (1, 2) préparée (entrée de RIFE écrite, ou mesures de la suivante lancées pour DistilDRBA) :
  // elle attend l'emplacement. Une condition, pas une attente fixe (B40-08) : sous une machine chargée, 150
  // ms ne suffisaient pas toujours.
  REQUIRE(logged(rig, drba ? "scores_start 2 3" : "rife_input"));
  clock.jump(10'000'000);  // l'horloge passe la paire pendant l'attente
  rig.release_all();
  (void)rig.collect(4, true, 400ms);
  rig.stage.reset();
  return Played{.shape = shape_of(rig.sink.frames()),
                .rife = rig.backend.count("engine rife"),
                .span = rig.backend.count("engine span"),
                .drba = rig.backend.count("engine drba_motion")};
}

}  // namespace

TEST_CASE("Étage gpu : Rapide, un instant passé pendant l'attente n'est ni calculé ni écrit",
          "[app][stage][free]") {
  const Played late = play(true);
  CHECK(late.shape == std::vector<std::string>{"1"});
  CHECK(late.rife == 0);
  CHECK(late.span == 1);  // la source 2 en retard : sans warp après SPAN, rien ne lit son SPAN
}

TEST_CASE("Étage gpu : Rapide, DistilDRBA ne calcule pas non plus un instant déjà passé (A09-07)",
          "[app][stage][free]") {
  const Played late = play(true, true);
  CHECK(late.shape == std::vector<std::string>{"1"});
  CHECK(late.drba == 0);
  const Played old = play(false, true);
  INFO("forme " << old.shape.size() << ", passes " << old.drba);
  REQUIRE(old.shape.size() >= 2);
  CHECK(old.shape[1] == "1-2");
  CHECK(old.drba >= 1);
}

TEST_CASE("Étage gpu : Rapide sans late_fate, la paire sort quand même, en retard", "[app][stage][free]") {
  const Played old = play(false);
  INFO("forme " << old.shape.size() << ", rife " << old.rife << ", span " << old.span);
  REQUIRE(old.shape.size() >= 2);
  CHECK(old.shape.front() == "1");
  CHECK(old.shape[1] == "1-2");  // calculée et écrite après son instant
  CHECK(old.rife >= 1);
}

}  // namespace ajp::app::play
