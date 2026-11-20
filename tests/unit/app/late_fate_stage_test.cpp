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
#include <catch2/generators/catch_generators.hpp>

#include "app/play/live_settings.h"
#include "upscale_rig.h"

namespace kirari::app::play {
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

TEST_CASE("Étage gpu : Rapide garde le SPAN de B quand tous ses intermédiaires passent",
          "[app][stage][free]") {
  Rig rig;
  MovingClock clock;
  clock.value =
      ClockSnapshot{.media_us = -1'000'000, .host_us = host_now_us(), .rate = 1.0, .epoch = 0, .valid = true};
  rig.live->set_free(true, 6'060, host_now_us());
  UpscaleSetup s = rig.setup(true, Factor{.num = 2, .den = 1}, 1);
  s.clock = &clock;
  s.late_fate = true;
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.warp_after_span = true;
  rig.start(std::move(s), &clock);
  rig.feed(1, 2);
  REQUIRE(rig.collect(1, false));
  REQUIRE(logged(rig, "rife_input"));

  // Les deux instants 46 060 et 66 060 sont en retard. B (80 000) n'a pas de sortie propre sur la
  // grille ; elle reste toutefois la source gauche de la paire (B, C).
  clock.jump(70'000);
  rig.release_all();
  const auto until = std::chrono::steady_clock::now() + 2s;
  while (!rig.stage->idle() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(1ms);
  REQUIRE(rig.stage->idle());
  CHECK(pts_of(rig.sink.frames()) == std::vector<std::int64_t>{40'000});
  CHECK(rig.backend.count("engine rife_motion") == 0);

  clock.jump(70'000);
  rig.feed(3, 3);
  REQUIRE(rig.collect(2));
  rig.stage.reset();
  const auto pairs = rig.backend.warp_pairs();
  REQUIRE_FALSE(pairs.empty());
  CHECK(pairs.front() == std::pair<std::uint32_t, std::uint32_t>{2, 3});
  CHECK(rig.backend.count("engine span") == 3);
}

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

TEST_CASE("Étage gpu : Rapide, le SPAN de B en avance ne force ni passe RIFE ni écriture tardive",
          "[app][stage][free]") {
  const bool ahead = GENERATE(false, true);
  const bool expired = GENERATE(false, true);
  const bool source_on_grid = GENERATE(false, true);
  INFO("avance " << ahead << ", retard " << expired << ", source sur la grille " << source_on_grid);
  Rig rig;
  MovingClock clock;
  clock.value =
      ClockSnapshot{.media_us = -1'000'000, .host_us = host_now_us(), .rate = 1.0, .epoch = 0, .valid = true};
  rig.live->set_free(true, source_on_grid ? 20'000 : 6'060, host_now_us());
  UpscaleSetup s = rig.setup(true, Factor{.num = 2, .den = 1}, 1);
  s.clock = &clock;
  s.late_fate = true;
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.span_b_ahead = ahead;
  s.warp_after_span = true;
  rig.start(std::move(s), &clock);
  rig.feed(1, 2);
  REQUIRE(rig.collect(1, false));  // A garde le seul slot
  REQUIRE(logged(rig, "rife_input"));
  const bool enqueued = ahead && source_on_grid;
  if (enqueued) REQUIRE(logged(rig, "engine span", 2));
  CHECK(rig.backend.count("engine span") == (enqueued ? 2 : 1));
  CHECK(rig.backend.count("engine rife_motion") == 0);

  if (expired) clock.jump(10'000'000);
  rig.release_all();
  if (expired) {
    const auto until = std::chrono::steady_clock::now() + 2s;
    while (!rig.stage->idle() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(1ms);
    REQUIRE(rig.stage->idle());
    CHECK(pts_of(rig.sink.frames()) == std::vector<std::int64_t>{40'000});
    CHECK(rig.backend.count("engine rife_motion") == 0);
  } else {
    REQUIRE(rig.collect(3));
    const std::vector<std::int64_t> pts = source_on_grid ? std::vector<std::int64_t>{40'000, 60'000, 80'000}
                                                         : std::vector<std::int64_t>{40'000, 46'060, 66'060};
    CHECK(pts_of(rig.sink.frames()) == pts);
    CHECK(rig.backend.rife_ts() == std::vector<float>{0.5F});
    CHECK(rig.backend.count("engine rife_motion") == 1);
  }
  rig.stage.reset();
  if (!expired || source_on_grid) CHECK(rig.backend.count("engine span") == 2);
}

TEST_CASE("Étage gpu : Rapide, la copie d'une pose attend toujours le slot", "[app][stage][free]") {
  Rig rig;
  MovingClock clock;
  clock.value =
      ClockSnapshot{.media_us = -1'000'000, .host_us = host_now_us(), .rate = 1.0, .epoch = 0, .valid = true};
  rig.live->set_free(true, 20'000, host_now_us());
  rig.live->set_keep_poses(true);
  rig.backend.offer_ahead(false);
  rig.backend.score(SourceId{1}, SourceId{2}, 0.0, 10);
  UpscaleSetup s = rig.setup(true, Factor{.num = 2, .den = 1}, 1);
  s.clock = &clock;
  s.late_fate = true;
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.skip_held = true;
  s.span_b_ahead = true;
  s.warp_after_span = true;
  rig.start(std::move(s), &clock);
  rig.feed(1, 3);
  REQUIRE(rig.collect(1, false));
  REQUIRE(logged(rig, "scores_now 2 3"));
  CHECK(rig.backend.count("engine span") == 1);
  CHECK(rig.backend.count("engine rife_motion") == 0);
  REQUIRE(rig.collect(2));
  rig.stage.reset();
  CHECK(pts_of(rig.sink.frames()) == std::vector<std::int64_t>{40'000, 100'000});
  CHECK(rig.backend.count("engine span") == 2);
  CHECK(rig.backend.count("engine rife_motion") == 0);
}

}  // namespace kirari::app::play
