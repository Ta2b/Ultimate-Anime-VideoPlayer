// Les branches de l'étage gpu que l'audit A09-16 a trouvées sans test : le RIFE des paires calmes, et
// l'image que DistilDRBA saute puis reprend comme image d'avant (keep_skipped et son pont).
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "doubles.h"
#include "upscale_rig.h"

namespace ajp::app::play {
using namespace std::chrono_literals;
using namespace upscale_test;

TEST_CASE("Étage gpu : RIFE des paires calmes, la passe du milieu change de moteur une fois le calme mesuré",
          "[app][stage]") {
  Rig rig;
  test::FakeEngine calm{"rife_calm", {"input"}, {"flow", "mask"}};
  UpscaleSetup s = rig.setup(true, Factor{.num = 3, .den = 1});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.rife_motion_calm = &calm;
  rig.start(std::move(s));
  rig.feed(1, 4);
  REQUIRE(rig.collect(10));
  // La première paire n'a pas de mesure : le moteur entier. Les deux suivantes relisent un mouvement faible.
  CHECK(rig.backend.count("engine rife_motion") == 1);
  CHECK(rig.backend.count("engine rife_calm") == 2);
  // Un mouvement fort, relu à la paire d'après : le moteur entier revient, à chaque instant.
  rig.backend.motion_max(40.0F);
  rig.feed(5, 7);
  REQUIRE(rig.collect(19));
  rig.stage.reset();
  CHECK(rig.backend.count("engine rife_calm") == 3);  // la paire (4, 5) se décide encore sur le calme
  CHECK(rig.backend.count("engine rife_motion") > 1);
  CHECK(rig.reported().empty());
}

// Une horloge que le test change pendant que l'étage la lit.
struct SwitchedClock final : MasterClock {
  ClockSnapshot snapshot() const override {
    const std::scoped_lock lock(mutex);
    return value;
  }
  void set(const ClockSnapshot& next) {
    const std::scoped_lock lock(mutex);
    value = next;
  }
  mutable std::mutex mutex;
  ClockSnapshot value;
};

TEST_CASE("Étage gpu : DistilDRBA, une image sautée pour retard devient l'image d'avant de la reprise",
          "[app][stage]") {
  Rig rig;
  SwitchedClock clock;  // d'abord sans valeur : rien n'est sauté
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.distil_drba = &rig.drba;
  s.clock = &clock;
  rig.start(std::move(s), &clock);
  rig.feed(1, 2);
  REQUIRE(rig.collect(1));
  // L'horloge loin devant : la 2, ingérée en avance derrière la 1, est sautée dès que la 3 est là.
  clock.set(ClockSnapshot{
      .media_us = 10'000'000, .host_us = host_now_us(), .rate = 1.0, .epoch = 0, .valid = true});
  rig.feed(3, 3);
  for (int i = 0; i < 200 && rig.live->skipped() == 0; ++i) std::this_thread::sleep_for(5ms);
  REQUIRE(rig.live->skipped() == 1);
  clock.set(ClockSnapshot{});
  rig.feed(4, 6);
  (void)rig.collect(6, true, 500ms);
  rig.stage.reset();
  // La 2 n'est ni calculée ni montrée ; la 3 repart seule, puis les paires reprennent.
  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "3", "3-4", "4", "4-5", "5"});
  // Elle n'est pas relue : son image, ingérée derrière la 1, sert d'image d'avant, et le pont la mesure
  // contre l'image de reprise pour savoir si une coupe les sépare.
  CHECK(rig.backend.count("ingest 2") == 1);
  CHECK(rig.backend.count("scores_start 2 3") == 1);
  CHECK(rig.backend.count("engine drba") == 2);
  CHECK(rig.live->skipped() == 1);
  CHECK(rig.credits.load() == 5);  // la 6 attend sa suivante
  CHECK(rig.reported().empty());
}

}  // namespace ajp::app::play
