// Smooth (--smooth-drawings, chantier 7.9) : un dessin tenu deux sources, suivi d'un dessin nouveau, se fond
// en un seul mouvement. T3 : u linéaire sur toute la tenue ; l'étage gpu interpole a → b à travers la tenue.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "app/play/live_settings.h"
#include "app/play/rife_plan.h"
#include "upscale_rig.h"

namespace ajp::app::play {
using namespace upscale_test;

TEST_CASE("Smooth : T3, u linéaire à 1e-9 sur toute la tenue de deux sources", "[free][drawings]") {
  // La tenue va de a (position 0) à b (position 2) ; u = position / 2, continu au passage de a'.
  for (int k = 1; k <= 2000; ++k) {
    const double position = static_cast<double>(k) / 1000.0;  // ]0, 2]
    const bool second = position > 1.0;
    const double t = second ? position - 1.0 : position;
    CHECK(std::abs(smooth_hold_u(t, second) - position / 2.0) < 1e-9);
  }
  CHECK(smooth_hold_u(1.0, false) == smooth_hold_u(0.0, true));
}

TEST_CASE("Étage gpu : Smooth, la tenue (1, 2) se fond dans le mouvement de 1 vers 3",
          "[app][stage][smooth]") {
  const bool smooth = GENERATE(false, true);
  const bool spanned = GENERATE(false, true);  // warp après SPAN : SPAN de 3 fait en avance, une fois
  INFO("smooth " << smooth << ", warp après SPAN " << spanned);
  Rig rig;
  rig.backend.score(SourceId{1}, SourceId{2}, 0.0, 10);  // 2 tient le dessin de 1 ; 3 est nouveau
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.smooth_drawings = smooth;
  s.warp_after_span = spanned;
  rig.start(std::move(s));
  // Smooth attend l'image suivante : la 5 permet de faire la 4, et attend elle-même.
  rig.feed(1, smooth ? 5 : 4);
  const std::size_t expected = smooth ? 7 : 6;
  REQUIRE(rig.collect(expected));
  rig.stage.reset();
  const std::vector<RingFrame> frames = rig.sink.frames();
  // Un SPAN par source montrée ou lue, la tenue 2 reprend celui de 1 : 1, 3, 4 (et 5 jamais faite).
  if (spanned) CHECK(rig.backend.count("engine span") == 3);
  if (!smooth) {
    CHECK(shape_of(frames) == std::vector<std::string>{"1", "2", "2-3", "3", "3-4", "4"});
    return;
  }
  // La source tenue 2 ne sort pas telle quelle : à son pts, l'image à mi-chemin de 1 vers 3.
  CHECK(shape_of(frames) == std::vector<std::string>{"1", "1-3", "1-3", "2-3", "3", "3-4", "4"});
  CHECK(pts_of(frames) ==
        std::vector<std::int64_t>{40'000, 60'000, 80'000, 100'000, 120'000, 140'000, 160'000});
  // Le warp de la paire (1, 3) : 1/4 et 1/2 sur la passe du milieu. La paire (2, 3) reprend à 3/4 : RIFE y
  // estime l'instant lui-même (une seule image, pas de passe au milieu qui la serve).
  const std::vector<float> warps = rig.backend.warp_ts();
  REQUIRE(warps.size() >= 2);
  CHECK(warps[0] == 0.25F);
  CHECK(warps[1] == 0.5F);
  const std::vector<float> rife = rig.backend.rife_ts();
  CHECK(std::ranges::find(rife, 0.75F) != rife.end());
}

TEST_CASE("Étage gpu : Smooth, une tenue de trois sources garde le rythme des sources",
          "[app][stage][smooth]") {
  Rig rig;
  rig.backend.score(SourceId{1}, SourceId{2}, 0.0, 10);
  rig.backend.score(SourceId{2}, SourceId{3}, 0.0, 10);  // 3 tient encore : b n'est pas la suivante
  UpscaleSetup s = rig.setup(true);
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.smooth_drawings = true;
  rig.start(std::move(s));
  rig.feed(1, 5);
  REQUIRE(rig.collect(5));
  rig.stage.reset();
  CHECK(shape_of(rig.sink.frames()) == std::vector<std::string>{"1", "2", "3", "3-4", "4"});
}

}  // namespace ajp::app::play
