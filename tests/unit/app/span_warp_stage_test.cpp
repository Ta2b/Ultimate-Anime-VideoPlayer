// Warp après SPAN (upscale_stage.h, warps_spanned) : chaque instant mélange les deux sources agrandies de sa
// paire, y compris le premier instant d'une paire qui suit un dessin tenu, dont SPAN reprend le tampon
// de la précédente.
#include <cstdint>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "upscale_rig.h"

namespace ajp::app::play {
using namespace upscale_test;

TEST_CASE("Étage gpu : warp après SPAN, après un dessin tenu le premier instant lit b, pas a deux fois",
          "[app][stage]") {
  const bool poses = GENERATE(false, true);
  const float motion = GENERATE(0.0F, 500.0F);  // passe au milieu seule, ou RIFE à chaque instant
  INFO("poses " << poses << ", mouvement " << motion);
  Rig rig;
  // Un dessin sur deux images, comme l'anime : 3 tient le dessin de 2, 5 celui de 4.
  rig.backend.score(SourceId{2}, SourceId{3}, 0.0, 10);
  rig.backend.score(SourceId{4}, SourceId{5}, 0.0, 10);
  rig.backend.motion_max(motion);
  rig.live->set_keep_poses(poses);
  UpscaleSetup s = rig.setup(true, Factor{.num = 3, .den = 1});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.warp_after_span = true;
  rig.start(std::move(s));
  rig.feed(1, 6);
  REQUIRE(rig.collect(12));  // 6 sources, deux instants dans les paires (1, 2), (3, 4) et (5, 6)
  rig.stage.reset();
  // a est l'agrandissement du dessin d'avant (celui de 2 sert à 3, celui de 4 à 5), b celui de la source.
  using Pair = std::pair<std::uint32_t, std::uint32_t>;
  CHECK(rig.backend.warp_pairs() == std::vector<Pair>{{1, 2}, {1, 2}, {2, 4}, {2, 4}, {4, 6}, {4, 6}});
  CHECK(rig.backend.count("engine span") == 4);  // 1, 2, 4 et 6 : les dessins tenus reprennent la précédente
}

}  // namespace ajp::app::play
