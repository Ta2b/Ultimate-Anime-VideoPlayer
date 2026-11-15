// GMFSS Fortuna Union dans l'étage gpu (upscale_gmfss.cpp) : le mélange (M au milieu des paires fortes, RIFE
// sur les demi-paires) et GMFSS seul, sur le backend factice.
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "upscale_rig.h"

namespace ajp::app::play {
using namespace upscale_test;

namespace {

struct GmfssEngines {
  test::FakeEngine pair{
      "gmfss_pair",
      {"frames"},
      {"feature0_1", "motion0_1", "source0_2", "motion0_2", "source0_3", "motion0_3", "feature1_1",
       "motion1_1", "source1_2", "motion1_2", "source1_3", "motion1_3", "half0", "half1"}};
  test::FakeEngine fusion{"gmfss_fusion",
                          {"half0", "half1", "time", "warped0_1", "warped0_2", "warped0_3", "warped1_1",
                           "warped1_2", "warped1_3"},
                          {"out"}};
  GmfssSetup setup(GmfssUse use) { return {.use = use, .pair = &pair, .fusion = &fusion}; }
};

}  // namespace

TEST_CASE("Étage gpu : le mélange, M par GMFSS au milieu des paires fortes, RIFE sur les demi-paires",
          "[app][stage][gmfss]") {
  const bool poses = GENERATE(false, true);
  INFO((poses ? "poses gardées" : "sans poses"));
  Rig rig;
  GmfssEngines gmfss;
  rig.live->set_keep_poses(poses);
  UpscaleSetup s = rig.setup(true, Factor{.num = 4, .den = 1});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.gmfss = gmfss.setup(GmfssUse::Middle);
  rig.start(std::move(s));
  rig.feed(1, 4);
  REQUIRE(rig.collect(13));
  // Mouvement faible : RIFE seul.
  CHECK(rig.backend.count("engine gmfss_pair") == 0);
  CHECK(rig.backend.count("soft_splat_fill") == 0);
  const std::size_t calm_warps = rig.backend.warp_pairs().size();
  rig.backend.motion_max(40.0F);
  rig.feed(5, 8);
  REQUIRE(rig.collect(29));
  rig.stage.reset();
  CHECK(rig.reported().empty());
  // Les paires fortes : une paire et une fusion chacune, au milieu seulement.
  const std::size_t pairs = rig.backend.count("engine gmfss_pair");
  CHECK(pairs >= 2);
  CHECK(rig.backend.count("engine gmfss_fusion") == pairs);
  const std::vector<float> splats = rig.backend.splat_ts();
  CHECK(std::ranges::all_of(splats, [](float t) { return t == 0.5F; }));
  // Chaque instant hors du milieu se warpe entre une source et M (qui n'est pas une image ingérée, 0) : a et
  // M avant le milieu, M et b après ; M elle-même au milieu, sans warp.
  const std::vector<std::pair<std::uint32_t, std::uint32_t>> warps = rig.backend.warp_pairs();
  std::size_t with_middle = 0;
  for (std::size_t i = calm_warps; i < warps.size(); ++i) {
    if ((warps[i].first != 0) != (warps[i].second != 0)) ++with_middle;
  }
  CHECK(with_middle == 2 * pairs);
}

TEST_CASE("Étage gpu : GMFSS seul, la fusion à chaque instant, la paire une fois", "[app][stage][gmfss]") {
  Rig rig;
  GmfssEngines gmfss;
  UpscaleSetup s = rig.setup(true, Factor{.num = 3, .den = 1});
  s.rife = nullptr;
  s.gmfss = gmfss.setup(GmfssUse::Only);
  rig.start(std::move(s));
  rig.feed(1, 4);
  REQUIRE(rig.collect(10));
  rig.stage.reset();
  CHECK(rig.reported().empty());
  CHECK(rig.backend.count("engine gmfss_pair") == 3);
  CHECK(rig.backend.count("engine gmfss_fusion") == 6);
  CHECK(rig.backend.count("engine rife") == 0);
  const std::vector<float> splats = rig.backend.splat_ts();
  REQUIRE(splats.size() == 6);
  for (std::size_t i = 0; i < splats.size(); ++i)
    CHECK(splats[i] == (i % 2 == 0 ? 1.0F / 3.0F : 2.0F / 3.0F));
}

TEST_CASE("Étage gpu : GMFSS refusé sans ses graphes, ou seul à côté de RIFE", "[app][stage][gmfss]") {
  Rig rig;
  GmfssEngines gmfss;
  UpscaleSetup s = rig.setup(true, Factor{.num = 2, .den = 1});
  s.gmfss = gmfss.setup(GmfssUse::Only);  // RIFE image reste
  auto both = UpscaleStage::start(rig.init(nullptr), std::move(s));
  REQUIRE_FALSE(both);
  CHECK(both.error().code == RefusalCode::FieldOutOfRange);
  UpscaleSetup missing = rig.setup(true, Factor{.num = 2, .den = 1});
  missing.rife = nullptr;
  missing.rife_motion = &rig.motion;
  missing.gmfss = {.use = GmfssUse::Middle};
  auto none = UpscaleStage::start(rig.init(nullptr), std::move(missing));
  REQUIRE_FALSE(none);
  CHECK(none.error().code == RefusalCode::FieldMissing);
}

}  // namespace ajp::app::play
