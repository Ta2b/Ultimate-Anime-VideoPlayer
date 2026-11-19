// GMFSS Fortuna Union dans l'étage gpu (upscale_gmfss.cpp) : le mélange (M au milieu des paires fortes, RIFE
// sur les demi-paires) et GMFSS seul, sur le backend factice.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "app/play/gmfss_credit.h"
#include "doubles.h"
#include "upscale_rig.h"

namespace kirari::app::play {
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

// Une horloge arrêtée au pts que le test choisit, lue comme si elle venait d'être prise.
struct HeldClock final : MasterClock {
  mutable std::mutex mutex;
  std::int64_t media_us = 0;
  ClockSnapshot snapshot() const override {
    const std::scoped_lock lock(mutex);
    return ClockSnapshot{
        .media_us = media_us, .host_us = host_now_us(), .rate = 1.0, .epoch = 0, .valid = true};
  }
  void hold(std::int64_t us) {
    const std::scoped_lock lock(mutex);
    media_us = us;
  }
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
  s.gmfss.max_keys = 1;  // la clé au milieu seulement
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
  CHECK(rig.backend.count("gmfss_still") == pairs);  // le décor fixe de chaque clé repris des sources
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

TEST_CASE("Étage gpu : le mélange en travail plus grand que RIFE, M composée ou agrandie",
          "[app][stage][gmfss]") {
  const bool compose = GENERATE(false, true);
  INFO((compose ? "composée" : "agrandie"));
  Rig rig;
  GmfssEngines gmfss;
  UpscaleSetup s = rig.setup(true, Factor{.num = 4, .den = 1});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.rife_width = kWidth / 2;
  s.rife_height = kHeight / 2;
  s.gmfss = gmfss.setup(GmfssUse::Middle);
  s.gmfss.compose = compose;
  rig.backend.motion_max(40.0F);
  rig.start(std::move(s));
  rig.feed(1, 5);
  REQUIRE(rig.collect(17));
  rig.stage.reset();
  CHECK(rig.reported().empty());
  const std::size_t pairs = rig.backend.count("engine gmfss_fusion");
  CHECK(pairs >= 2);
  // Une composition par M (ses quatre splats d'abord), sinon un agrandissement.
  CHECK(rig.backend.count("soft_splat_grid") == (compose ? pairs : 0));
  CHECK(rig.backend.count("gmfss_compose") == (compose ? pairs : 0));
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
  // Chaque paire une seule fois, sur ses deux images, qu'elle soit faite en avance (l'image d'après déjà
  // derrière celle-ci) ou à l'arrivée.
  const std::vector<std::pair<std::uint32_t, std::uint32_t>> fed = rig.backend.gmfss_pairs();
  REQUIRE(fed.size() == 3);
  for (std::size_t i = 0; i < fed.size(); ++i)
    CHECK(fed[i] == std::pair{static_cast<std::uint32_t>(i + 1), static_cast<std::uint32_t>(i + 2)});
  const std::vector<std::string> laned = rig.backend.laned();
  CHECK(std::ranges::count(laned, std::string("pair gmfss_frames")) == 3);
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

TEST_CASE("Étage gpu : le mélange au crédit de temps, une M plus chère que la durée d'une image reste rare",
          "[app][stage][gmfss]") {
  Rig rig;
  GmfssEngines gmfss;
  UpscaleSetup s = rig.setup(true, Factor{.num = 4, .den = 1});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.gmfss = gmfss.setup(GmfssUse::Middle);
  s.gmfss.cost_us = 100'000;  // plus que les trois quarts de deux images : une M toutes les trois au plus
  test::FakeClock clock;  // le crédit ne vaut qu'en lecture (sans horloge, l'export prend toutes les clés)
  s.clock = &clock;
  rig.start(std::move(s), &clock);
  rig.backend.motion_max(40.0F);
  rig.feed(1, 4);
  REQUIRE(rig.collect(13));
  rig.feed(5, 8);
  REQUIRE(rig.collect(29));
  rig.feed(9, 12);
  REQUIRE(rig.collect(45));
  rig.stage.reset();
  CHECK(rig.reported().empty());
  const std::size_t fusions = rig.backend.count("engine gmfss_fusion");
  CHECK(fusions >= 1);
  CHECK(fusions <= 5);  // onze paires fortes : le crédit plein de départ, puis une sur trois environ
}

TEST_CASE("Étage gpu : le mélange, une clé sous le seuil quand le crédit déborde, aucune sinon",
          "[app][stage][gmfss]") {
  const bool spare = GENERATE(false, true);
  INFO((spare ? "crédit qui déborde" : "crédit juste"));
  Rig rig;
  GmfssEngines gmfss;
  UpscaleSetup s = rig.setup(true, Factor{.num = 4, .den = 1});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.gmfss = gmfss.setup(GmfssUse::Middle);
  // Plein à 120 ms : une M de 30 ms laisse la moitié du plein, une de 80 ms non.
  s.gmfss.cost_us = spare ? 30'000 : 80'000;
  s.gmfss.strong_px = 1.0e6F;  // aucune paire forte, quelle que soit la taille du banc
  test::FakeClock clock;  // le crédit ne vaut qu'en lecture (sans horloge, l'export prend toutes les clés)
  s.clock = &clock;
  rig.start(std::move(s), &clock);
  rig.backend.motion_max(8.0F);  // au-dessus du plancher de 15 px à 720 lignes (3,75 px ici)
  rig.feed(1, 4);
  REQUIRE(rig.collect(13));
  rig.stage.reset();
  CHECK(rig.reported().empty());
  const std::size_t fusions = rig.backend.count("engine gmfss_fusion");
  if (spare) {
    CHECK(fusions >= 1);
  } else {
    CHECK(fusions == 0);
  }
}

TEST_CASE("Étage gpu : le mélange, plus de clés quand le mouvement grandit, RIFE entre elles",
          "[app][stage][gmfss]") {
  const float motion = GENERATE(0.6F, 1.5F, 3.0F);
  INFO("mouvement " << motion << " fois le pas");
  Rig rig;
  GmfssEngines gmfss;
  UpscaleSetup s = rig.setup(true, Factor{.num = 4, .den = 1});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.gmfss = gmfss.setup(GmfssUse::Middle);
  s.gmfss.strong_px = 0.0F;  // Très élevé : chaque paire qui bouge
  s.gmfss.floor_px = 15.0F;  // sauf sous 15 px (gmfss_key_floor)
  rig.start(std::move(s));
  // Le mouvement se mesure à la paire d'avant : un plan calme d'abord (aucune clé sous le plancher, même en
  // Très élevé), puis le pas de 30 px à 720 lignes, à la hauteur du banc.
  rig.feed(1, 4);
  REQUIRE(rig.collect(13));
  const std::size_t calm_pairs = rig.backend.count("engine gmfss_pair");
  const std::size_t calm_fusions = rig.backend.count("engine gmfss_fusion");
  const std::size_t calm_splats = rig.backend.splat_ts().size();
  rig.backend.motion_max(motion * 30.0F * static_cast<float>(kHeight) / 720.0F);  // RIFE à la taille du banc
  rig.feed(5, 8);
  REQUIRE(rig.collect(29));
  rig.stage.reset();
  CHECK(rig.reported().empty());
  const std::size_t pairs = rig.backend.count("engine gmfss_pair") - calm_pairs;
  REQUIRE(pairs >= 2);
  const std::size_t keys = motion < 1.0F ? 1 : motion < 2.0F ? 2 : 3;  // une de plus par doublement
  // Au calme, aucune clé (ni paire GMFSS) ; la première paire du mouvement, décidée d'après la mesure de la
  // paire calme d'avant, non plus.
  CHECK(calm_fusions == 0);
  CHECK(rig.backend.count("engine gmfss_fusion") == keys * pairs);
  // Les clés aux instants k / (n + 1) ; à ×4, trois clés tombent sur les trois instants.
  const std::vector<float> all = rig.backend.splat_ts();
  const std::vector<float> splats(all.begin() + static_cast<std::ptrdiff_t>(calm_splats), all.end());
  const auto at = [&](float t) {
    return std::ranges::count_if(splats, [t](float x) { return std::abs(x - t) < 1e-4F; });
  };
  if (keys == 1) {
    CHECK(at(0.5F) == static_cast<std::ptrdiff_t>(splats.size()));
  } else if (keys == 2) {
    CHECK(at(1.0F / 3.0F) > 0);
    CHECK(at(2.0F / 3.0F) > 0);
    CHECK(at(0.5F) == 0);
  } else {
    CHECK(at(0.25F) > 0);
    CHECK(at(0.5F) > 0);
    CHECK(at(0.75F) > 0);
  }
}

TEST_CASE("Étage gpu : le mélange sans horloge (l'export), une clé à chaque paire, sans crédit",
          "[app][stage][gmfss]") {
  Rig rig;
  GmfssEngines gmfss;
  UpscaleSetup s = rig.setup(true, Factor{.num = 4, .den = 1});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.gmfss = gmfss.setup(GmfssUse::Middle);
  s.gmfss.max_keys = 1;
  s.gmfss.cost_us = 1'000'000;  // bien plus que le crédit plein
  rig.start(std::move(s));
  rig.backend.motion_max(40.0F);
  rig.feed(1, 4);
  REQUIRE(rig.collect(13));
  rig.stage.reset();
  CHECK(rig.reported().empty());
  const std::size_t pairs = rig.backend.count("engine gmfss_pair");
  CHECK(pairs >= 2);
  CHECK(rig.backend.count("engine gmfss_fusion") == pairs);
}

TEST_CASE("Étage gpu : Rapide, la grille place la première image après les clés de la paire",
          "[app][stage][gmfss][free]") {
  // L'horloge tient chaque paire au plus juste : prête à la source gauche. Sans le temps des clés dans la
  // grille, le premier instant serait en retard dès qu'on y ajoute la clé, et la paire forte restait à RIFE.
  constexpr std::int64_t kGap = 6'060;  // 165 Hz
  Rig rig;
  GmfssEngines gmfss;
  rig.live->set_free(true, kGap, host_now_us());
  UpscaleSetup s = rig.setup(true, Factor{.num = 2, .den = 1});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.gmfss = gmfss.setup(GmfssUse::Middle);
  s.gmfss.max_keys = 1;
  s.gmfss.cost_us = 20'000;  // la moitié d'une paire de 40 ms
  HeldClock clock;
  s.clock = &clock;
  rig.start(std::move(s), &clock);
  rig.backend.motion_max(40.0F);
  for (std::uint32_t id = 1; id <= 6; ++id) {
    clock.hold((static_cast<std::int64_t>(id) - 1) * kPeriodUs - kGap);
    rig.feed(id, id);
    (void)rig.collect(2 * id - 1, true, std::chrono::milliseconds(500));
  }
  rig.stage.reset();
  CHECK(rig.reported().empty());
  CHECK(rig.backend.count("engine gmfss_fusion") >= 2);
}

TEST_CASE("Étage gpu : le crédit des clés GMFSS garde toujours deux images de RIFE et SPAN",
          "[app][stage][gmfss]") {
  constexpr std::int64_t kSource = 41'708;  // 23,976 i/s
  // Service inconnu ou léger (travail 720p) : les trois quarts de l'image, comme avant.
  CHECK(gmfss_earned_us(kSource, 0) == kSource * 3 / 4);
  CHECK(gmfss_earned_us(kSource, 5'000) == kSource * 3 / 4);
  // Travail 1080p vers la 4K, 16 ms par image : les clés n'ont plus que ce qui reste après deux images.
  CHECK(gmfss_earned_us(kSource, 16'000) == kSource - 32'000);
  // Une image plus chère que la moitié de la source : plus aucune clé.
  CHECK(gmfss_earned_us(kSource, 25'000) == 0);
}

}  // namespace kirari::app::play
