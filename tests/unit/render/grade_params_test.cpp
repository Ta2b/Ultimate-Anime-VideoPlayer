// decide_grade() : les valeurs des //!PARAM du grade anime guidé (anime-hdr-params.hook) pour une route, une
// source et un écran donnés.
#include "render/grade_params.h"

#include <cmath>
#include <limits>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace ajp::render {
namespace {

using Catch::Approx;

GradeContext pq_context(core::HdrMode route, double peak = 1000.0) {
  return GradeContext{
      .route = route, .source_hdr = false, .output_pq = true, .enabled = true, .peak_nits = peak};
}

// La réponse des curseurs du grade (grade_params.cpp) : puissance 0,7 sur la course, butée poussée par
// `gain`.
double resp(double x, double hi, double gain) {
  return gain * hi * std::pow(x / hi, 0.7);
}

}  // namespace

TEST_CASE("route native : blanc de la source à 203 nits, éclats depuis la moitié du pic") {
  GradeSettings settings;
  settings.shadows = 0.5;
  settings.highlights_rtx = 1.0;
  settings.area_rtx = 0.1;
  const auto decision = decide_grade(settings, pq_context(core::HdrMode::Native, 800.0));
  REQUIRE(decision);
  CHECK(decision->mode == GradeMode::Native);
  const GradeParams& p = decision->params;
  CHECK(p.peak == 800.0);
  CHECK(p.paper == 400.0);  // la moitié du pic
  CHECK(p.adapt == 1.0);
  CHECK(p.sdr_white == 203.0);
  CHECK(p.shadows == Approx(resp(0.5, 0.8, 1.2)));
  // h = 1 : 1 + 1,2 + 1,6 × 0,75 = 3,4.
  CHECK(p.spec_gain == Approx(3.4));
  CHECK(p.black_hi == 0.0);  // sans vrai noir
}

// C41-07 : la sortie SDR n'a plus de grade propre : elle prend le grade PQ ramené au SDR (sdr_tonemap).
TEST_CASE("sortie SDR : pas de grade propre, la méthode anime y ramène le grade PQ") {
  GradeSettings settings;
  for (const core::HdrMode route : {core::HdrMode::Off, core::HdrMode::Native}) {
    auto context = pq_context(route);
    context.output_pq = false;
    const auto decision = decide_grade(settings, context);
    REQUIRE(decision);
    CHECK(decision->mode == GradeMode::Neutral);
  }
}

TEST_CASE("le mode SDR sur une sortie HDR10 est gradé au blanc papier") {
  GradeSettings settings;
  const auto off = decide_grade(settings, pq_context(core::HdrMode::Off));
  REQUIRE(off);
  CHECK(off->mode == GradeMode::Native);
  CHECK(off->params.adapt == 0.0);
  CHECK(off->params.paper == 203.0);  // le blanc papier fixe
  // Sous un pic bas, le blanc papier reste à 75 % du pic : la marge des éclats.
  const auto low = decide_grade(settings, pq_context(core::HdrMode::Off, 200.0));
  REQUIRE(low);
  CHECK(low->params.paper == 150.0);
  // Une source HDR en mode SDR reste neutre.
  auto hdr_source = pq_context(core::HdrMode::Off);
  hdr_source.source_hdr = true;
  CHECK(decide_grade(settings, hdr_source)->mode == GradeMode::Neutral);
}

TEST_CASE("le grade reste neutre hors de ses conditions") {
  GradeSettings settings;
  auto context = pq_context(core::HdrMode::Native);
  context.enabled = false;
  CHECK(decide_grade(settings, context)->mode == GradeMode::Neutral);
  settings.grade = false;
  CHECK(decide_grade(settings, pq_context(core::HdrMode::Native))->mode == GradeMode::Neutral);
}

TEST_CASE("une source HDR n'est pas gradée") {
  GradeSettings settings;
  auto context = pq_context(core::HdrMode::Native);
  context.source_hdr = true;
  CHECK(decide_grade(settings, context)->mode == GradeMode::Neutral);
}

TEST_CASE("le peps de la passe anime et la lumière de la scène") {
  GradeSettings settings;
  settings.anime_contrast = 150;
  settings.anime_saturation = 80;
  settings.anime_highlights = 200;
  settings.scene_light = SceneLight{.apl_nits = 40.0, .lit = 0.1};
  settings.anime_exposure = 0.25;
  const auto decision = decide_grade(settings, pq_context(core::HdrMode::Native));
  REQUIRE(decision);
  const GradeParams& p = decision->params;
  CHECK(p.an_punch == Approx(0.5));
  CHECK(p.an_chroma == Approx(0.8));
  CHECK(p.an_spec == Approx(2.0));
  CHECK(p.fs_apl == Approx(40.0));
  CHECK(p.an_lift == Approx(kAnimeLift * std::exp2(0.25)));
  // Sans lumière connue : la luminance moyenne reste inconnue, le relèvement fixe.
  const auto plain = decide_grade(GradeSettings{}, pq_context(core::HdrMode::Native));
  REQUIRE(plain);
  CHECK(plain->params.fs_apl < 0.0);
  CHECK(plain->params.an_lift == kAnimeLift);
}

TEST_CASE("un réglage hors plage est refusé en nommant le champ") {
  GradeSettings settings;
  settings.black_lo = 51.0;
  auto refused = decide_grade(settings, pq_context(core::HdrMode::Native));
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
  CHECK(refused.error().detail.starts_with("grade.black_lo"));
  settings = GradeSettings{};
  settings.shadows = std::numeric_limits<double>::quiet_NaN();
  CHECK_FALSE(check_grade_settings(settings));
  settings = GradeSettings{};
  settings.area_rtx = 1.5;
  CHECK_FALSE(check_grade_settings(settings));
  CHECK_FALSE(decide_grade(GradeSettings{}, pq_context(core::HdrMode::Native, 99.0)));
}

TEST_CASE("un curseur float poussé à sa borne garde le grade") {
  // 0,8f relu en double vaut 0,800000012 : au bout du curseur Noirs profonds, le grade entier était refusé.
  GradeSettings settings;
  settings.shadows = static_cast<double>(0.8F);
  CHECK(check_grade_settings(settings));
  CHECK(decide_grade(settings, pq_context(core::HdrMode::Native)));
  settings.shadows = 0.81;
  CHECK_FALSE(check_grade_settings(settings));
}

TEST_CASE("vrai noir : seuils tirés de la rampe mesurée") {
  // Noir de la source relevé à 0,02 nit, puis une marche tous les 0,01 nit.
  std::vector<double> ramp(kBlackRampSteps);
  for (std::size_t k = 0; k < ramp.size(); ++k) ramp[k] = 0.02 + 0.01 * static_cast<double>(k);
  const auto two = black_thresholds(ramp, 2.0);
  REQUIRE(two);
  CHECK(two->lo == Approx(0.025));  // entre le noir et la première marche
  CHECK(two->hi == Approx(0.04));   // la deuxième marche
  const auto half = black_thresholds(ramp, 2.5);
  REQUIRE(half);
  CHECK(half->hi == Approx(0.045));
  // Plate ou ratée : rien.
  CHECK_FALSE(black_thresholds(std::vector<double>(kBlackRampSteps, 0.0), 2.0));
  CHECK_FALSE(black_thresholds(std::vector<double>{0.1}, 2.0));
}

TEST_CASE("sortie SDR de l'anime : surfaces au blanc SDR, marge des éclats du plan, vrai noir gardé",
          "[render]") {
  GradeParams p;
  p.peak = 800.0;
  p.paper = 400.0;
  p.sdr_white = 203.0;
  p.an_lift = 1.15;
  p.black_lo = 0.1;
  p.black_hi = 2.0;
  p.fs_apl = 10.0;        // scène sombre : toute la marge
  p.look_mixes[0] = 1.0;  // la pile d'HDR
  p.look_mixes[10] = 0.5;
  p.look_pq = 1.0;
  sdr_tonemap(p);
  CHECK(p.look_mixes[0] == 0.0);
  CHECK(p.look_mixes[10] == 0.0);
  CHECK(p.look_pq == 0.0);
  CHECK(p.sdr_tm == 1.0);
  // A41-07 : le vrai noir annoncé par le panneau s'applique aussi sur une sortie SDR.
  CHECK(p.black_lo == 0.1);
  CHECK(p.black_hi == 2.0);
  CHECK(p.tm_white == Approx(203.0 * 1.15));
  CHECK(p.tm_top == Approx(800.0));
  p.fs_apl = 203.0;  // scène très claire : la moitié du pic, comme 3-anime.hook
  sdr_tonemap(p);
  CHECK(p.tm_top == Approx(400.0));
  p.peak = 200.0;  // jamais sous les surfaces
  sdr_tonemap(p);
  CHECK(p.tm_top == Approx(1.25 * p.tm_white));
}

// C41-01, décision en attente de Tatoub : en sortie SDR, l'exposition par plan (±0,25 diaphragme) devrait
// bouger les surfaces, mais tm_white suit an_lift et l'annule sous le genou (une surface sort à
// L × an_lift / tm_white du blanc SDR). Marqué comme connu ([!shouldfail]) : il passe tant que le défaut est
// là, et échouera quand l'un des deux choix sera fait. Exposition par plan voulue : tm_white fixe, retirer la
// marque. Surfaces stables : retirer l'exposition par plan et inverser ce test.
TEST_CASE("sortie SDR de l'anime : l'exposition par plan bouge les surfaces (C41-01, connu)",
          "[render][!shouldfail]") {
  const auto sdr_anime = [](double e) {
    GradeSettings s;
    s.scene_light = SceneLight{.apl_nits = 40.0, .lit = 0.1};
    s.anime_exposure = e;
    const auto d = decide_grade(s, pq_context(core::HdrMode::Native));
    REQUIRE(d);
    GradeParams p = d->params;
    sdr_tonemap(p);
    return p;
  };
  const GradeParams dark = sdr_anime(-kAnimeExposureMaxE);
  const GradeParams bright = sdr_anime(kAnimeExposureMaxE);
  // +0,5 diaphragme : une surface 2^0,5 = 1,41 fois plus claire attendue.
  CHECK(bright.an_lift / bright.tm_white > 1.3 * dark.an_lift / dark.tm_white);
}

}  // namespace ajp::render
