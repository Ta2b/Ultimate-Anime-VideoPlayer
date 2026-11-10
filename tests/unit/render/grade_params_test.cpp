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
  CHECK(decision->source_white_nits == 203.0);
  const GradeParams& p = decision->params;
  CHECK(p.peak == 800.0);
  CHECK(p.paper == 400.0);  // la moitié du pic
  CHECK(p.adapt == 1.0);
  CHECK(p.sdr_white == 203.0);
  CHECK(p.shadows == Approx(resp(0.5, 0.8, 1.2)));
  // h = 1 : 1 + 1,2 + 1,6 × 0,75 = 3,4.
  CHECK(p.spec_gain == Approx(3.4));
  CHECK(p.sdr_out == 0.0);
  CHECK(p.black_hi == 0.0);  // sans vrai noir
}

TEST_CASE("blanc papier (truehdr_look coupé) : éclats jusqu'au pic, plafonné à 0,75 × pic") {
  GradeSettings settings;
  settings.truehdr_look = false;
  settings.paper_white = 400.0;
  settings.highlights_rtx = 0.5;  // x 2 : le plein effet des éclats
  const auto decision = decide_grade(settings, pq_context(core::HdrMode::Native, 401.0));
  REQUIRE(decision);
  CHECK(decision->params.paper == 301.0);  // floor(300,75 + 0,5)
  CHECK(decision->source_white_nits == 301.0);
  CHECK(decision->params.adapt == 0.0);
  CHECK(decision->params.spec_gain == Approx(401.0 / 301.0));  // borné par la marge pic / papier
}

TEST_CASE("sortie SDR : noirs et vrai noir au blanc de l'écran, sans marge d'éclats") {
  GradeSettings settings;
  settings.shadows = 0.4;
  settings.true_black = true;
  settings.black_lo = 0.1;
  settings.black_hi = 0.5;
  settings.chroma_base = 0.1;
  for (const core::HdrMode route : {core::HdrMode::Off, core::HdrMode::Native}) {
    auto context = pq_context(route);
    context.output_pq = false;
    const auto decision = decide_grade(settings, context);
    REQUIRE(decision);
    CHECK(decision->mode == GradeMode::Sdr);
    const GradeParams& p = decision->params;
    CHECK(p.sdr_out == 1.0);
    CHECK(p.peak == 203.0);
    CHECK(p.spec_gain == 1.0);
    CHECK(p.shadows == Approx(resp(0.4, 0.8, 1.2)));
    CHECK(p.chroma == Approx(1.1));
    CHECK(p.black_lo == 0.1);
    CHECK(p.black_hi == 0.5);
  }
  auto context = pq_context(core::HdrMode::Off);
  context.output_pq = false;
  context.source_hdr = true;
  CHECK(decide_grade(settings, context)->mode == GradeMode::Neutral);
}

TEST_CASE("le mode SDR sur une sortie HDR10 est gradé au blanc papier") {
  GradeSettings settings;
  const auto off = decide_grade(settings, pq_context(core::HdrMode::Off));
  REQUIRE(off);
  CHECK(off->mode == GradeMode::Native);
  CHECK(off->params.adapt == 0.0);
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

TEST_CASE("une source HDR n'est gradée que si grade_native_hdr le demande") {
  GradeSettings settings;
  auto context = pq_context(core::HdrMode::Native);
  context.source_hdr = true;
  CHECK(decide_grade(settings, context)->mode == GradeMode::Neutral);
  settings.grade_native_hdr = true;
  const auto decision = decide_grade(settings, context);
  REQUIRE(decision);
  CHECK(decision->mode == GradeMode::NativeHdr);
  CHECK(decision->params.adapt == 0.0);
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
  settings.paper_white = 79.0;
  auto refused = decide_grade(settings, pq_context(core::HdrMode::Native));
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
  CHECK(refused.error().detail.starts_with("grade.paper_white"));
  settings = GradeSettings{};
  settings.shadows = std::numeric_limits<double>::quiet_NaN();
  CHECK_FALSE(check_grade_settings(settings));
  settings = GradeSettings{};
  settings.area_rtx = 1.5;
  CHECK_FALSE(check_grade_settings(settings));
  CHECK_FALSE(decide_grade(GradeSettings{}, pq_context(core::HdrMode::Native, 99.0)));
}

TEST_CASE("pic du grade : celui de l'écran s'il est connu, 1000 nits sinon") {
  core::DisplayInfo display;
  CHECK(*grade_peak(display) == 1000.0);
  display.peak_nits = 1500.0;
  display.provenance.peak = core::Provenance::Profile;
  CHECK(*grade_peak(display) == 1500.0);
  display.peak_nits = 50.0;
  CHECK_FALSE(grade_peak(display));
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
  p.sdr_out = 1.0;
  p.fs_apl = 10.0;        // scène sombre : toute la marge
  p.look_mixes[0] = 1.0;  // la pile d'HDR
  p.look_mixes[10] = 0.5;
  p.look_pq = 1.0;
  sdr_tonemap(p);
  CHECK(p.look_mixes[0] == 0.0);
  CHECK(p.look_mixes[10] == 0.0);
  CHECK(p.look_pq == 0.0);
  CHECK(p.sdr_tm == 1.0);
  CHECK(p.sdr_out == 0.0);
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

}  // namespace ajp::render
