// Parité de decide_grade() avec M.plan de hdr-grade.lua (lecteur actuel, commit 33ff79d) :
// les valeurs attendues sont celles que le script calcule pour les mêmes réglages.
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

}  // namespace

TEST_CASE("route native SDR avec émulation TrueHDR : les réglages par défaut du lecteur actuel") {
  const auto decision = decide_grade(GradeSettings{}, pq_context(core::HdrMode::Native));
  REQUIRE(decision);
  CHECK(decision->mode == GradeMode::Native);
  CHECK(decision->source_white_nits == 203.0);
  const GradeParams& p = decision->params;
  CHECK(p.peak == 1000.0);
  CHECK(p.paper == 500.0);  // 0,5 × pic avec l'émulation
  CHECK(p.in_max == 1000.0);
  CHECK(p.shadow_knee == Approx(0.25 * 203.0));
  CHECK(p.shadows == 0.15);
  CHECK(p.emulate == 1.0);
  CHECK(p.em_base == 200.0);
  CHECK(p.em_pow == Approx(6.0));
  CHECK(p.em_lo == 1.0);
  CHECK(p.em_hi == 1.0);
  CHECK(p.em_sat == 1.0);
  CHECK(p.area_gain == 1.0);
  CHECK(p.spec_gain == Approx(1.3));
  CHECK(p.vibrance == 0.10);
  CHECK(p.abl == 0.0);
  CHECK(p.gamut_p3 == 0.0);
}

TEST_CASE("route native SDR sans émulation : éclats jusqu'au pic depuis le blanc papier") {
  GradeSettings settings;
  settings.truehdr_look = false;
  const auto decision = decide_grade(settings, pq_context(core::HdrMode::Native));
  REQUIRE(decision);
  CHECK(decision->mode == GradeMode::Native);
  CHECK(decision->source_white_nits == 203.0);
  const GradeParams& p = decision->params;
  const double room = 1000.0 / 203.0;
  CHECK(p.paper == 203.0);
  CHECK(p.in_max == 203.0);
  CHECK(p.emulate == 0.0);
  CHECK(p.area_gain == Approx(1.25));
  CHECK(p.spec_gain == Approx(1.0 + 0.5 * (room - 1.0)));
  CHECK(p.vibrance == 0.15);
}

TEST_CASE("le blanc papier est plafonné à 0,75 × pic et arrondi") {
  GradeSettings settings;
  settings.truehdr_look = false;
  settings.paper_white = 400.0;
  settings.highlights = 1.0;
  const auto decision = decide_grade(settings, pq_context(core::HdrMode::Native, 401.0));
  REQUIRE(decision);
  CHECK(decision->params.paper == 301.0);  // floor(300,75 + 0,5)
  CHECK(decision->source_white_nits == 301.0);
  CHECK(decision->params.area_gain == Approx(401.0 / 301.0));  // 1,5 borné par la marge pic / papier
  CHECK(decision->params.spec_gain == Approx(401.0 / 301.0));
}

TEST_CASE("émulation : contraste, saturation et gris moyen aux bornes") {
  GradeSettings settings;
  settings.contrast = 200.0;
  settings.saturation = 0.0;
  settings.middle_gray = 10.0;
  const auto decision = decide_grade(settings, pq_context(core::HdrMode::Native, 400.0));
  REQUIRE(decision);
  const GradeParams& p = decision->params;
  CHECK(p.em_base == 40.0);
  CHECK(p.em_pow == Approx(6.0 * std::pow(5.0, 0.35)));
  CHECK(p.em_lo == Approx(1.38));
  CHECK(p.em_hi == Approx(1.28));
  CHECK(p.em_sat == Approx(0.81));
  settings.contrast = 0.0;
  settings.middle_gray = 100.0;
  const auto low = decide_grade(settings, pq_context(core::HdrMode::Native, 4000.0));
  REQUIRE(low);
  CHECK(low->params.em_base == 400.0);
  CHECK(low->params.em_pow == Approx(6.0 * std::pow(0.5, 0.35)));
  CHECK(low->params.em_lo == Approx(0.65));
  CHECK(low->params.em_hi == Approx(0.74));
}

TEST_CASE("route TrueHDR : les éclats montent à partir de la moitié du pic") {
  GradeSettings settings;
  settings.highlights_rtx = 1.0;
  const auto decision = decide_grade(settings, pq_context(core::HdrMode::TrueHdr, 800.0));
  REQUIRE(decision);
  CHECK(decision->mode == GradeMode::TrueHdr);
  CHECK(decision->params.paper == 400.0);
  CHECK(decision->params.in_max == 800.0);
  CHECK(decision->params.emulate == 0.0);
  CHECK(decision->params.spec_gain == Approx(2.2));
}

TEST_CASE("le grade reste neutre hors de ses conditions") {
  GradeSettings settings;
  auto context = pq_context(core::HdrMode::Native);
  context.output_pq = false;
  CHECK(decide_grade(settings, context)->mode == GradeMode::Neutral);
  context = pq_context(core::HdrMode::Native);
  context.enabled = false;
  CHECK(decide_grade(settings, context)->mode == GradeMode::Neutral);
  CHECK(decide_grade(settings, pq_context(core::HdrMode::Off))->mode == GradeMode::Neutral);
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
  CHECK(decision->params.emulate == 0.0);
  CHECK(decision->params.in_max == 1000.0);
}

TEST_CASE("OLED : limiteur des scènes claires et seuils du vrai noir") {
  GradeSettings settings;
  settings.black_lo = 0.05;
  settings.black_hi = 0.4;
  auto decision = decide_grade(settings, pq_context(core::HdrMode::Native));
  REQUIRE(decision);
  CHECK(decision->params.abl == 0.0);
  CHECK(decision->params.black_hi == 0.0);  // pas d'OLED : la dixième passe est sautée
  settings.oled = true;
  decision = decide_grade(settings, pq_context(core::HdrMode::Native));
  REQUIRE(decision);
  CHECK(decision->params.abl == Approx(0.6));
  CHECK(decision->params.black_lo == 0.05);
  CHECK(decision->params.black_hi == 0.4);
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
  settings.subtitle_white = 1001.0;
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

TEST_CASE("préréglages : Naturel est le grade du lecteur actuel, Éclatant pousse tout") {
  CHECK(grade_preset(GradePreset::Naturel) == GradeSettings{});
  const GradeSettings vivid = grade_preset(GradePreset::Eclatant);
  const GradeSettings natural;
  CHECK(vivid.highlights_rtx > natural.highlights_rtx);
  CHECK(vivid.vibrance_rtx > natural.vibrance_rtx);
  CHECK(vivid.shadows > natural.shadows);
  CHECK(vivid.guard > 0.0);
  CHECK(check_grade_settings(vivid).has_value());
  GradeSettings mine;
  mine.shadows = 0.7;
  CHECK(grade_preset(GradePreset::Custom, mine) == mine);
  CHECK(preset_key(GradePreset::Eclatant) == "eclatant");
}

TEST_CASE("Éclatant sur TrueHDR : aplats et éclats montent, le garde-fou passe au hook") {
  const auto natural = decide_grade(GradeSettings{}, pq_context(core::HdrMode::TrueHdr));
  const auto vivid = decide_grade(grade_preset(GradePreset::Eclatant), pq_context(core::HdrMode::TrueHdr));
  REQUIRE(natural);
  REQUIRE(vivid);
  CHECK(natural->params.area_gain == 1.0);
  CHECK(natural->params.guard == 0.0);
  CHECK(vivid->params.area_gain == Approx(1.175));
  CHECK(vivid->params.spec_gain == Approx(1.96));
  CHECK(vivid->params.guard == 0.8);
  CHECK(vivid->params.vibrance == 0.45);
  GradeSettings bad;
  bad.guard = 1.5;
  CHECK_FALSE(decide_grade(bad, pq_context(core::HdrMode::TrueHdr)));
}

}  // namespace ajp::render

namespace ajp::render {

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

}  // namespace ajp::render
