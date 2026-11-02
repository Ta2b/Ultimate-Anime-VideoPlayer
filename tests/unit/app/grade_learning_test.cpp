// Apprentissage du grade adaptatif : la courbe rend ce que with_adaptive applique, « Cette scène » répartit
// une correction sur ses bouts, et l'ajustement retrouve la courbe d'un utilisateur à partir de ses
// corrections.
#include "app/play/grade_learning.h"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "app/play/chain.h"

namespace {

using namespace ajp::app::play;
using Catch::Approx;

constexpr std::int64_t kNow = 1'800'000'000;

GradeCurve calibrated() {
  return GradeCurve{.lo = 13.0,
                    .hi = 54.0,
                    .shadows = 0.15,
                    .shadows_bright = 0.60,
                    .shadows_glow = 0.20,
                    .colors = 0.30,
                    .colors_cut = 0.5,
                    .colors_glow = 0.05,
                    .colors_sky = 0.0,
                    .gray = 50.0,
                    .gray_bright = 30.0,
                    .gray_glow = 40.0};
}

// Une correction : la scène, et ce que la courbe voulue y rend.
LearnSample correction(const GradeCurve& wanted, double x, double glow_seen, std::int64_t at = kNow) {
  LearnSample s{.unix_s = at, .kind = LearnSample::Kind::Correction, .touched = kAllFamilies};
  s.scene = LearnScene{.x = x, .glow_seen = glow_seen};
  const SceneMix m = scene_mix(s.scene, wanted.lo, wanted.hi);
  s.look = look_at(wanted, m.t, m.glow, m.tv);
  return s;
}

}  // namespace

TEST_CASE("apprentissage : la courbe rend noirs, couleurs et gris moyen comme le grade adaptatif",
          "[app][grade][learning]") {
  PlaySettings s;
  s.grade_values.adaptive = true;
  set_curve(s, calibrated());
  for (const double t : {0.0, 0.3, 1.0})
    for (const double glow : {0.0, 0.6})
      for (const double tv : {0.0, 0.5}) {
        const PlaySettings applied = with_adaptive(s, t, tv, 0.0, glow);
        const SceneLook look = look_at(curve_of(s), t, glow, tv);
        CHECK(look.shadows == Approx(applied.grade_values.shadows));
        CHECK(look.colors == Approx(applied.grade_values.vibrance_rtx));
        CHECK(std::lround(look.gray) == applied.true_hdr_gray);
      }
  CHECK(curve_of(s) == calibrated());
}

TEST_CASE("apprentissage : « Cette scène » rend la valeur voulue, partagée selon la part de chaque scène",
          "[app][grade][learning]") {
  GradeCurve c = calibrated();
  // Une scène à moitié claire : les noirs sombre et clair bougent autant, l'éclat ne bouge pas.
  correct_scene(c, 0.5, 0.0, 0.0, kShadowsFamily, 0.50);
  CHECK(look_at(c, 0.5, 0.0, 0.0).shadows == Approx(0.50));
  CHECK(c.shadows - 0.15 == Approx(c.shadows_bright - 0.60));
  CHECK(c.shadows_glow == 0.20);
  CHECK(c.gray == 50.0);  // une seule famille
  // Le bout sombre en butée : le reste passe au bout clair.
  GradeCurve high = calibrated();
  high.shadows = 0.8;
  correct_scene(high, 0.2, 0.0, 0.0, kShadowsFamily, 0.8);
  CHECK(look_at(high, 0.2, 0.0, 0.0).shadows == Approx(0.8));
  CHECK(high.shadows == 0.8);
  CHECK(high.shadows_bright == Approx(0.8));
  // Gris moyen d'une scène pleinement avec éclat : seul son bout bouge.
  GradeCurve glow = calibrated();
  correct_scene(glow, 0.0, 1.0, 0.0, kGrayFamily, 55.0);
  CHECK(glow.gray_glow == Approx(55.0));
  CHECK(glow.gray == 50.0);
  CHECK(touched_families(calibrated(), glow) == kGrayFamily);
}

TEST_CASE("apprentissage : sans au moins deux corrections, rien n'est ajusté", "[app][grade][learning]") {
  const std::vector<LearnSample> one{correction(calibrated(), 0.05, 0.0)};
  CHECK_FALSE(learn_grade(one, calibrated(), kNow).has_value());
  LearnSample approval = correction(calibrated(), 0.5, 0.0);
  approval.kind = LearnSample::Kind::Approval;
  const std::vector<LearnSample> approvals(20, approval);
  CHECK_FALSE(learn_grade(approvals, calibrated(), kNow).has_value());
}

TEST_CASE("apprentissage : les corrections retrouvent la courbe voulue, seuils compris",
          "[app][grade][learning]") {
  // L'utilisateur veut des noirs plus forts dans le clair, un gris moyen plus bas dans le sombre, et des
  // scènes claires qui ne commencent qu'à 22 % (jusqu'à 64 %).
  GradeCurve wanted = calibrated();
  wanted.lo = 22.0;
  wanted.hi = 64.0;
  wanted.shadows_bright = 0.75;
  wanted.gray = 42.0;
  wanted.gray_glow = 46.0;
  std::vector<LearnSample> samples;
  for (const double x : {0.05, 0.12, 0.18, 0.25, 0.32, 0.40, 0.48, 0.55, 0.62, 0.70})
    samples.push_back(correction(wanted, x, 0.0));
  samples.push_back(correction(wanted, 0.08, 0.9));
  samples.push_back(correction(wanted, 0.10, 0.8));
  const auto report = learn_grade(samples, calibrated(), kNow);
  REQUIRE(report.has_value());
  CHECK(report->corrections == 12);
  CHECK(report->error_after < report->error_before / 3.0);
  const GradeCurve& c = report->curve;
  CHECK(std::abs(c.lo - wanted.lo) < std::abs(calibrated().lo - wanted.lo));
  CHECK(std::abs(c.hi - wanted.hi) < std::abs(calibrated().hi - wanted.hi));
  CHECK(c.shadows_bright == Approx(wanted.shadows_bright).margin(0.06));
  CHECK(c.gray == Approx(wanted.gray).margin(3.0));
  CHECK(c.gray_glow == Approx(wanted.gray_glow).margin(4.0));
  // Aucune scène de ciel bleu : sa vibrance reste celle de la calibration.
  CHECK(c.colors_sky == calibrated().colors_sky);
  CHECK(report->support_glow > 1.0);
}

TEST_CASE("apprentissage : une correction aberrante ne tord pas la courbe, la plus récente l'emporte",
          "[app][grade][learning]") {
  GradeCurve wanted = calibrated();
  wanted.shadows = 0.25;
  std::vector<LearnSample> samples;
  for (const double x : {0.03, 0.05, 0.07, 0.09, 0.11}) samples.push_back(correction(wanted, x, 0.0));
  GradeCurve slip = wanted;
  slip.shadows = 0.8;  // une main qui a glissé
  samples.push_back(correction(slip, 0.06, 0.0));
  auto report = learn_grade(samples, calibrated(), kNow);
  REQUIRE(report.has_value());
  CHECK(report->curve.shadows == Approx(0.25).margin(0.12));
  // La même scène corrigée deux fois : l'ancienne correction compte moitié moins. La courbe en cours porte
  // déjà la dernière (« Cette scène » l'a posée).
  GradeCurve before = calibrated();
  before.shadows = 0.30;
  GradeCurve after = calibrated();
  after.shadows = 0.50;
  const std::vector<LearnSample> twice{correction(before, 0.05, 0.0, kNow - 600),
                                       correction(after, 0.05, 0.0, kNow)};
  report = learn_grade(twice, after, kNow);
  REQUIRE(report.has_value());
  CHECK(report->curve.shadows > 0.42);
  CHECK(report->curve.shadows < 0.50);
}

TEST_CASE("apprentissage : adaptive-learning.tsv se relit tel qu'il s'écrit", "[app][grade][learning]") {
  LearnSample s{.unix_s = kNow, .kind = LearnSample::Kind::Calibration, .touched = kGrayFamily};
  s.scene = LearnScene{.x = 0.31234, .glow_seen = 0.25, .tv = 0.5, .force = 2.0};
  s.look = SceneLook{.shadows = 0.2, .colors = 0.1, .gray = 44.5};
  const std::string text =
      std::string(learn_header()) + learn_line(s) + "abîmée\t1\n" + learn_line(LearnSample{});
  const std::vector<LearnSample> back = parse_learn_samples(text);
  REQUIRE(back.size() == 2);
  CHECK(back[0].unix_s == kNow);
  CHECK(back[0].kind == LearnSample::Kind::Calibration);
  CHECK(back[0].touched == kGrayFamily);
  CHECK(back[0].scene.x == Approx(0.31234));
  CHECK(back[0].scene.force == 2.0);
  CHECK(back[0].look.gray == Approx(44.5));
  CHECK(back[1].kind == LearnSample::Kind::Correction);
  // La scène forcée de la calibration garde ses poids, quels que soient les seuils.
  const SceneMix m = scene_mix(back[0].scene, 10.0, 50.0);
  CHECK(m.t == 0.0);
  CHECK(m.glow == 1.0);
}

TEST_CASE("calibration : les huit plans, chacun à sa place, ajustent la courbe sur tous à la fois",
          "[app][grade][learning][calibration]") {
  // Les huit plans de la vidéo de calibration : deux sombres de luminosités différentes, deux avec éclat,
  // quatre clairs. L'utilisateur règle chacun à l'œil ; la courbe qu'il voulait sans le savoir en ressort.
  GradeCurve wanted = calibrated();
  wanted.lo = 18.0;
  wanted.shadows = 0.25;
  wanted.gray = 44.0;
  wanted.gray_bright = 26.0;
  wanted.shadows_glow = 0.10;
  std::vector<LearnSample> shots;
  for (const auto& [x, glow] :
       {std::pair{0.06, 0.0}, std::pair{0.16, 0.0}, std::pair{0.08, 0.9}, std::pair{0.10, 0.8},
        std::pair{0.45, 0.0}, std::pair{0.55, 0.0}, std::pair{0.62, 0.0}, std::pair{0.70, 0.0}}) {
    LearnSample s = correction(wanted, x, glow);
    s.kind = LearnSample::Kind::Calibration;
    shots.push_back(s);
  }
  const auto report = learn_grade(shots, calibrated(), kNow);
  REQUIRE(report.has_value());
  CHECK(report->corrections == 8);
  CHECK(report->error_after < report->error_before / 2.0);
  // Les deux plans sombres ne reçoivent pas les mêmes valeurs : chacun garde celle de sa luminosité.
  const GradeCurve& c = report->curve;
  const SceneMix dim = scene_mix(LearnScene{.x = 0.06}, c.lo, c.hi);
  const SceneMix dusk = scene_mix(LearnScene{.x = 0.16}, c.lo, c.hi);
  CHECK(dusk.t > dim.t);
  CHECK(c.gray == Approx(wanted.gray).margin(3.0));
  CHECK(c.gray_bright == Approx(wanted.gray_bright).margin(3.0));
}
