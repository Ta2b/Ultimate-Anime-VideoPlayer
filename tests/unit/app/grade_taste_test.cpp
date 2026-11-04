// Le goût du grade v2 (app/play/grade_taste.h, chantier 8.10) : table de migration, classes de scène,
// apprentissage, fichier v2 qui écarte les lignes d'avant, et le goût dans le planificateur et la courbe.
#include "app/play/grade_taste.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "app/play/grade_planner.h"
#include "render/curve.h"

using namespace ajp::app::play;
using Catch::Approx;

namespace {

constexpr std::int64_t kNow = 1'800'000'000;

std::size_t at(SceneClass c) {
  return static_cast<std::size_t>(c);
}

TasteSample sample(SceneClass scene, double e, double toe = 0.0, double vibrance = 1.0,
                   LearnSample::Kind kind = LearnSample::Kind::Correction) {
  TasteSample s{.unix_s = kNow, .kind = kind, .scene = scene};
  s.taste.e.at(at(scene)) = e;
  s.taste.toe = toe;
  s.taste.vibrance = vibrance;
  return s;
}

}  // namespace

TEST_CASE("grade_taste : la table de migration", "[app][learning]") {
  CHECK(migrate(GradeCurve{}) == GradeTaste{});  // les réglages par défaut : le goût neutre
  GradeCurve c;
  c.gray = 100.0;  // un diaphragme au-dessus : +0,5 borné à +0,25
  c.gray_bright = 30.0 * std::exp2(-0.2);
  c.gray_glow = 40.0 * std::exp2(0.2);
  c.shadows = 0.35;
  c.colors = 0.30;
  const GradeTaste t = migrate(c);
  CHECK(t.e_of(SceneClass::Dark) == Approx(0.25));
  CHECK(t.e_of(SceneClass::Glow) == Approx(0.1));
  CHECK(t.e_of(SceneClass::Bright) == Approx(-0.1));
  CHECK(t.e_of(SceneClass::Sky) == Approx(-0.1));
  CHECK(t.toe == Approx(0.1));
  CHECK(t.vibrance == Approx(1.15));
  c.colors = 1.0;  // la vibrance ne passe jamais 1,15, ni sous 1
  CHECK(migrate(c).vibrance == Approx(ajp::render::kCurveMaxVibrance));
  c.colors = 0.0;
  CHECK(migrate(c).vibrance == Approx(1.0));
}

TEST_CASE("grade_taste : classes de scène", "[app][learning]") {
  CHECK(class_of(SceneMix{.t = 0.0, .glow = 0.0, .tv = 0.0}) == SceneClass::Dark);
  CHECK(class_of(SceneMix{.t = 0.1, .glow = 0.8, .tv = 0.0}) == SceneClass::Glow);
  CHECK(class_of(SceneMix{.t = 0.9, .glow = 0.0, .tv = 0.0}) == SceneClass::Bright);
  CHECK(class_of(SceneMix{.t = 0.9, .glow = 0.0, .tv = 0.7}) == SceneClass::Sky);
  ajp::media::ShotStats dark{.mean = 0.10F, .p50 = 0.08F, .p995 = 0.2F};
  CHECK(class_of(dark, 0.30) == SceneClass::Dark);
  ajp::media::ShotStats glow = dark;
  glow.p995 = 0.9F;
  glow.bright = 0.01F;
  CHECK(class_of(glow, 0.30) == SceneClass::Glow);
  ajp::media::ShotStats bright{.mean = 0.45F, .p50 = 0.45F, .p995 = 0.9F};
  CHECK(class_of(bright, 0.30) == SceneClass::Bright);
  ajp::media::ShotStats sky = bright;
  sky.blue = 0.2F;
  CHECK(class_of(sky, 0.30) == SceneClass::Sky);
}

TEST_CASE("grade_taste : l'apprentissage, régularisé vers le goût en cours", "[app][learning]") {
  const std::vector<TasteSample> one{sample(SceneClass::Dark, 0.2)};
  CHECK_FALSE(learn_taste(one, {}, kNow).has_value());  // une seule correction : refus
  const std::vector<TasteSample> samples{
      sample(SceneClass::Dark, 0.2, 0.1, 1.1), sample(SceneClass::Dark, 0.2, 0.1, 1.1),
      sample(SceneClass::Bright, -0.2, 0.1, 1.1),
      sample(SceneClass::Bright, -0.2, 0.1, 1.1, LearnSample::Kind::Approval)};
  const auto report = learn_taste(samples, {}, kNow);
  REQUIRE(report.has_value());
  CHECK(report->corrections == 3);
  CHECK(report->approvals == 1);
  CHECK(report->taste.e_of(SceneClass::Dark) == Approx(0.4 / 3.0));  // (0 + 2 × 0,2) / (1 + 2)
  CHECK(report->taste.e_of(SceneClass::Bright) == Approx(-0.25 / 2.25));
  CHECK(report->taste.e_of(SceneClass::Glow) == 0.0);  // rien ne la contredit
  CHECK(report->taste.toe == Approx(0.325 / 4.25));
  CHECK(report->taste.vibrance > 1.0);
  CHECK(report->taste.vibrance <= ajp::render::kCurveMaxVibrance);
  // Un échantillon de 90 jours compte moitié moins.
  std::vector<TasteSample> old = samples;
  for (TasteSample& s : old) s.unix_s = kNow - 90 * 86'400;
  CHECK(std::abs(learn_taste(old, {}, kNow)->taste.e_of(SceneClass::Dark)) <
        std::abs(report->taste.e_of(SceneClass::Dark)));
}

TEST_CASE("grade_taste : le fichier v2 écarte les lignes d'avant la v2", "[app][learning]") {
  const TasteSample s = sample(SceneClass::Glow, -0.15, 0.05, 1.08, LearnSample::Kind::Calibration);
  std::string text(taste_header());
  text += taste_line(s);
  // Une ligne d'adaptive-learning.tsv (l'autre grade) et son en-tête : écartées, comptées.
  text += learn_header();
  text += "1800000000\tc\t4\t0.30000\t0.0000\t0.0000\t-1.0\t0.1500\t0.1000\t55.00\n";
  const TasteRead read = parse_taste_samples(text);
  REQUIRE(read.samples.size() == 1);
  CHECK(read.rejected == 2);
  const TasteSample& back = read.samples.front();
  CHECK(back.kind == LearnSample::Kind::Calibration);
  CHECK(back.scene == SceneClass::Glow);
  CHECK(back.taste.e_of(SceneClass::Glow) == Approx(-0.15));
  CHECK(back.taste.toe == Approx(0.05));
  CHECK(back.taste.vibrance == Approx(1.08));
  CHECK(taste_log_of("/x/adaptive-learning.tsv") == std::filesystem::path("/x/adaptive-learning-v2.tsv"));
}

TEST_CASE("grade_taste : le pied de la courbe", "[app][learning]") {
  ajp::render::CurveInput in;
  const ajp::render::ShotCurve flat = ajp::render::plan_curve(in);
  in.toe = 0.2;
  const ajp::render::ShotCurve deep = ajp::render::plan_curve(in);
  in.toe = 1.0;
  CHECK(ajp::render::plan_curve(in).toe == Approx(ajp::render::kCurveMaxToe));
  CHECK(ajp::render::map_luminance(deep, 2.0) < ajp::render::map_luminance(flat, 2.0));
  CHECK(ajp::render::map_luminance(deep, 2.0) ==
        Approx(ajp::render::map_luminance(flat, 2.0) * std::pow(0.1, 0.2)));
  CHECK(ajp::render::map_luminance(deep, 50.0) == ajp::render::map_luminance(flat, 50.0));
  double last = 0.0;
  for (double n = 0.01; n < 30.0; n *= 1.1) {  // toujours monotone
    const double out = ajp::render::map_luminance(deep, n);
    CHECK(out > last);
    last = out;
  }
}

TEST_CASE("grade_taste : le goût change les cuissons du planificateur", "[app][learning]") {
  GradePlan plan({.lut_size = 9});
  ajp::media::Shot dark;
  dark.start_us = 0;
  dark.end_us = 2'000'000;
  dark.first = {.mean = 0.10F, .p50 = 0.08F, .p995 = 0.2F, .apl_nits = 20.0F};
  dark.last = dark.first;
  dark.meta.apl_mean_nits = 20.0F;
  dark.meta.max_scl_nits = 1000.0F;
  ajp::media::Shot bright = dark;
  bright.start_us = 2'000'000;
  bright.end_us = 4'000'000;
  bright.first = {.mean = 0.8F, .p50 = 0.8F, .p995 = 0.9F, .apl_nits = 20.0F};
  bright.last = bright.first;
  plan.set_shots({dark, bright});
  const std::vector<BakeJob> before = plan.wanted(0);
  REQUIRE(before.size() == 2);
  CHECK(before[0].input.user_e == 0.0);
  GradeTaste taste;
  taste.e.at(at(SceneClass::Dark)) = 0.125;
  taste.e.at(at(SceneClass::Bright)) = -0.25;
  taste.toe = 0.1;
  taste.vibrance = 1.1;
  plan.set_taste(taste);
  const std::vector<BakeJob> after = plan.wanted(0);
  REQUIRE(after.size() == 2);
  CHECK(after[0].input.user_e == Approx(0.125));
  CHECK(after[1].input.user_e == Approx(-0.25));
  CHECK(after[0].input.toe == Approx(0.1));
  CHECK(after[0].input.vibrance == Approx(1.1));
  CHECK(after[0].key != before[0].key);
}
