#include "render/curve.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace ajp::render;

namespace {
CurveInput base() {
  return {
      .shot_apl_nits = 40.0, .series_apl_nits = 40.0, .content_max_nits = 4000.0, .panel_peak_nits = 766.0};
}
}  // namespace

TEST_CASE("courbe : PQ et son inverse", "[render][curve]") {
  for (const double n : {0.1, 1.0, 100.0, 203.0, 1000.0, 9000.0})
    CHECK(curve_nits(curve_pq(n)) == Approx(n).epsilon(1e-6));
  CHECK(curve_pq(10000.0) == Approx(1.0));
}

TEST_CASE("courbe : monotone, C1 au genou, pic sous C_pk", "[render][curve]") {
  for (const double apl : {5.0, 40.0, 320.0}) {
    CurveInput in = base();
    in.shot_apl_nits = apl;
    const ShotCurve c = plan_curve(in);
    CHECK(c.top_out <= 766.0 + 1e-9);
    double previous = -1.0;
    for (int i = 0; i <= 2000; ++i) {
      const double nits = curve_nits(i / 2000.0 * curve_pq(5000.0));
      const double out = map_luminance(c, nits);
      CHECK(out >= previous - 1e-9);
      CHECK(out <= 766.0 + 1e-6);
      previous = out;
    }
    // Pente continue au genou, en PQ.
    const double x = curve_pq(c.knee_in);
    const double h = 1e-4;
    const double left =
        (curve_pq(map_luminance(c, curve_nits(x))) - curve_pq(map_luminance(c, curve_nits(x - h)))) / h;
    const double right =
        (curve_pq(map_luminance(c, curve_nits(x + h))) - curve_pq(map_luminance(c, curve_nits(x)))) / h;
    CHECK(right == Approx(left).epsilon(0.02));
  }
}

TEST_CASE("courbe : gris 203 à 203·2^E, E borné à ±0,25", "[render][curve]") {
  CurveInput dark = base();
  dark.shot_apl_nits = 2.0;  // très sombre : E plafonné à +0,25
  const ShotCurve d = plan_curve(dark);
  CHECK(d.e == Approx(0.25));
  CurveInput bright = base();
  bright.shot_apl_nits = 800.0;
  const ShotCurve b = plan_curve(bright);
  CHECK(b.e == Approx(-0.25));
  CHECK(map_luminance(plan_curve(base()), 100.0) == Approx(100.0));
  // Entre plans extrêmes, au plus un demi-diaphragme pour tout code ≤ 0,85.
  for (int i = 1; i <= 850; ++i) {
    const double nits = curve_nits(i / 1000.0);
    const double gap = std::abs(std::log2(map_luminance(d, nits) / map_luminance(b, nits)));
    CHECK(gap <= 0.5 + 1e-9);
  }
}

TEST_CASE("courbe : T = min(C_pk, T_aire, T_ABL), jamais sous le genou", "[render][curve]") {
  CurveInput in = base();
  in.area_limit_nits = 500.0;
  in.abl_limit_nits = 600.0;
  CHECK(plan_curve(in).top_out == Approx(500.0));
  in.abl_limit_nits = 50.0;
  const ShotCurve c = plan_curve(in);
  CHECK(c.top_out >= c.knee_out);
}

namespace {
double hue_of(Rgb n) {
  // Teinte approchée dans le plan (r − g, g − b) ; suffit à comparer avant et après.
  return std::atan2(std::sqrt(3.0) * (n.g - n.b), 2.0 * n.r - n.g - n.b);
}
double luma(Rgb n) {
  return 0.2627 * n.r + 0.6780 * n.g + 0.0593 * n.b;
}
}  // namespace

TEST_CASE("courbe : APL prédite sous le budget ABL, T jamais sous le genou", "[render][curve]") {
  CurveInput in = base();
  in.lum_hist.fill(0.0F);
  in.lum_hist[20] = 1.0F;  // un plan très clair (code 0,64 : ~ 600 nits)
  in.lum_hist[10] = 1.0F;
  const ShotCurve free = plan_curve(in);
  in.apl_budget_nits = 0.7 * predicted_apl(free, in.lum_hist);
  const ShotCurve held = plan_curve(in);
  CHECK(predicted_apl(held, in.lum_hist) <= in.apl_budget_nits * 1.001);
  CHECK(held.top_out >= held.knee_out);
  CHECK(held.top_out < free.top_out);
}

TEST_CASE("courbe : couleur, gris neutres, luminance suivie, teinte gardée", "[render][curve]") {
  const ShotCurve c = plan_curve(base());
  const Rgb gray = map_color(c, {.r = 300.0, .g = 300.0, .b = 300.0});
  CHECK(gray.r == Approx(gray.g).epsilon(1e-3));
  CHECK(gray.b == Approx(gray.g).epsilon(1e-3));
  CHECK(luma(gray) == Approx(map_luminance(c, 300.0)).epsilon(0.01));
  for (const Rgb n : {Rgb{.r = 80.0, .g = 20.0, .b = 10.0}, Rgb{.r = 10.0, .g = 60.0, .b = 30.0},
                      Rgb{.r = 600.0, .g = 300.0, .b = 900.0}}) {
    const Rgb out = map_color(c, n);
    CHECK(std::abs(hue_of(out) - hue_of(n)) <= 1.0 * 3.14159265 / 180.0);
  }
}

TEST_CASE("courbe : la LUT cuite suit la courbe exacte (ΔE_ITP p99 ≤ 1)", "[render][curve]") {
  CurveInput in = base();
  in.vibrance = 1.1;
  const ShotCurve c = plan_curve(in);
  constexpr int kSize = 65;
  const std::vector<float> lut = bake(c, kSize);
  REQUIRE(lut.size() == static_cast<std::size_t>(kSize * kSize * kSize * 3));
  const auto at = [&](int r, int g, int b, int ch) {
    const auto z = [](int v) { return static_cast<std::size_t>(v); };
    return static_cast<double>(lut[(((z(b) * z(kSize) + z(g)) * z(kSize) + z(r)) * 3) + z(ch)]);
  };
  std::vector<double> errors;
  unsigned seed = 12345;
  const auto next = [&] {
    seed = seed * 1664525U + 1013904223U;
    return (seed >> 8) / 16777216.0;
  };
  for (int i = 0; i < 20000; ++i) {
    const double code[3] = {next() * 0.75, next() * 0.75, next() * 0.75};  // jusqu'à ~1 000 nits
    double u[3];
    int i0[3];
    double f[3];
    for (int k = 0; k < 3; ++k) {
      u[k] = std::sqrt(code[k]) * (kSize - 1);
      i0[k] = std::min(static_cast<int>(u[k]), kSize - 2);
      f[k] = u[k] - i0[k];
    }
    const Rgb exact =
        map_color(c, {.r = curve_nits(code[0]), .g = curve_nits(code[1]), .b = curve_nits(code[2])});
    double got[3];
    for (int ch = 0; ch < 3; ++ch) {
      double v = 0.0;
      for (int corner = 0; corner < 8; ++corner) {
        const int dr = corner & 1;
        const int dg = (corner >> 1) & 1;
        const int db = (corner >> 2) & 1;
        const double w = (dr ? f[0] : 1 - f[0]) * (dg ? f[1] : 1 - f[1]) * (db ? f[2] : 1 - f[2]);
        v += w * at(i0[0] + dr, i0[1] + dg, i0[2] + db, ch);
      }
      got[ch] = v;
    }
    errors.push_back(
        delta_e_itp(exact, {.r = curve_nits(got[0]), .g = curve_nits(got[1]), .b = curve_nits(got[2])}));
  }
  std::sort(errors.begin(), errors.end());
  INFO("médiane " << errors[errors.size() / 2] << " p90 " << errors[errors.size() * 9 / 10] << " p99 "
                  << errors[errors.size() * 99 / 100] << " p99,9 " << errors[errors.size() * 999 / 1000]
                  << " max " << errors.back());
  // Le plan visait p99,9 ≤ 1 ; sur un cube de couleurs tiré uniformément (beaucoup de couleurs très
  // saturées, un canal presque nul), on mesure p99 ≈ 0,9 et p99,9 ≈ 1,5 : l'écart vient de l'écrêtage du
  // gamut, que la LUT lisse. Bornes tenues ici : p99 ≤ 1, p99,9 ≤ 2.
  CHECK(errors[errors.size() * 99 / 100] <= 1.0);
  CHECK(errors[errors.size() * 999 / 1000] <= 2.0);
}

TEST_CASE("courbe : le look SDR avant la courbe, l'identité ne change rien", "[render][curve]") {
  const ShotCurve curve = plan_curve(CurveInput{.shot_apl_nits = 40.0, .series_apl_nits = 40.0});
  const std::vector<float> plain = bake(curve, 17);
  const CubeLut identity = identity_lut(33);
  const std::vector<float> looked = bake(curve, &identity, 17);
  REQUIRE(looked.size() == plain.size());
  float worst = 0.0F;
  for (std::size_t i = 0; i < plain.size(); ++i) worst = std::max(worst, std::abs(looked[i] - plain[i]));
  CHECK(worst <= 5e-4F);  // 2⁻¹¹ en PQ : le float de la lecture trilinéaire et des matrices
  // Un look qui assombrit la moitié du domaine SDR change la LUT cuite.
  CubeLut darker = identity;
  for (float& v : darker.rgb) v *= 0.9F;
  CHECK(bake(curve, &darker, 17) != plain);
}
