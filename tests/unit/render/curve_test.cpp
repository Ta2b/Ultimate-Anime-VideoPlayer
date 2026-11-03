#include "render/curve.h"

#include <cmath>

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
