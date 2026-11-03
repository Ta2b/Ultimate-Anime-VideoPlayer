#include "render/curve.h"

#include <algorithm>
#include <cmath>

namespace ajp::render {
namespace {

constexpr double kM1 = 2610.0 / 16384.0;
constexpr double kM2 = 2523.0 / 4096.0 * 128.0;
constexpr double kC1 = 3424.0 / 4096.0;
constexpr double kC2 = 2413.0 / 4096.0 * 32.0;
constexpr double kC3 = 2392.0 / 4096.0 * 32.0;

}  // namespace

double curve_pq(double nits) {
  const double y = std::pow(std::clamp(nits, 0.0, 10000.0) / 10000.0, kM1);
  return std::pow((kC1 + kC2 * y) / (1.0 + kC3 * y), kM2);
}

double curve_nits(double pq) {
  const double p = std::pow(std::clamp(pq, 0.0, 1.0), 1.0 / kM2);
  return 10000.0 * std::pow(std::max(p - kC1, 0.0) / (kC2 - kC3 * p), 1.0 / kM1);
}

ShotCurve plan_curve(const CurveInput& in) {
  ShotCurve c;
  double e = in.user_e;
  if (in.series_apl_nits > 0.0 && in.shot_apl_nits > 0.0)
    e -= kCurveSlope * std::log2(in.shot_apl_nits / in.series_apl_nits);
  c.e = std::clamp(e, -kCurveMaxE, kCurveMaxE);
  c.gain = std::exp2(c.e);
  c.knee_in = 0.75 * 203.0;
  c.knee_out = c.knee_in * c.gain;
  double top = in.panel_peak_nits;
  if (in.area_limit_nits > 0.0) top = std::min(top, in.area_limit_nits);
  if (in.abl_limit_nits > 0.0) top = std::min(top, in.abl_limit_nits);
  c.top_in = std::max(in.content_max_nits, c.knee_in * 1.01);
  c.top_out = std::max(top, c.knee_out * 1.05);  // jamais sous le genou
  const double x0 = curve_pq(c.knee_in);
  const double y0 = curve_pq(c.knee_out);
  const double x1 = curve_pq(c.top_in);
  const double y1 = curve_pq(c.top_out);
  // Pente du gain constant au genou, en PQ (différence finie centrée).
  const double h = 1e-4;
  const double slope0 =
      (curve_pq(curve_nits(x0 + h) * c.gain) - curve_pq(curve_nits(x0 - h) * c.gain)) / (2.0 * h);
  const double d = (y1 - y0) / (x1 - x0);
  double m0 = slope0;
  double m1 = 0.0;
  if (d <= 0.0) {
    m0 = 0.0;
  } else {
    const double a = m0 / d;
    if (a > 3.0) m0 = 3.0 * d;  // Fritsch-Carlson : β = 0, la condition α ≤ 3 garde la monotonie
  }
  c.m0 = m0;
  c.m1 = m1;
  return c;
}

double map_luminance(const ShotCurve& c, double nits) {
  if (nits <= c.knee_in) return nits * c.gain;
  if (nits >= c.top_in) return c.top_out;
  const double x0 = curve_pq(c.knee_in);
  const double x1 = curve_pq(c.top_in);
  const double y0 = curve_pq(c.knee_out);
  const double y1 = curve_pq(c.top_out);
  const double w = x1 - x0;
  const double t = (curve_pq(nits) - x0) / w;
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double y = (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * w * c.m0 + (-2 * t3 + 3 * t2) * y1 +
                   (t3 - t2) * w * c.m1;
  return curve_nits(y);
}

}  // namespace ajp::render
