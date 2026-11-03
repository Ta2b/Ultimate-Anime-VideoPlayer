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

namespace {

// Les pentes Hermite pour un T donné (Fritsch-Carlson, β = 0 : α ≤ 3 garde la monotonie).
void shape(ShotCurve& c) {
  const double x0 = curve_pq(c.knee_in);
  const double y0 = curve_pq(c.knee_out);
  const double x1 = curve_pq(c.top_in);
  const double y1 = curve_pq(c.top_out);
  const double h = 1e-4;
  const double slope0 =
      (curve_pq(curve_nits(x0 + h) * c.gain) - curve_pq(curve_nits(x0 - h) * c.gain)) / (2.0 * h);
  const double d = (y1 - y0) / (x1 - x0);
  c.m0 = d <= 0.0 ? 0.0 : std::min(slope0, 3.0 * d);
  c.m1 = 0.0;
}

}  // namespace

ShotCurve plan_curve(const CurveInput& in) {
  ShotCurve c;
  double e = in.user_e;
  if (in.series_apl_nits > 0.0 && in.shot_apl_nits > 0.0)
    e -= kCurveSlope * std::log2(in.shot_apl_nits / in.series_apl_nits);
  c.e = std::clamp(e, -kCurveMaxE, kCurveMaxE);
  c.gain = std::exp2(c.e);
  c.vibrance = std::clamp(in.vibrance, 1.0, kCurveMaxVibrance);
  c.knee_in = 0.75 * 203.0;
  c.knee_out = c.knee_in * c.gain;
  double top = in.panel_peak_nits;
  if (in.area_limit_nits > 0.0) top = std::min(top, in.area_limit_nits);
  if (in.abl_limit_nits > 0.0) top = std::min(top, in.abl_limit_nits);
  c.top_in = std::max(in.content_max_nits, c.knee_in * 1.01);
  const double floor_top = c.knee_out * 1.05;  // jamais sous le genou
  c.top_out = std::max(top, floor_top);
  shape(c);
  // Budget ABL : T baissé par bissection tant que l'APL prédite le dépasse.
  if (in.apl_budget_nits > 0.0 && predicted_apl(c, in.lum_hist) > in.apl_budget_nits) {
    double lo = floor_top;
    double hi = c.top_out;
    for (int i = 0; i < 40; ++i) {
      c.top_out = 0.5 * (lo + hi);
      shape(c);
      (predicted_apl(c, in.lum_hist) > in.apl_budget_nits ? hi : lo) = c.top_out;
    }
    c.top_out = lo;
    shape(c);
  }
  return c;
}

double map_luminance(const ShotCurve& c, double nits) {
  if (nits <= 0.0) return 0.0;
  if (nits < 0.5) {  // pied : le gain revient vers 1 dans les noirs
    const double t = std::clamp((nits - 0.1) / 0.4, 0.0, 1.0);
    const double s = t * t * (3.0 - 2.0 * t);
    return nits * (1.0 + (c.gain - 1.0) * s);
  }
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

double predicted_apl(const ShotCurve& c, const std::array<float, kCurveHistBins>& hist) {
  double weight = 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < kCurveHistBins; ++i) {
    if (hist[i] <= 0.0F) continue;
    const double code = (static_cast<double>(i) + 0.5) / static_cast<double>(kCurveHistBins);
    weight += hist[i];
    sum += hist[i] * map_luminance(c, curve_nits(code));
  }
  return weight > 0.0 ? sum / weight : 0.0;
}

namespace {

struct Ictcp {
  double i = 0.0;
  double ct = 0.0;
  double cp = 0.0;
};

Ictcp to_ictcp(Rgb n) {
  const double l = (1688.0 * n.r + 2146.0 * n.g + 262.0 * n.b) / 4096.0;
  const double m = (683.0 * n.r + 2951.0 * n.g + 462.0 * n.b) / 4096.0;
  const double s = (99.0 * n.r + 309.0 * n.g + 3688.0 * n.b) / 4096.0;
  const double lp = curve_pq(l);
  const double mp = curve_pq(m);
  const double sp = curve_pq(s);
  return {.i = 0.5 * lp + 0.5 * mp,
          .ct = (6610.0 * lp - 13613.0 * mp + 7003.0 * sp) / 4096.0,
          .cp = (17933.0 * lp - 17390.0 * mp - 543.0 * sp) / 4096.0};
}

Rgb from_ictcp(Ictcp c) {
  // Inverse de la matrice ICtCp (BT.2100), puis de LMS.
  const double lp = c.i + 0.008609037 * c.ct + 0.111029625 * c.cp;
  const double mp = c.i - 0.008609037 * c.ct - 0.111029625 * c.cp;
  const double sp = c.i + 0.560031336 * c.ct - 0.320627175 * c.cp;
  const double l = curve_nits(lp);
  const double m = curve_nits(mp);
  const double s = curve_nits(sp);
  return {.r = 3.436606694 * l - 2.506452118 * m + 0.069845424 * s,
          .g = -0.791329556 * l + 1.983600406 * m - 0.192270850 * s,
          .b = -0.025949900 * l - 0.098913713 * m + 1.124863613 * s};
}

double luminance(Rgb n) {
  return 0.2627 * n.r + 0.6780 * n.g + 0.0593 * n.b;
}

// Le secteur des peaux : autour de la teinte ICtCp de la peau sRGB (255, 224, 196).
double skin_hue() {
  const auto lin = [](double v) {
    v /= 255.0;
    return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
  };
  const double r = lin(255.0);
  const double g = lin(224.0);
  const double b = lin(196.0);
  // sRGB (BT.709) vers BT.2020, à 100 nits.
  const Rgb n{.r = 100.0 * (0.6274 * r + 0.3293 * g + 0.0433 * b),
              .g = 100.0 * (0.0691 * r + 0.9195 * g + 0.0114 * b),
              .b = 100.0 * (0.0164 * r + 0.0880 * g + 0.8956 * b)};
  const Ictcp c = to_ictcp(n);
  return std::atan2(c.cp, c.ct);
}

}  // namespace

double delta_e_itp(Rgb a, Rgb b) {
  const Ictcp x = to_ictcp(a);
  const Ictcp y = to_ictcp(b);
  const double di = x.i - y.i;
  const double dt = 0.5 * (x.ct - y.ct);
  const double dp = x.cp - y.cp;
  return 720.0 * std::sqrt(di * di + dt * dt + dp * dp);
}

Rgb map_color(const ShotCurve& curve, Rgb n) {
  n = {.r = std::max(n.r, 0.0), .g = std::max(n.g, 0.0), .b = std::max(n.b, 0.0)};
  const double y = luminance(n);
  if (y <= 0.0) return {};
  const double k = map_luminance(curve, y) / y;
  const Ictcp in = to_ictcp(n);
  const Ictcp scaled = to_ictcp({.r = n.r * k, .g = n.g * k, .b = n.b * k});
  double chroma = in.i > 0.0 ? std::pow(scaled.i / in.i, 0.4) : 1.0;
  if (curve.vibrance > 1.0) {
    static const double skin = skin_hue();
    const double hue = std::atan2(in.cp, in.ct);
    double gap = std::abs(hue - skin);
    gap = std::min(gap, 2.0 * 3.14159265358979 - gap);
    const double t = std::clamp((gap - 0.25) / 0.45, 0.0, 1.0);  // 0 dans le secteur (±15°), plein à ±40°
    const double away = t * t * (3.0 - 2.0 * t);
    chroma *= 1.0 + (curve.vibrance - 1.0) * away;
  }
  const Rgb out = from_ictcp({.i = scaled.i, .ct = in.ct * chroma, .cp = in.cp * chroma});
  return {.r = std::max(out.r, 0.0), .g = std::max(out.g, 0.0), .b = std::max(out.b, 0.0)};
}

std::vector<float> bake(const ShotCurve& curve, int size) {
  std::vector<float> lut(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) *
                         static_cast<std::size_t>(size) * 3);
  const auto nits_at = [&](int i) {
    const double u = static_cast<double>(i) / static_cast<double>(size - 1);
    return curve_nits(u * u);  // indexée par √code
  };
  std::size_t at = 0;
  for (int b = 0; b < size; ++b) {
    for (int g = 0; g < size; ++g) {
      for (int r = 0; r < size; ++r) {
        const Rgb out = map_color(curve, {.r = nits_at(r), .g = nits_at(g), .b = nits_at(b)});
        lut[at++] = static_cast<float>(curve_pq(out.r));
        lut[at++] = static_cast<float>(curve_pq(out.g));
        lut[at++] = static_cast<float>(curve_pq(out.b));
      }
    }
  }
  return lut;
}

}  // namespace ajp::render
