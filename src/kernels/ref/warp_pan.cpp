// Mouvement d'ensemble d'une paire (spec/warp_pan.h) : translation globale par Lucas-Kanade sur la luma.
// Les sommes se font en double ici ; les implémentations GPU somment en float par réduction : écarts de
// l'ordre de 1e-4 px, sous la tolérance des tests (tests/unit/kernels/warp_pan_test.cpp).
#include <algorithm>
#include <cmath>
#include <cstddef>

#include "kernels/ref/warp.h"

namespace kirari::kernels::ref {
namespace {

float luma_at(const RgbImage& img, int x, int y) {
  return kPanLumaR * img.planes[0].at(x, y) + kPanLumaG * img.planes[1].at(x, y) +
         kPanLumaB * img.planes[2].at(x, y);
}

float luma_bilinear(const RgbImage& img, float x, float y) {
  const int w = img.width();
  const int h = img.height();
  x = std::fmin(std::fmax(x, 0.0F), static_cast<float>(w - 1));
  y = std::fmin(std::fmax(y, 0.0F), static_cast<float>(h - 1));
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = std::min(x0 + 1, w - 1);
  const int y1 = std::min(y0 + 1, h - 1);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const float top = luma_at(img, x0, y0) * (1.0F - fx) + luma_at(img, x1, y0) * fx;
  const float bottom = luma_at(img, x0, y1) * (1.0F - fx) + luma_at(img, x1, y1) * fx;
  return top * (1.0F - fy) + bottom * fy;
}

}  // namespace

PanEstimate pan_estimate(const RgbImage& a, const RgbImage& b) {
  const int w = a.width();
  const int h = a.height();
  if (w < 16 || h < 16 || b.width() != w || b.height() != h) return {};
  constexpr int kPoints = kPanGrid * kPanGrid;
  float la[kPoints];
  float gx[kPoints];
  float gy[kPoints];
  int px[kPoints];
  int py[kPoints];
  for (int q = 0; q < kPoints; ++q) {
    px[q] = pan_point(q % kPanGrid, w);
    py[q] = pan_point(q / kPanGrid, h);
    la[q] = luma_at(a, px[q], py[q]);
    gx[q] = 0.5F * (luma_at(a, px[q] + 1, py[q]) - luma_at(a, px[q] - 1, py[q]));
    gy[q] = 0.5F * (luma_at(a, px[q], py[q] + 1) - luma_at(a, px[q], py[q] - 1));
  }
  PanEstimate e;
  bool valid = false;
  for (int it = 0; it < kPanIterations; ++it) {
    double sxx = 0.0;
    double sxy = 0.0;
    double syy = 0.0;
    double sxr = 0.0;
    double syr = 0.0;
    for (int q = 0; q < kPoints; ++q) {
      const float r =
          luma_bilinear(b, static_cast<float>(px[q]) + e.dx, static_cast<float>(py[q]) + e.dy) - la[q];
      sxx += gx[q] * gx[q];
      sxy += gx[q] * gy[q];
      syy += gy[q] * gy[q];
      sxr += gx[q] * r;
      syr += gy[q] * r;
    }
    const double det = sxx * syy - sxy * sxy;
    valid = det > static_cast<double>(kPanDetRatio) * (sxx + syy) * (sxx + syy) && det > 0.0;
    if (!valid) break;
    e.dx -= static_cast<float>((syy * sxr - sxy * syr) / det);
    e.dy -= static_cast<float>((sxx * syr - sxy * sxr) / det);
  }
  e.sure = valid && std::hypot(e.dx, e.dy) < kPanMaxShift;
  return e;
}

}  // namespace kirari::kernels::ref
