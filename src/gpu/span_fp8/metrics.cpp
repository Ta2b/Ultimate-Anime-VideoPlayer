#include "gpu/span_fp8/metrics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ajp::gpu::span_fp8 {
namespace {

constexpr std::size_t kWindow = 11;
constexpr double kSigma = 1.5;
constexpr double kC1 = 0.01 * 0.01;
constexpr double kC2 = 0.03 * 0.03;

std::array<double, kWindow> gaussian() {
  std::array<double, kWindow> w{};
  double sum = 0.0;
  for (std::size_t i = 0; i < kWindow; ++i) {
    const double x = static_cast<double>(i) - static_cast<double>(kWindow - 1) / 2.0;
    w[i] = std::exp(-(x * x) / (2.0 * kSigma * kSigma));
    sum += w[i];
  }
  for (double& v : w) v /= sum;
  return w;
}

}  // namespace

double psnr(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size() || a.empty()) return 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += d * d;
  }
  const double mse = sum / static_cast<double>(a.size());
  return mse <= 1e-10 ? 99.0 : std::min(99.0, 10.0 * std::log10(1.0 / mse));
}

double ssim(int width, int height, std::span<const float> a, std::span<const float> b) {
  const auto w_px = static_cast<std::size_t>(std::max(width, 0));
  const auto h_px = static_cast<std::size_t>(std::max(height, 0));
  const std::size_t plane = w_px * h_px;
  if (a.size() != 3 * plane || b.size() != a.size() || w_px < kWindow || h_px < kWindow) return 0.0;
  const std::array<double, kWindow> w = gaussian();
  const std::size_t out_w = w_px - kWindow + 1;
  const std::size_t out_h = h_px - kWindow + 1;
  constexpr std::size_t kQuantities = 5;  // x, y, xx, yy, xy
  // Les cinq grandeurs floutées horizontalement des 11 dernières lignes : le flou vertical n'a besoin que
  // d'elles, et l'image entière en double ne tient pas en mémoire pour rien.
  std::vector<double> ring(kWindow * kQuantities * out_w);
  std::vector<double> lx(w_px), ly(w_px);
  constexpr std::array<double, 3> kLuma{0.2126, 0.7152, 0.0722};
  double total = 0.0;
  for (std::size_t r = 0; r < h_px; ++r) {
    for (std::size_t c = 0; c < w_px; ++c) {
      double x = 0.0, y = 0.0;
      for (std::size_t ch = 0; ch < 3; ++ch) {
        x += kLuma[ch] * static_cast<double>(a[ch * plane + r * w_px + c]);
        y += kLuma[ch] * static_cast<double>(b[ch * plane + r * w_px + c]);
      }
      lx[c] = x;
      ly[c] = y;
    }
    double* row = &ring[(r % kWindow) * kQuantities * out_w];
    for (std::size_t c = 0; c < out_w; ++c) {
      double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
      for (std::size_t i = 0; i < kWindow; ++i) {
        const double x = lx[c + i], y = ly[c + i], k = w[i];
        sx += k * x;
        sy += k * y;
        sxx += k * x * x;
        syy += k * y * y;
        sxy += k * x * y;
      }
      row[0 * out_w + c] = sx;
      row[1 * out_w + c] = sy;
      row[2 * out_w + c] = sxx;
      row[3 * out_w + c] = syy;
      row[4 * out_w + c] = sxy;
    }
    if (r + 1 < kWindow) continue;
    const std::size_t first = r + 1 - kWindow;  // la ligne de sortie, dont la fenêtre finit à r
    for (std::size_t c = 0; c < out_w; ++c) {
      std::array<double, kQuantities> m{};
      for (std::size_t i = 0; i < kWindow; ++i) {
        const double* line = &ring[((first + i) % kWindow) * kQuantities * out_w];
        for (std::size_t q = 0; q < kQuantities; ++q) m[q] += w[i] * line[q * out_w + c];
      }
      const double mx = m[0], my = m[1];
      const double vxx = m[2] - mx * mx, vyy = m[3] - my * my, vxy = m[4] - mx * my;
      total += ((2 * mx * my + kC1) * (2 * vxy + kC2)) / ((mx * mx + my * my + kC1) * (vxx + vyy + kC2));
    }
  }
  return total / static_cast<double>(out_w * out_h);
}

}  // namespace ajp::gpu::span_fp8
