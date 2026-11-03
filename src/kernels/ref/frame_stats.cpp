#include "kernels/ref/frame_stats.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <vector>

#include "kernels/ref/transfer.h"

namespace ajp::kernels::ref {
namespace {

namespace fs = frame_stats;

float clamp01(float v) {
  if (!(v > 0.0F)) return 0.0F;  // NaN compris
  return v < 1.0F ? v : 1.0F;
}

int bin_of(float unit, int bins) {
  return std::min(static_cast<int>(clamp01(unit) * static_cast<float>(bins)), bins - 1);
}

bool is_skin(const Rgb& e) {
  if (!(e.r > e.g && e.g > e.b && e.r >= fs::kSkinMinRed)) return false;
  const float span = e.r - e.b;
  const float hue = (e.g - e.b) / span;
  const float sat = span / e.r;
  return hue >= fs::kSkinHueLo && hue <= fs::kSkinHueHi && sat >= fs::kSkinSatLo && sat <= fs::kSkinSatHi;
}

struct Point {
  Rgb encoded;
  float luma = 0.0F;
  float maxrgb = 0.0F;  // cd/m²
};

// Le centile pct (pour cent) d'un histogramme de parts sur le code PQ, en cd/m², interpolé dans sa
// case.
float percentile_nits(const std::array<float, fs::kLinearBins>& parts, float pct) {
  const float want = pct / 100.0F;
  float below = 0.0F;
  for (int bin = 0; bin < fs::kLinearBins; ++bin) {
    const float part = parts[static_cast<std::size_t>(bin)];
    if (part > 0.0F && below + part >= want) {
      const float within = std::clamp((want - below) / part, 0.0F, 1.0F);
      return pq_eotf((static_cast<float>(bin) + within) / static_cast<float>(fs::kLinearBins));
    }
    below += part;
  }
  return pq_eotf(1.0F);
}

}  // namespace

Rgb frame_nits(const Rgb& encoded, FrameTransfer transfer) {
  const Rgb e{.r = clamp01(encoded.r), .g = clamp01(encoded.g), .b = clamp01(encoded.b)};
  switch (transfer) {
    case FrameTransfer::Sdr: {
      const auto light = [](float v) { return std::pow(v, bt1886::kGamma) * fs::kSdrWhiteNits; };
      return {.r = light(e.r), .g = light(e.g), .b = light(e.b)};
    }
    case FrameTransfer::Pq:
      return {.r = pq_eotf(e.r), .g = pq_eotf(e.g), .b = pq_eotf(e.b)};
    case FrameTransfer::Hlg: {
      const Rgb scene{.r = hlg_inverse_oetf(e.r), .g = hlg_inverse_oetf(e.g), .b = hlg_inverse_oetf(e.b)};
      return hlg_ootf(scene, HlgDisplay{.peak = fs::kHlgPeakNits, .gamma = hlg::kGammaAt1000});
    }
  }
  return {};
}

float ictcp_i(const Rgb& n) {
  const float l = (1688.0F * n.r + 2146.0F * n.g + 262.0F * n.b) / 4096.0F;
  const float m = (683.0F * n.r + 2951.0F * n.g + 462.0F * n.b) / 4096.0F;
  return 0.5F * pq_inverse_eotf(l) + 0.5F * pq_inverse_eotf(m);
}

Result<FrameCounts> frame_counts(const RgbImage& rgb, FrameTransfer transfer) {
  const int w = rgb.width(), h = rgb.height();
  if (w < 2 || h < 2 || w > kMaxDimension || h > kMaxDimension)
    return fail(ErrorCode::DimensionOutOfRange, std::format("analyse : image {}x{}", w, h));
  const Coefficients k = coefficients(transfer == FrameTransfer::Sdr ? Matrix::Bt709 : Matrix::Bt2020Ncl);
  const auto at = [](int gx, int gy) {
    return static_cast<std::size_t>(gy) * fs::kGridW + static_cast<std::size_t>(gx);
  };
  std::vector<Point> points(static_cast<std::size_t>(fs::kGridW) * fs::kGridH);
  std::array<float, fs::kGridW> column_max{};
  std::array<float, fs::kGridH> row_max{};
  for (int gy = 0; gy < fs::kGridH; ++gy) {
    const int sy = std::min((2 * gy + 1) * h / (2 * fs::kGridH), h - 1);
    for (int gx = 0; gx < fs::kGridW; ++gx) {
      const int sx = std::min((2 * gx + 1) * w / (2 * fs::kGridW), w - 1);
      const Rgb raw = rgb.at(sx, sy);
      Point& p = points[at(gx, gy)];
      p.encoded = {.r = clamp01(raw.r), .g = clamp01(raw.g), .b = clamp01(raw.b)};
      p.luma = 0.2126F * p.encoded.r + 0.7152F * p.encoded.g + 0.0722F * p.encoded.b;
      const Rgb n = frame_nits(p.encoded, transfer);
      p.maxrgb = std::max({n.r, n.g, n.b});
      auto& column = column_max[static_cast<std::size_t>(gx)];
      auto& row = row_max[static_cast<std::size_t>(gy)];
      column = std::max(column, p.luma);
      row = std::max(row, p.luma);
    }
  }
  int x0 = 0, x1 = fs::kGridW, y0 = 0, y1 = fs::kGridH;
  while (x0 < fs::kGridW && column_max[static_cast<std::size_t>(x0)] < fs::kBarLuma) ++x0;
  while (x1 > x0 && column_max[static_cast<std::size_t>(x1 - 1)] < fs::kBarLuma) --x1;
  while (y0 < fs::kGridH && row_max[static_cast<std::size_t>(y0)] < fs::kBarLuma) ++y0;
  while (y1 > y0 && row_max[static_cast<std::size_t>(y1 - 1)] < fs::kBarLuma) --y1;
  if (x1 - x0 < fs::kGridW / 4 || y1 - y0 < fs::kGridH / 4) x0 = 0, x1 = fs::kGridW, y0 = 0, y1 = fs::kGridH;

  FrameCounts c;
  c.x0 = static_cast<std::uint32_t>(x0), c.x1 = static_cast<std::uint32_t>(x1);
  c.y0 = static_cast<std::uint32_t>(y0), c.y1 = static_cast<std::uint32_t>(y1);
  // Sommes en double : l'ordre des additions du noyau (par fil, puis en arbre) ne se refait pas.
  double luma_sum = 0.0, blue_sum = 0.0, chroma_sum = 0.0, nits_sum = 0.0, maxrgb_sum = 0.0;
  std::array<double, static_cast<std::size_t>(fs::kZones) * fs::kZones> zone_sum{};
  for (int gy = y0; gy < y1; ++gy) {
    for (int gx = x0; gx < x1; ++gx) {
      const Point& p = points[at(gx, gy)];
      const Rgb& e = p.encoded;
      const Rgb n = frame_nits(e, transfer);
      const Rgb n2020 = transfer == FrameTransfer::Sdr ? convert_primaries(n, kBt709ToBt2020) : n;
      ++c.luma_hist[static_cast<std::size_t>(bin_of(p.luma, fs::kLumaBins))];
      // Le code PQ de maxRGB : en PQ, le plus grand canal encodé lui-même (l'EOTF est croissante).
      const float code =
          transfer == FrameTransfer::Pq ? std::max({e.r, e.g, e.b}) : pq_inverse_eotf(p.maxrgb);
      ++c.maxrgb_hist[static_cast<std::size_t>(bin_of(code, fs::kLinearBins))];
      ++c.i_hist[static_cast<std::size_t>(bin_of(ictcp_i(n2020), fs::kLinearBins))];
      const int zx = (gx - x0) * fs::kZones / (x1 - x0), zy = (gy - y0) * fs::kZones / (y1 - y0);
      const std::size_t zone = static_cast<std::size_t>(zy) * fs::kZones + static_cast<std::size_t>(zx);
      zone_sum[zone] += p.luma;
      ++c.zone_count[zone];
      ++c.points;
      if (p.maxrgb > fs::kBrightNits) {
        ++c.bright;
        float around = 0.0F;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            around += points[at(std::clamp(gx + dx, x0, x1 - 1), std::clamp(gy + dy, y0, y1 - 1))].maxrgb;
          }
        }
        if (p.maxrgb >= fs::kIsolatedRatio * (around / 8.0F)) ++c.highlights;
      }
      if (is_skin(e)) ++c.skin;
      luma_sum += p.luma;
      blue_sum += std::max(0.0F, e.b - std::max(e.r, e.g));
      chroma_sum += std::max({e.r, e.g, e.b}) - std::min({e.r, e.g, e.b});
      nits_sum += k.kr * n.r + (1.0F - k.kr - k.kb) * n.g + k.kb * n.b;
      maxrgb_sum += p.maxrgb;
    }
  }
  for (std::size_t i = 0; i < zone_sum.size(); ++i) c.zone_sum[i] = static_cast<float>(zone_sum[i]);
  c.luma_sum = static_cast<float>(luma_sum);
  c.blue_sum = static_cast<float>(blue_sum);
  c.chroma_sum = static_cast<float>(chroma_sum);
  c.nits_sum = static_cast<float>(nits_sum);
  c.maxrgb_sum = static_cast<float>(maxrgb_sum);
  return c;
}

FrameStats frame_stats_of(const FrameCounts& c) {
  FrameStats s;
  if (c.points == 0) return s;
  const auto n = static_cast<float>(c.points);
  for (std::size_t i = 0; i < s.luma_hist.size(); ++i)
    s.luma_hist[i] = static_cast<float>(c.luma_hist[i]) / n;
  for (std::size_t i = 0; i < s.maxrgb_hist.size(); ++i) {
    s.maxrgb_hist[i] = static_cast<float>(c.maxrgb_hist[i]) / n;
    s.i_hist[i] = static_cast<float>(c.i_hist[i]) / n;
  }
  for (std::size_t i = 0; i < s.zones.size(); ++i)
    s.zones[i] = c.zone_count[i] == 0 ? 0.0F : c.zone_sum[i] / static_cast<float>(c.zone_count[i]);
  for (std::size_t i = 0; i < fs::kPercentiles.size(); ++i)
    s.maxrgb_nits[i] = percentile_nits(s.maxrgb_hist, fs::kPercentiles[i]);
  s.mean = c.luma_sum / n;
  s.blue = c.blue_sum / n;
  s.chroma = c.chroma_sum / n;
  s.apl_nits = c.nits_sum / n;
  s.maxrgb_mean_nits = c.maxrgb_sum / n;
  s.bright = static_cast<float>(c.bright) / n;
  s.highlights = static_cast<float>(c.highlights) / n;
  s.skin = static_cast<float>(c.skin) / n;
  s.left = static_cast<float>(c.x0) / fs::kGridW;
  s.right = static_cast<float>(c.x1) / fs::kGridW;
  s.top = static_cast<float>(c.y0) / fs::kGridH;
  s.bottom = static_cast<float>(c.y1) / fs::kGridH;
  return s;
}

}  // namespace ajp::kernels::ref
