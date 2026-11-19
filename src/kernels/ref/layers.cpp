#include "kernels/ref/layers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>

#include "kernels/ref/warp.h"

namespace kirari::kernels::ref {
namespace {

constexpr float kLumaR = 0.2126F;
constexpr float kLumaG = 0.7152F;
constexpr float kLumaB = 0.0722F;

float luma(const std::array<float, 3>& v) {
  return kLumaR * v[0] + kLumaG * v[1] + kLumaB * v[2];
}

float bilinear(const PlaneF& p, float x, float y) {
  const int w = p.width;
  const int h = p.height;
  x = std::fmin(std::fmax(x, 0.0F), static_cast<float>(w - 1));
  y = std::fmin(std::fmax(y, 0.0F), static_cast<float>(h - 1));
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = std::min(x0 + 1, w - 1);
  const int y1 = std::min(y0 + 1, h - 1);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const float top = p.at(x0, y0) * (1.0F - fx) + p.at(x1, y0) * fx;
  const float bottom = p.at(x0, y1) * (1.0F - fx) + p.at(x1, y1) * fx;
  return top * (1.0F - fy) + bottom * fy;
}

bool grid_ok(const LayerGrid& g) {
  return g.fits(g.cols, g.rows) && g.tile_w > 0.0F && g.tile_h > 0.0F;
}

}  // namespace

float layer_mask(const LayerGrid& grid, float x, float y) {
  const float u = std::clamp((x + 0.5F) / grid.tile_w - 0.5F, 0.0F, static_cast<float>(grid.cols - 1));
  const float v = std::clamp((y + 0.5F) / grid.tile_h - 0.5F, 0.0F, static_cast<float>(grid.rows - 1));
  const int c0 = static_cast<int>(u);
  const int r0 = static_cast<int>(v);
  const int c1 = std::min(c0 + 1, grid.cols - 1);
  const int r1 = std::min(r0 + 1, grid.rows - 1);
  const float fu = u - static_cast<float>(c0);
  const float fv = v - static_cast<float>(r0);
  const auto bit = [&](int c, int r) { return grid.get(c, r) ? 1.0F : 0.0F; };
  const float g = (bit(c0, r0) * (1.0F - fu) + bit(c1, r0) * fu) * (1.0F - fv) +
                  (bit(c0, r1) * (1.0F - fu) + bit(c1, r1) * fu) * fv;
  return std::clamp(2.0F * g - 1.0F, 0.0F, 1.0F);
}

namespace {

// Un pixel (étapes 2 à 5 de la spec), m déjà calculé et non nul.
void compose_pixel(RgbImage& out, const std::array<const RgbImage*, 2>& images, const LayerParams& params,
                   int x, int y, float m) {
  const auto fx = static_cast<float>(x);
  const auto fy = static_cast<float>(y);
  const float wmax = static_cast<float>(out.width() - 1);
  const float hmax = static_cast<float>(out.height() - 1);
  std::array<std::array<float, 3>, 2> value{};
  std::array<bool, 2> inside{};
  std::array<float, 3> sum{};
  float total = 0.0F;
  for (std::size_t i = 0; i < static_cast<std::size_t>(params.count); ++i) {
    const auto& s = params.sources[i].shift;
    const float qx = fx + (s[0] + s[1] * fx + s[2] * fy);
    const float qy = fy + (s[3] + s[4] * fx + s[5] * fy);
    inside[i] = qx >= 0.0F && qx <= wmax && qy >= 0.0F && qy <= hmax;
    if (!inside[i]) continue;
    float wi = params.sources[i].weight;
    if (i == 0 && params.source_grid.cols != 0) wi *= layer_mask(params.source_grid, qx, qy);
    for (std::size_t c = 0; c < 3; ++c) {
      const PlaneF& plane = images[i]->planes[c];
      value[i][c] = params.cubic ? catmull_rom_sample(plane, qx, qy) : bilinear(plane, qx, qy);
      sum[c] += wi * value[i][c];
    }
    total += wi;
  }
  if (total <= 0.0F) return;
  const std::array<float, 3> here{out.planes[0].at(x, y), out.planes[1].at(x, y), out.planes[2].at(x, y)};
  float d = 0.0F;
  if (params.count == 2 && inside[0] && inside[1]) {
    d = std::fabs(luma(value[0]) - luma(value[1]));
  } else if (params.count == 1) {
    d = std::fabs(luma(value[0]) - luma(here));
  }
  const float k =
      1.0F - std::clamp((d - params.guard_low) / (params.guard_high - params.guard_low), 0.0F, 1.0F);
  for (std::size_t c = 0; c < 3; ++c) {
    out.planes[c].at(x, y) = std::clamp(here[c] + m * k * (sum[c] / total - here[c]), 0.0F, 1.0F);
  }
}

}  // namespace

Result<void> compose_layers(RgbImage& out, const RgbImage& a, const RgbImage& b, const LayerParams& params) {
  const int w = out.width();
  const int h = out.height();
  if (params.count < 1 || params.count > 2)
    return fail(ErrorCode::ParameterOutOfRange, std::format("calques : {} sources", params.count));
  if (a.width() != w || a.height() != h || (params.count == 2 && (b.width() != w || b.height() != h)))
    return fail(ErrorCode::SizeMismatch, "calques : sources et sortie de tailles différentes");
  if (!grid_ok(params.grid) || (params.source_grid.cols != 0 && !grid_ok(params.source_grid)))
    return fail(ErrorCode::ParameterOutOfRange, "calques : grille vide ou trop grande");
  if (!(params.guard_high > params.guard_low))
    return fail(ErrorCode::ParameterOutOfRange, "calques : garde vide");
  const std::array<const RgbImage*, 2> images{&a, &b};
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float m = layer_mask(params.grid, static_cast<float>(x), static_cast<float>(y));
      if (m > 0.0F) compose_pixel(out, images, params, x, y, m);
    }
  }
  return {};
}

}  // namespace kirari::kernels::ref
