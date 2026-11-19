// L'ajustement de la caméra de la carte sur ses tuiles (kernels/ref/scene_fit.h).
#include "kernels/ref/scene_fit.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace kirari::kernels::ref::scene_fit {
namespace {

// Moindres carrés pondérés de f ≈ p0 + p1 x + p2 y, équations normales 3 × 3 par élimination de Gauss.
bool solve3(std::array<std::array<double, 4>, 3> m, std::array<double, 3>& out) {
  for (std::size_t col = 0; col < 3; ++col) {
    std::size_t pivot = col;
    for (std::size_t r = col + 1; r < 3; ++r) {
      if (std::abs(m[r][col]) > std::abs(m[pivot][col])) pivot = r;
    }
    if (std::abs(m[pivot][col]) < 1e-9) return false;
    std::swap(m[col], m[pivot]);
    for (std::size_t r = 0; r < 3; ++r) {
      if (r == col) continue;
      const double k = m[r][col] / m[col][col];
      for (std::size_t c = col; c < 4; ++c) m[r][c] -= k * m[col][c];
    }
  }
  for (std::size_t r = 0; r < 3; ++r) out[r] = m[r][3] / m[r][r];
  return true;
}

struct FitPoint {
  double x = 0.0;  // coordonnées centrées et réduites
  double y = 0.0;
  double dx = 0.0;
  double dy = 0.0;
  double w = 0.0;
};

bool fit_affine(std::span<const FitPoint> points, const std::vector<std::uint8_t>& keep, Displacement& out) {
  std::array<std::array<double, 4>, 3> mx{};
  std::array<std::array<double, 4>, 3> my{};
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (!keep[i]) continue;
    const FitPoint& p = points[i];
    const std::array<double, 3> basis{1.0, p.x, p.y};
    for (std::size_t r = 0; r < 3; ++r) {
      for (std::size_t c = 0; c < 3; ++c) {
        mx[r][c] += p.w * basis[r] * basis[c];
        my[r][c] += p.w * basis[r] * basis[c];
      }
      mx[r][3] += p.w * basis[r] * p.dx;
      my[r][3] += p.w * basis[r] * p.dy;
    }
  }
  std::array<double, 3> px{};
  std::array<double, 3> py{};
  if (!solve3(mx, px) || !solve3(my, py)) return false;
  out.c = {px[0], px[1], px[2], py[0], py[1], py[2]};
  return true;
}

// Le seuil du rejet d'un tour : kFitMad écarts médians des tuiles à moins de kFitNear, borné ; unit : pixels
// du flux par pixel à 720p.
double reject_bound(const std::vector<double>& residuals, double unit) {
  std::vector<double> near;
  for (const double r : residuals) {
    if (r < scene::kFitNear * unit) near.push_back(r);
  }
  if (near.empty()) return scene::kFitBoundHigh * unit;
  return std::clamp(scene::kFitMad * median_of(std::move(near)), scene::kFitBoundLow * unit,
                    scene::kFitBoundHigh * unit);
}

}  // namespace

double median_of(std::vector<double> values) {
  if (values.empty()) return 0.0;
  const auto mid = static_cast<std::ptrdiff_t>(values.size() / 2);
  std::ranges::nth_element(values, values.begin() + mid);
  const double upper = values[static_cast<std::size_t>(mid)];
  if (values.size() % 2 == 1) return upper;
  return 0.5 * (upper + *std::ranges::max_element(values.begin(), values.begin() + mid));
}

// La caméra, sur les tuiles sûres, texturées et cohérentes (la méthode du banc 2.1, spec/scene_flow.h) :
// départ sur la translation médiane, puis kFitRounds tours de moindres carrés pondérés sur les tuiles sous le
// seuil du rejet (les dessins s'en vont).
CameraFit fit_camera(const std::vector<TileView>& views, double cx, double cy, double scale, double unit) {
  std::vector<double> costs;
  for (const TileView& v : views) {
    if (fit_candidate(v)) costs.push_back(v.cost);
  }
  const double typical = std::max(1.0, median_of(costs));
  std::vector<FitPoint> points;
  std::vector<std::size_t> tile_of;
  for (std::size_t i = 0; i < views.size(); ++i) {
    const TileView& v = views[i];
    if (!fit_candidate(v)) continue;
    points.push_back({.x = (v.cx - cx) / scale,
                      .y = (v.cy - cy) / scale,
                      .dx = v.mx,
                      .dy = v.my,
                      .w = texture_weight(v) / (1.0 + v.cost / typical)});
    tile_of.push_back(i);
  }
  const auto minimum =
      static_cast<std::size_t>(std::max(static_cast<double>(scene::kFitMinTiles),
                                        std::ceil(scene::kFitMinShare * static_cast<double>(views.size()))));
  if (points.size() < minimum) return {};
  CameraFit result;
  std::vector<double> xs;
  std::vector<double> ys;
  for (const FitPoint& p : points) {
    xs.push_back(p.dx);
    ys.push_back(p.dy);
  }
  result.model.c = {median_of(std::move(xs)), 0.0, 0.0, median_of(std::move(ys)), 0.0, 0.0};
  std::vector<std::uint8_t> keep(points.size(), 0U);
  std::vector<double> residuals(points.size());
  const auto off = [&](const FitPoint& p) {
    return std::hypot(p.dx - result.model.dx(p.x, p.y), p.dy - result.model.dy(p.x, p.y));
  };
  for (int round = 0; round < scene::kFitRounds; ++round) {
    for (std::size_t i = 0; i < points.size(); ++i) residuals[i] = off(points[i]);
    const double bound = reject_bound(residuals, unit);
    std::size_t kept = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
      keep[i] = residuals[i] < bound ? 1U : 0U;
      kept += keep[i];
    }
    if (kept < minimum || !fit_affine(points, keep, result.model)) return {};
  }
  double sum = 0.0;
  double weight = 0.0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (!keep[i]) continue;
    sum += points[i].w * off(points[i]);
    weight += points[i].w;
  }
  result.residual = weight > 0.0 ? sum / weight : 0.0;
  result.decor.assign(views.size(), 0U);
  for (std::size_t i = 0; i < points.size(); ++i) result.decor[tile_of[i]] = keep[i];
  result.fit = true;
  return result;
}

}  // namespace kirari::kernels::ref::scene_fit
