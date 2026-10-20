#include "kernels/ref/gmfss.h"

#include <cmath>
#include <cstddef>
#include <format>

#include "kernels/ref/check.h"

namespace ajp::kernels::ref {

Result<std::vector<PlaneF>> soft_splat(const std::vector<PlaneF>& source, const SplatMotion& motion,
                                       float s) {
  if (source.empty()) return fail(ErrorCode::SizeMismatch, "splat : aucun plan source");
  if (!(s >= 0.0F && s <= 1.0F)) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("splat : instant {} hors de [0, 1]", s));
  }
  const PlaneF& first = source.front();
  if (auto ok = check_plane(first, "splat : source"); !ok) return std::unexpected(ok.error());
  for (const PlaneF& p : source) {
    if (auto ok = check_same_size(first, p, "splat : source"); !ok) return std::unexpected(ok.error());
  }
  for (const PlaneF& p : motion.planes) {
    if (auto ok = check_same_size(first, p, "splat : mouvement"); !ok) return std::unexpected(ok.error());
  }
  const int w = first.width;
  const int h = first.height;
  std::vector<PlaneF> sums(source.size(), PlaneF(w, h));
  PlaneF weights(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float fx = s * motion.planes[0].at(x, y);
      const float fy = s * motion.planes[1].at(x, y);
      const float scale = std::exp(s * motion.planes[2].at(x, y));
      const float tx = static_cast<float>(x) + fx;
      const float ty = static_cast<float>(y) + fy;
      if (!std::isfinite(tx) || !std::isfinite(ty)) continue;
      const int x0 = static_cast<int>(std::floor(tx));
      const int y0 = static_cast<int>(std::floor(ty));
      const float east = tx - static_cast<float>(x0);
      const float west = static_cast<float>(x0 + 1) - tx;
      const float south = ty - static_cast<float>(y0);
      const float north = static_cast<float>(y0 + 1) - ty;
      const std::array corners{std::array{x0, y0}, std::array{x0 + 1, y0}, std::array{x0, y0 + 1},
                               std::array{x0 + 1, y0 + 1}};
      const std::array<float, 4> share{west * north, east * north, west * south, east * south};
      for (std::size_t k = 0; k < corners.size(); ++k) {
        const auto [cx, cy] = corners[k];
        if (cx < 0 || cx >= w || cy < 0 || cy >= h) continue;
        for (std::size_t c = 0; c < source.size(); ++c)
          sums[c].at(cx, cy) += source[c].at(x, y) * scale * share[k];
        weights.at(cx, cy) += scale * share[k];
      }
    }
  }
  for (PlaneF& p : sums) {
    for (std::size_t i = 0; i < p.data.size(); ++i) p.data[i] /= weights.data[i] + kGmfssNormalizeEpsilon;
  }
  return sums;
}

}  // namespace ajp::kernels::ref
