#include "kernels/ref/gmfss.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <tuple>
#include <utility>

#include "kernels/ref/check.h"
#include "kernels/ref/warp.h"

namespace kirari::kernels::ref {
namespace {

struct PixelMotion {
  float fx = 0.0F;
  float fy = 0.0F;
  float z = 0.0F;
};

// Le splat commun : le mouvement de chaque pixel source par motion_at, sommes et poids accumulés.
template <typename MotionAt>
Splatted splat_core(const std::vector<PlaneF>& source, float s, MotionAt motion_at) {
  const int w = source.front().width;
  const int h = source.front().height;
  std::vector<PlaneF> sums(source.size(), PlaneF(w, h));
  PlaneF weights(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const PixelMotion m = motion_at(x, y);
      // z borné (NaN : le bas de l'intervalle, comme fmaxf), destination dans ]−1, taille[.
      const float bounded_z =
          m.z > -kGmfssMetricLimit ? std::min(m.z, kGmfssMetricLimit) : -kGmfssMetricLimit;
      const float scale = std::exp(s * bounded_z);
      const float tx = static_cast<float>(x) + s * m.fx;
      const float ty = static_cast<float>(y) + s * m.fy;
      if (!(tx > -1.0F && tx < static_cast<float>(w) && ty > -1.0F && ty < static_cast<float>(h))) continue;
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
  Splatted out{.planes = std::move(sums), .coverage = PlaneF(w, h)};
  for (PlaneF& p : out.planes) {
    for (std::size_t i = 0; i < p.data.size(); ++i) p.data[i] /= weights.data[i] + kGmfssNormalizeEpsilon;
  }
  for (std::size_t i = 0; i < weights.data.size(); ++i)
    out.coverage.data[i] = weights.data[i] / (weights.data[i] + kGmfssNormalizeEpsilon);
  return out;
}

Result<void> check_source(const std::vector<PlaneF>& source, float s) {
  if (source.empty()) return fail(ErrorCode::SizeMismatch, "splat : aucun plan source");
  if (!(s >= 0.0F && s <= 1.0F)) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("splat : instant {} hors de [0, 1]", s));
  }
  if (auto ok = check_plane(source.front(), "splat : source"); !ok) return ok;
  for (const PlaneF& p : source) {
    if (auto ok = check_same_size(source.front(), p, "splat : source"); !ok) return ok;
  }
  return {};
}

Result<void> check_splatted(const Splatted& s, int w, int h, const char* what) {
  if (s.planes.size() != 3) return fail(ErrorCode::SizeMismatch, std::format("{} : 3 plans attendus", what));
  for (const PlaneF& p : s.planes) {
    if (p.width != w || p.height != h)
      return fail(ErrorCode::SizeMismatch,
                  std::format("{} : plan {}x{} pour {}x{}", what, p.width, p.height, w, h));
  }
  if (s.coverage.width != w || s.coverage.height != h)
    return fail(ErrorCode::SizeMismatch, std::format("{} : couverture de taille différente", what));
  return {};
}

}  // namespace

Result<std::vector<PlaneF>> soft_splat(const std::vector<PlaneF>& source, const SplatMotion& motion,
                                       float s) {
  if (auto ok = check_source(source, s); !ok) return std::unexpected(ok.error());
  for (const PlaneF& p : motion.planes) {
    if (auto ok = check_same_size(source.front(), p, "splat : mouvement"); !ok)
      return std::unexpected(ok.error());
  }
  Splatted out = splat_core(source, s, [&](int x, int y) {
    return PixelMotion{
        .fx = motion.planes[0].at(x, y), .fy = motion.planes[1].at(x, y), .z = motion.planes[2].at(x, y)};
  });
  return std::move(out.planes);
}

Result<Splatted> soft_splat_at(const std::vector<PlaneF>& source, const SplatMotion& motion, float s,
                               const MotionGrid& grid) {
  if (auto ok = check_source(source, s); !ok) return std::unexpected(ok.error());
  if (!(grid.scale > 0.0F) || !std::isfinite(grid.x) || !std::isfinite(grid.y))
    return fail(ErrorCode::ParameterOutOfRange, std::format("splat : échelle {} de la grille", grid.scale));
  for (const PlaneF& p : motion.planes) {
    if (auto ok = check_plane(p, "splat : mouvement"); !ok) return std::unexpected(ok.error());
    if (auto ok = check_same_size(motion.planes[0], p, "splat : mouvement"); !ok)
      return std::unexpected(ok.error());
  }
  const bool same = grid.scale == 1.0F && grid.x == 0.0F && grid.y == 0.0F &&
                    motion.planes[0].width == source.front().width &&
                    motion.planes[0].height == source.front().height;
  return splat_core(source, s, [&](int x, int y) {
    if (same) {
      return PixelMotion{
          .fx = motion.planes[0].at(x, y), .fy = motion.planes[1].at(x, y), .z = motion.planes[2].at(x, y)};
    }
    const float u = (static_cast<float>(x) + 0.5F) / grid.scale - 0.5F + grid.x;
    const float v = (static_cast<float>(y) + 0.5F) / grid.scale - 0.5F + grid.y;
    return PixelMotion{.fx = *sample(motion.planes[0], u, v) * grid.scale,
                       .fy = *sample(motion.planes[1], u, v) * grid.scale,
                       .z = *sample(motion.planes[2], u, v)};
  });
}

Result<Splatted> soft_pull_at(const std::vector<PlaneF>& source, const SplatMotion& motion, float s,
                              const MotionGrid& grid) {
  if (auto ok = check_source(source, s); !ok) return std::unexpected(ok.error());
  const int w = source.front().width;
  const int h = source.front().height;
  // Le flot sur la grille de la source, poussé avec sa métrique comme le serait la source.
  std::vector<PlaneF> flow(2, PlaneF(w, h));
  SplatMotion own{.planes = {PlaneF(w, h), PlaneF(w, h), PlaneF(w, h)}};
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const bool same = grid.scale == 1.0F && grid.x == 0.0F && grid.y == 0.0F &&
                        motion.planes[0].width == w && motion.planes[0].height == h;
      const float u =
          same ? static_cast<float>(x) : (static_cast<float>(x) + 0.5F) / grid.scale - 0.5F + grid.x;
      const float v =
          same ? static_cast<float>(y) : (static_cast<float>(y) + 0.5F) / grid.scale - 0.5F + grid.y;
      const float scale = same ? 1.0F : grid.scale;
      for (std::size_t c = 0; c < 3; ++c) {
        auto value = sample(motion.planes[c], u, v);
        if (!value) return std::unexpected(value.error());
        own.planes[c].at(x, y) = *value * (c < 2 ? scale : 1.0F);
      }
      flow[0].at(x, y) = own.planes[0].at(x, y);
      flow[1].at(x, y) = own.planes[1].at(x, y);
    }
  }
  auto pushed = soft_splat_at(flow, own, s, MotionGrid{});
  if (!pushed) return std::unexpected(pushed.error());
  Splatted out{.planes = std::vector<PlaneF>(source.size(), PlaneF(w, h)), .coverage = pushed->coverage};
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float px = static_cast<float>(x) - s * pushed->planes[0].at(x, y);
      const float py = static_cast<float>(y) - s * pushed->planes[1].at(x, y);
      for (std::size_t c = 0; c < source.size(); ++c)
        out.planes[c].at(x, y) = catmull_rom_sample(source[c], px, py);
    }
  }
  return out;
}

Result<std::vector<PlaneF>> gmfss_compose(const Splatted& high_a, const Splatted& high_b,
                                          const Splatted& low_a, const Splatted& low_b,
                                          const std::vector<PlaneF>& fusion, const Window& window, float t) {
  if (!(t >= 0.0F && t <= 1.0F))
    return fail(ErrorCode::ParameterOutOfRange, std::format("composition : instant {} hors de [0, 1]", t));
  if (low_a.planes.size() != 3 || high_a.planes.size() != 3)
    return fail(ErrorCode::SizeMismatch, "composition : 3 plans attendus");
  const int w = low_a.planes[0].width;
  const int h = low_a.planes[0].height;
  const int ow = high_a.planes[0].width;
  const int oh = high_a.planes[0].height;
  for (const auto& [s, sw, sh, what] :
       {std::tuple{&low_a, w, h, "composition : RIFE a"}, std::tuple{&low_b, w, h, "composition : RIFE b"},
        std::tuple{&high_a, ow, oh, "composition : travail a"},
        std::tuple{&high_b, ow, oh, "composition : travail b"}}) {
    if (auto ok = check_splatted(*s, sw, sh, what); !ok) return std::unexpected(ok.error());
  }
  if (ow < w || oh < h)
    return fail(ErrorCode::SizeMismatch,
                std::format("composition : sortie {}x{} plus petite que {}x{}", ow, oh, w, h));
  if (fusion.size() != 3) return fail(ErrorCode::SizeMismatch, "composition : fusion de 3 plans attendue");
  for (const PlaneF& p : fusion) {
    if (auto ok = check_same_size(fusion[0], p, "composition : fusion"); !ok)
      return std::unexpected(ok.error());
  }
  if (window.width != w || window.height != h)
    return fail(ErrorCode::SizeMismatch, "composition : fenêtre de la fusion de la taille de RIFE attendue");
  if (auto ok = sample(fusion[0], window, 0.0F, 0.0F); !ok) return std::unexpected(ok.error());
  const auto mix = [t](float p0, float c0, float p1, float c1) {
    const float w0 = (1.0F - t) * c0;
    const float w1 = t * c1;
    return (w0 * p0 + w1 * p1) / (w0 + w1 + kGmfssNormalizeEpsilon);
  };
  const float rx = static_cast<float>(w) / static_cast<float>(ow);
  const float ry = static_cast<float>(h) / static_cast<float>(oh);
  std::vector<PlaneF> out(3, PlaneF(ow, oh));
  for (int y = 0; y < oh; ++y) {
    for (int x = 0; x < ow; ++x) {
      const float xw = (static_cast<float>(x) + 0.5F) * rx - 0.5F;
      const float yw = (static_cast<float>(y) + 0.5F) * ry - 0.5F;
      const float ca = high_a.coverage.at(x, y);
      const float cb = high_b.coverage.at(x, y);
      const float la = *sample(low_a.coverage, xw, yw);
      const float lb = *sample(low_b.coverage, xw, yw);
      for (std::size_t c = 0; c < 3; ++c) {
        const float high = mix(high_a.planes[c].at(x, y), ca, high_b.planes[c].at(x, y), cb);
        const float low = mix(*sample(low_a.planes[c], xw, yw), la, *sample(low_b.planes[c], xw, yw), lb);
        const float fused = *sample(fusion[c], window, xw, yw);
        out[c].at(x, y) = std::clamp(high + fused - low, 0.0F, 1.0F);
      }
    }
  }
  return out;
}

Result<std::vector<PlaneF>> gmfss_still(const std::vector<PlaneF>& key, const std::vector<PlaneF>& a,
                                        const std::vector<PlaneF>& b, float t) {
  if (!(t >= 0.0F && t <= 1.0F))
    return fail(ErrorCode::ParameterOutOfRange, std::format("décor fixe : instant {} hors de [0, 1]", t));
  if (key.size() != 3 || a.size() != 3 || b.size() != 3)
    return fail(ErrorCode::SizeMismatch, "décor fixe : 3 plans attendus");
  for (std::size_t c = 0; c < 3; ++c) {
    if (auto ok = check_same_size(key[0], key[c], "décor fixe : clé"); !ok)
      return std::unexpected(ok.error());
    if (auto ok = check_same_size(key[0], a[c], "décor fixe : a"); !ok) return std::unexpected(ok.error());
    if (auto ok = check_same_size(key[0], b[c], "décor fixe : b"); !ok) return std::unexpected(ok.error());
  }
  const int w = key[0].width;
  const int h = key[0].height;
  std::vector<PlaneF> out = key;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float d = 0.0F;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int sx = std::clamp(x + dx, 0, w - 1);
          const int sy = std::clamp(y + dy, 0, h - 1);
          for (std::size_t c = 0; c < 3; ++c) d = std::max(d, std::abs(a[c].at(sx, sy) - b[c].at(sx, sy)));
        }
      }
      const float m = std::clamp((kGmfssStillHigh - d) / (kGmfssStillHigh - kGmfssStillLow), 0.0F, 1.0F);
      for (std::size_t c = 0; c < 3; ++c) {
        const float k = key[c].at(x, y);
        const float source = (1.0F - t) * a[c].at(x, y) + t * b[c].at(x, y);
        out[c].at(x, y) = std::clamp(k + m * (source - k), 0.0F, 1.0F);
      }
    }
  }
  return out;
}

}  // namespace kirari::kernels::ref
