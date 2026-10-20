#include "kernels/ref/gmfss.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>

#include "kernels/ref/check.h"

namespace ajp::kernels::ref {
namespace {

// Lecture bilinéaire bornée au plan, centres des pixels aux entiers (bilinear_at de warp.cpp).
float bilinear(const PlaneF& p, const Window& area, float x, float y) {
  const int w = area.width;
  const int h = area.height;
  x = std::fmin(std::fmax(x, 0.0F), static_cast<float>(w - 1));
  y = std::fmin(std::fmax(y, 0.0F), static_cast<float>(h - 1));
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = std::min(x0 + 1, w - 1);
  const int y1 = std::min(y0 + 1, h - 1);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const float top = p.at(area.x + x0, area.y + y0) * (1.0F - fx) + p.at(area.x + x1, area.y + y0) * fx;
  const float bottom = p.at(area.x + x0, area.y + y1) * (1.0F - fx) + p.at(area.x + x1, area.y + y1) * fx;
  return top * (1.0F - fy) + bottom * fy;
}

float bilinear(const PlaneF& p, float x, float y) {
  return bilinear(p, Window{.x = 0, .y = 0, .width = p.width, .height = p.height}, x, y);
}

Result<void> check_splatted(const Splatted& s, int width, int height, std::string_view what) {
  if (s.planes.size() != 3) return fail(ErrorCode::SizeMismatch, std::format("{} : 3 plans attendus", what));
  for (const PlaneF& p : s.planes) {
    if (auto ok = check_plane(p, what); !ok) return ok;
    if (p.width != width || p.height != height) {
      return fail(ErrorCode::SizeMismatch,
                  std::format("{} : {}x{} au lieu de {}x{}", what, p.width, p.height, width, height));
    }
  }
  if (auto ok = check_plane(s.coverage, what); !ok) return ok;
  if (s.coverage.width != width || s.coverage.height != height) {
    return fail(ErrorCode::SizeMismatch, std::format("{} : couverture d'une autre taille", what));
  }
  return {};
}

}  // namespace

Result<Splatted> soft_splat_at(const std::vector<PlaneF>& source, const SplatMotion& motion, float s,
                               const MotionGrid& grid) {
  if (source.empty()) return fail(ErrorCode::SizeMismatch, "splat : aucun plan source");
  if (!(s >= 0.0F && s <= 1.0F)) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("splat : instant {} hors de [0, 1]", s));
  }
  if (!(grid.scale > 0.0F)) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("splat : échelle du mouvement {}", grid.scale));
  }
  const PlaneF& first = source.front();
  if (auto ok = check_plane(first, "splat : source"); !ok) return std::unexpected(ok.error());
  for (const PlaneF& p : source) {
    if (auto ok = check_same_size(first, p, "splat : source"); !ok) return std::unexpected(ok.error());
  }
  for (const PlaneF& p : motion.planes) {
    if (auto ok = check_plane(p, "splat : mouvement"); !ok) return std::unexpected(ok.error());
    if (auto ok = check_same_size(motion.planes[0], p, "splat : mouvement"); !ok)
      return std::unexpected(ok.error());
  }
  const int w = first.width;
  const int h = first.height;
  // Même grille, sans décalage ni échelle : le mouvement du pixel, tel quel.
  const bool direct = grid.scale == 1.0F && grid.x == 0.0F && grid.y == 0.0F && motion.planes[0].width == w &&
                      motion.planes[0].height == h;
  std::vector<PlaneF> sums(source.size(), PlaneF(w, h));
  PlaneF weights(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float fx = 0.0F;
      float fy = 0.0F;
      float z = 0.0F;
      if (direct) {
        fx = motion.planes[0].at(x, y);
        fy = motion.planes[1].at(x, y);
        z = motion.planes[2].at(x, y);
      } else {
        const float u = (static_cast<float>(x) + 0.5F) / grid.scale - 0.5F + grid.x;
        const float v = (static_cast<float>(y) + 0.5F) / grid.scale - 0.5F + grid.y;
        fx = grid.scale * bilinear(motion.planes[0], u, v);
        fy = grid.scale * bilinear(motion.planes[1], u, v);
        z = bilinear(motion.planes[2], u, v);
      }
      const float scale = std::exp(s * z);
      const float tx = static_cast<float>(x) + s * fx;
      const float ty = static_cast<float>(y) + s * fy;
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
  Splatted out{.planes = std::move(sums), .coverage = PlaneF(w, h)};
  for (PlaneF& p : out.planes) {
    for (std::size_t i = 0; i < p.data.size(); ++i) p.data[i] /= weights.data[i] + kGmfssNormalizeEpsilon;
  }
  for (std::size_t i = 0; i < weights.data.size(); ++i)
    out.coverage.data[i] = weights.data[i] / (weights.data[i] + kGmfssNormalizeEpsilon);
  return out;
}

Result<std::vector<PlaneF>> soft_splat(const std::vector<PlaneF>& source, const SplatMotion& motion,
                                       float s) {
  if (!source.empty()) {
    for (const PlaneF& p : motion.planes) {
      if (auto ok = check_same_size(source.front(), p, "splat : mouvement"); !ok)
        return std::unexpected(ok.error());
    }
  }
  auto out = soft_splat_at(source, motion, s, MotionGrid{});
  if (!out) return std::unexpected(out.error());
  return std::move(out->planes);
}

Result<std::vector<PlaneF>> gmfss_compose(const Splatted& high_a, const Splatted& high_b,
                                          const Splatted& low_a, const Splatted& low_b,
                                          const std::vector<PlaneF>& fusion, const Window& window, float t) {
  if (!(t >= 0.0F && t <= 1.0F)) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("composition : instant {} hors de [0, 1]", t));
  }
  if (low_a.planes.size() != 3 || high_a.planes.size() != 3) {
    return fail(ErrorCode::SizeMismatch, "composition : 3 plans attendus");
  }
  const int w = low_a.planes[0].width;
  const int h = low_a.planes[0].height;
  const int ow = high_a.planes[0].width;
  const int oh = high_a.planes[0].height;
  if (auto ok = check_splatted(low_a, w, h, "composition : travail a"); !ok)
    return std::unexpected(ok.error());
  if (auto ok = check_splatted(low_b, w, h, "composition : travail b"); !ok)
    return std::unexpected(ok.error());
  if (auto ok = check_splatted(high_a, ow, oh, "composition : agrandie a"); !ok)
    return std::unexpected(ok.error());
  if (auto ok = check_splatted(high_b, ow, oh, "composition : agrandie b"); !ok)
    return std::unexpected(ok.error());
  if (ow % w != 0 || oh % h != 0 || ow / w != oh / h || ow / w < 1) {
    return fail(ErrorCode::SizeMismatch,
                std::format("composition : sortie {}x{} pour un travail {}x{}, multiple entier attendu", ow,
                            oh, w, h));
  }
  if (fusion.size() != 3) return fail(ErrorCode::SizeMismatch, "composition : fusion de 3 plans attendue");
  for (const PlaneF& p : fusion) {
    if (auto ok = check_plane(p, "composition : fusion"); !ok) return std::unexpected(ok.error());
    if (auto ok = check_same_size(fusion[0], p, "composition : fusion"); !ok)
      return std::unexpected(ok.error());
  }
  if (window.x < 0 || window.y < 0 || window.width != w || window.height != h ||
      window.width > fusion[0].width - window.x || window.height > fusion[0].height - window.y) {
    return fail(
        ErrorCode::ParameterOutOfRange,
        std::format("composition : fenêtre {}x{} en ({}, {}) hors de la fusion {}x{} ou du travail {}x{}",
                    window.width, window.height, window.x, window.y, fusion[0].width, fusion[0].height, w,
                    h));
  }
  const float scale = static_cast<float>(ow / w);
  std::vector<PlaneF> out(3, PlaneF(ow, oh));
  const auto mix = [&](float p0, float c0, float p1, float c1) {
    const float w0 = (1.0F - t) * c0;
    const float w1 = t * c1;
    return (w0 * p0 + w1 * p1) / (w0 + w1 + kGmfssNormalizeEpsilon);
  };
  for (int y = 0; y < oh; ++y) {
    for (int x = 0; x < ow; ++x) {
      const float xw = (static_cast<float>(x) + 0.5F) / scale - 0.5F;
      const float yw = (static_cast<float>(y) + 0.5F) / scale - 0.5F;
      const float ca = high_a.coverage.at(x, y);
      const float cb = high_b.coverage.at(x, y);
      const float la = bilinear(low_a.coverage, xw, yw);
      const float lb = bilinear(low_b.coverage, xw, yw);
      for (int c = 0; c < 3; ++c) {
        const float high = mix(high_a.planes[static_cast<std::size_t>(c)].at(x, y), ca,
                               high_b.planes[static_cast<std::size_t>(c)].at(x, y), cb);
        const float low = mix(bilinear(low_a.planes[static_cast<std::size_t>(c)], xw, yw), la,
                              bilinear(low_b.planes[static_cast<std::size_t>(c)], xw, yw), lb);
        const float fused = bilinear(fusion[static_cast<std::size_t>(c)], window, xw, yw);
        out[static_cast<std::size_t>(c)].at(x, y) = std::clamp(high + fused - low, 0.0F, 1.0F);
      }
    }
  }
  return out;
}

}  // namespace ajp::kernels::ref
