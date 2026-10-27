// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:963-1096.
#include "kernels/ref/warp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <string_view>

#include "kernels/ref/check.h"

namespace ajp::kernels::ref {
namespace {

// Lecture bilinéaire relative au coin d'une fenêtre du plan, sans vérification : les appelants
// ont validé le plan et la fenêtre (check_window).
float bilinear_at(const PlaneF& p, const Window& area, float x, float y) {
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

float bilinear_at(const PlaneF& p, float x, float y) {
  return bilinear_at(p, Window{.x = 0, .y = 0, .width = p.width, .height = p.height}, x, y);
}

// Un plan conforme, et une fenêtre non vide incluse dans ce plan.
Result<void> check_window(const PlaneF& p, const Window& a, std::string_view what) {
  if (auto ok = check_plane(p, what); !ok) return ok;
  // Écrit sans a.x + a.width, qui pourrait déborder un int.
  if (a.x < 0 || a.y < 0 || a.width < 1 || a.height < 1 || a.width > p.width - a.x ||
      a.height > p.height - a.y) {
    return fail(ErrorCode::ParameterOutOfRange,
                std::format("{} : fenêtre {}x{} en ({}, {}) hors du plan {}x{}", what, a.width, a.height, a.x,
                            a.y, p.width, p.height));
  }
  return {};
}

// Déplacements vers a et vers b, et poids de a, pour un point de la grille du mouvement.
struct Displacement {
  float ax = 0.0F;
  float ay = 0.0F;
  float bx = 0.0F;
  float by = 0.0F;
  float weight = 0.0F;
};

// previous : mouvement de la paire précédente, pour garder les poses seulement.
using DisplacementAt = Displacement (*)(const Motion&, const Motion*, float, float, float);

// L'instant que RIFE a estimé : mouvement et masque tels quels.
Displacement estimated(const Motion& m, const Motion* /*previous*/, float u, float v, float /*t*/) {
  return {.ax = bilinear_at(m.flow[0], m.area, u, v),
          .ay = bilinear_at(m.flow[1], m.area, u, v),
          .bx = bilinear_at(m.flow[2], m.area, u, v),
          .by = bilinear_at(m.flow[3], m.area, u, v),
          .weight = bilinear_at(m.mask, m.area, u, v)};
}

// Un autre instant depuis le mouvement du milieu : ligne droite et loi de mélange ajustée.
Displacement from_middle(const Motion& m, const Motion* /*previous*/, float u, float v, float t) {
  const float a0x = bilinear_at(m.flow[0], m.area, u, v);
  const float a0y = bilinear_at(m.flow[1], m.area, u, v);
  const float a1x = bilinear_at(m.flow[2], m.area, u, v);
  const float a1y = bilinear_at(m.flow[3], m.area, u, v);
  const float u2 = u - ((1.0F - t) * a0x + t * a1x);
  const float v2 = v - ((1.0F - t) * a0y + t * a1y);
  const float tx = bilinear_at(m.flow[2], m.area, u2, v2) - bilinear_at(m.flow[0], m.area, u2, v2);
  const float ty = bilinear_at(m.flow[3], m.area, u2, v2) - bilinear_at(m.flow[1], m.area, u2, v2);
  const float mm =
      std::fmin(std::fmax(bilinear_at(m.mask, m.area, u2, v2), kMaskEpsilon), 1.0F - kMaskEpsilon);
  const float z =
      kBlendLawA * std::log(mm / (1.0F - mm)) + kBlendLawB * std::log((1.0F - t) / t) + kBlendLawC;
  return {.ax = -t * tx,
          .ay = -t * ty,
          .bx = (1.0F - t) * tx,
          .by = (1.0F - t) * ty,
          .weight = 1.0F / (1.0F + std::exp(-z))};
}

// L'instant réglé du pixel (u, v) puis from_middle à cet instant (spec/poses.h).
Displacement keeping_poses(const Motion& m, const Motion* previous, float u, float v, float t) {
  const Window& w = m.area;
  float tp = t;
  for (int k = 0; k < kPoseIterations; ++k) {
    const float u2 =
        u - ((1.0F - tp) * bilinear_at(m.flow[0], w, u, v) + tp * bilinear_at(m.flow[2], w, u, v));
    const float v2 =
        v - ((1.0F - tp) * bilinear_at(m.flow[1], w, u, v) + tp * bilinear_at(m.flow[3], w, u, v));
    const float dx = bilinear_at(m.flow[2], w, u2, v2) - bilinear_at(m.flow[0], w, u2, v2);
    const float dy = bilinear_at(m.flow[3], w, u2, v2) - bilinear_at(m.flow[1], w, u2, v2);
    float before = 0.0F;
    if (previous != nullptr) {
      const Motion& p = *previous;
      const float ax = u - tp * dx;
      const float ay = v - tp * dy;
      // Le point de la paire précédente qui arrive en pa : q = pa − f1'(q), deux pas depuis pa.
      float qx = ax;
      float qy = ay;
      for (int step = 0; step < 2; ++step) {
        const float nx = ax - bilinear_at(p.flow[2], p.area, qx, qy);
        const float ny = ay - bilinear_at(p.flow[3], p.area, qx, qy);
        qx = nx;
        qy = ny;
      }
      before = std::hypot(bilinear_at(p.flow[2], p.area, qx, qy) - bilinear_at(p.flow[0], p.area, qx, qy),
                          bilinear_at(p.flow[3], p.area, qx, qy) - bilinear_at(p.flow[1], p.area, qx, qy));
    }
    const float now = std::hypot(dx, dy);
    const float r = (before + kPoseEpsilon) / (before + now + 2.0F * kPoseEpsilon);
    tp = std::fmin(std::fmax(pose_time(t, r), kPoseTimeLimit), 1.0F - kPoseTimeLimit);
  }
  return from_middle(m, previous, u, v, tp);
}

Result<RgbImage> blend(const RgbImage& a, const RgbImage& b, const Motion& motion, const Motion* previous,
                       float t, DisplacementAt at) {
  if (auto ok = check_same_size(a, b, "images à mélanger"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_motion(motion); !ok) return std::unexpected(ok.error());
  const int w = a.width();
  const int h = a.height();
  // Où chaque pixel tombe dans la grille du mouvement, centres alignés.
  const float sx = static_cast<float>(w) / static_cast<float>(motion.area.width);
  const float sy = static_cast<float>(h) / static_cast<float>(motion.area.height);
  RgbImage out(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float fx = static_cast<float>(x);
      const float fy = static_cast<float>(y);
      const Displacement d = at(motion, previous, (fx + 0.5F) / sx - 0.5F, (fy + 0.5F) / sy - 0.5F, t);
      for (std::size_t c = 0; c < out.planes.size(); ++c) {
        const float va = bilinear_at(a.planes[c], fx + sx * d.ax, fy + sy * d.ay);
        const float vb = bilinear_at(b.planes[c], fx + sx * d.bx, fy + sy * d.by);
        out.planes[c].at(x, y) = va * d.weight + vb * (1.0F - d.weight);
      }
    }
  }
  return out;
}

}  // namespace

Result<float> sample(const PlaneF& p, float x, float y) {
  if (auto ok = check_plane(p, "plan échantillonné"); !ok) return std::unexpected(ok.error());
  return bilinear_at(p, x, y);
}

Result<float> sample(const PlaneF& p, const Window& area, float x, float y) {
  if (auto ok = check_window(p, area, "plan échantillonné"); !ok) return std::unexpected(ok.error());
  return bilinear_at(p, area, x, y);
}

Result<void> check_motion(const Motion& motion) {
  for (const PlaneF& p : motion.flow) {
    if (auto ok = check_same_size(p, motion.mask, "plans du mouvement"); !ok) return ok;
    if (auto ok = check_plane(p, "plans du mouvement"); !ok) return ok;
  }
  return check_window(motion.mask, motion.area, "plans du mouvement");
}

Result<RgbImage> warp_blend(const RgbImage& a, const RgbImage& b, const Motion& motion) {
  return blend(a, b, motion, nullptr, 0.5F, estimated);
}

Result<RgbImage> warp_blend_at(const RgbImage& a, const RgbImage& b, const Motion& motion, float t) {
  if (!(t > 0.0F && t < 1.0F)) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("instant {} hors de ]0, 1[", t));
  }
  return blend(a, b, motion, nullptr, t, t == 0.5F ? estimated : from_middle);
}

Result<RgbImage> warp_blend_poses(const RgbImage& a, const RgbImage& b, const Motion& motion,
                                  const Motion* previous, float t) {
  if (!(t > 0.0F && t < 1.0F)) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("instant {} hors de ]0, 1[", t));
  }
  if (previous != nullptr) {
    if (auto ok = check_motion(*previous); !ok) return std::unexpected(ok.error());
    if (auto ok = check_same_size(previous->mask, motion.mask, "mouvements de deux paires"); !ok)
      return std::unexpected(ok.error());
    const Window& p = previous->area;
    const Window& m = motion.area;
    if (p.x != m.x || p.y != m.y || p.width != m.width || p.height != m.height)
      return fail(ErrorCode::SizeMismatch, "mouvements de deux paires : fenêtres différentes");
  }
  return blend(a, b, motion, previous, t, keeping_poses);
}

Result<float> flow_max(const Motion& motion) {
  if (auto ok = check_motion(motion); !ok) return std::unexpected(ok.error());
  const Window& a = motion.area;
  std::array<unsigned int, kFlowBins> bins{};
  for (int y = a.y; y < a.y + a.height; ++y) {
    for (int x = a.x; x < a.x + a.width; ++x) {
      float m = 0.0F;
      for (const PlaneF& p : motion.flow) m = std::fmax(m, std::fabs(p.at(x, y)));
      ++bins[static_cast<std::size_t>(flow_bin(m))];
    }
  }
  return flow_percentile(bins.data(), static_cast<unsigned long long>(a.width) * static_cast<unsigned long long>(a.height));
}

}  // namespace ajp::kernels::ref
