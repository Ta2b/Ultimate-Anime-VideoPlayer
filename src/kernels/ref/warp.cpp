// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:963-1096.
#include "kernels/ref/warp.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>

#include "kernels/ref/check.h"

namespace ajp::kernels::ref {
namespace {

// Lecture bilinéaire relative au coin (x0w, y0w) d'une zone w × h du plan.
float bilinear_at(const PlaneF& p, int x0w, int y0w, int w, int h, float x, float y) {
  x = std::fmin(std::fmax(x, 0.0F), static_cast<float>(w - 1));
  y = std::fmin(std::fmax(y, 0.0F), static_cast<float>(h - 1));
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = std::min(x0 + 1, w - 1);
  const int y1 = std::min(y0 + 1, h - 1);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const float top = p.at(x0w + x0, y0w + y0) * (1.0F - fx) + p.at(x0w + x1, y0w + y0) * fx;
  const float bottom = p.at(x0w + x0, y0w + y1) * (1.0F - fx) + p.at(x0w + x1, y0w + y1) * fx;
  return top * (1.0F - fy) + bottom * fy;
}

// Déplacements vers a et vers b, et poids de a, pour un point de la grille du mouvement.
struct Displacement {
  float ax = 0.0F;
  float ay = 0.0F;
  float bx = 0.0F;
  float by = 0.0F;
  float weight = 0.0F;
};

using DisplacementAt = Displacement (*)(const Motion&, float, float, float);

// L'instant que RIFE a estimé : mouvement et masque tels quels.
Displacement estimated(const Motion& m, float u, float v, float /*t*/) {
  return {.ax = sample(m.flow[0], m.area, u, v),
          .ay = sample(m.flow[1], m.area, u, v),
          .bx = sample(m.flow[2], m.area, u, v),
          .by = sample(m.flow[3], m.area, u, v),
          .weight = sample(m.mask, m.area, u, v)};
}

// Un autre instant depuis le mouvement du milieu : ligne droite et loi de mélange ajustée.
Displacement from_middle(const Motion& m, float u, float v, float t) {
  const float a0x = sample(m.flow[0], m.area, u, v);
  const float a0y = sample(m.flow[1], m.area, u, v);
  const float a1x = sample(m.flow[2], m.area, u, v);
  const float a1y = sample(m.flow[3], m.area, u, v);
  const float u2 = u - ((1.0F - t) * a0x + t * a1x);
  const float v2 = v - ((1.0F - t) * a0y + t * a1y);
  const float tx = sample(m.flow[2], m.area, u2, v2) - sample(m.flow[0], m.area, u2, v2);
  const float ty = sample(m.flow[3], m.area, u2, v2) - sample(m.flow[1], m.area, u2, v2);
  const float mm = std::fmin(std::fmax(sample(m.mask, m.area, u2, v2), kMaskEpsilon), 1.0F - kMaskEpsilon);
  const float z =
      kBlendLawA * std::log(mm / (1.0F - mm)) + kBlendLawB * std::log((1.0F - t) / t) + kBlendLawC;
  return {.ax = -t * tx,
          .ay = -t * ty,
          .bx = (1.0F - t) * tx,
          .by = (1.0F - t) * ty,
          .weight = 1.0F / (1.0F + std::exp(-z))};
}

Result<RgbImage> blend(const RgbImage& a, const RgbImage& b, const Motion& motion, float t,
                       DisplacementAt at) {
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
      const Displacement d = at(motion, (fx + 0.5F) / sx - 0.5F, (fy + 0.5F) / sy - 0.5F, t);
      for (std::size_t c = 0; c < out.planes.size(); ++c) {
        const float va = sample(a.planes[c], fx + sx * d.ax, fy + sy * d.ay);
        const float vb = sample(b.planes[c], fx + sx * d.bx, fy + sy * d.by);
        out.planes[c].at(x, y) = va * d.weight + vb * (1.0F - d.weight);
      }
    }
  }
  return out;
}

}  // namespace

float sample(const PlaneF& p, float x, float y) {
  return bilinear_at(p, 0, 0, p.width, p.height, x, y);
}

float sample(const PlaneF& p, const Window& area, float x, float y) {
  return bilinear_at(p, area.x, area.y, area.width, area.height, x, y);
}

Result<void> check_motion(const Motion& motion) {
  for (const PlaneF& p : motion.flow) {
    if (auto ok = check_same_size(p, motion.mask, "plans du mouvement"); !ok) return ok;
  }
  if (auto ok = check_plane(motion.mask, "plans du mouvement"); !ok) return ok;
  const Window& a = motion.area;
  if (a.x < 0 || a.y < 0 || a.width < 1 || a.height < 1 || a.x + a.width > motion.mask.width ||
      a.y + a.height > motion.mask.height) {
    return fail(ErrorCode::ParameterOutOfRange,
                std::format("fenêtre {}x{} en ({}, {}) hors des plans {}x{}", a.width, a.height, a.x, a.y,
                            motion.mask.width, motion.mask.height));
  }
  return {};
}

Result<RgbImage> warp_blend(const RgbImage& a, const RgbImage& b, const Motion& motion) {
  return blend(a, b, motion, 0.5F, estimated);
}

Result<RgbImage> warp_blend_at(const RgbImage& a, const RgbImage& b, const Motion& motion, float t) {
  if (!(t > 0.0F && t < 1.0F)) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("instant {} hors de ]0, 1[", t));
  }
  return blend(a, b, motion, t, t == 0.5F ? estimated : from_middle);
}

Result<float> flow_max(const Motion& motion) {
  if (auto ok = check_motion(motion); !ok) return std::unexpected(ok.error());
  const Window& a = motion.area;
  float m = 0.0F;
  for (const PlaneF& p : motion.flow) {
    for (int y = a.y; y < a.y + a.height; ++y) {
      for (int x = a.x; x < a.x + a.width; ++x) m = std::fmax(m, std::fabs(p.at(x, y)));
    }
  }
  return m;
}

}  // namespace ajp::kernels::ref
