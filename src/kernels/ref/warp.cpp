// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:963-1096.
#include "kernels/ref/warp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <string_view>

#include "kernels/ref/check.h"

namespace kirari::kernels::ref {
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

// Lecture de Catmull-Rom (spec/warp.h : RifeWarp) : position ramenée dans le plan, points répétés au bord,
// sortie bornée au minimum et au maximum des quatre points centraux (anti-ringing local : le dépassement de
// la cubique sur un trait net est coupé, mais les sur-blancs et les négatifs de la source passent, C50-08).
float catmull_rom_at(const PlaneF& p, float x, float y) {
  const int w = p.width;
  const int h = p.height;
  x = std::fmin(std::fmax(x, 0.0F), static_cast<float>(w - 1));
  y = std::fmin(std::fmax(y, 0.0F), static_cast<float>(h - 1));
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const std::array<float, 4> wx = catmull_rom_weights(x - static_cast<float>(x0));
  const std::array<float, 4> wy = catmull_rom_weights(y - static_cast<float>(y0));
  float sum = 0.0F;
  float lo = std::numeric_limits<float>::infinity();
  float hi = -std::numeric_limits<float>::infinity();
  for (int j = 0; j < 4; ++j) {
    const int yy = std::clamp(y0 - 1 + j, 0, h - 1);
    float row = 0.0F;
    for (int i = 0; i < 4; ++i) {
      const float v = p.at(std::clamp(x0 - 1 + i, 0, w - 1), yy);
      row += v * wx[static_cast<std::size_t>(i)];
      if ((i == 1 || i == 2) && (j == 1 || j == 2)) {
        lo = std::fmin(lo, v);
        hi = std::fmax(hi, v);
      }
    }
    sum += row * wy[static_cast<std::size_t>(j)];
  }
  return std::fmin(std::fmax(sum, lo), hi);
}

float image_at(const PlaneF& p, RifeWarp warp, float x, float y) {
  return warp == RifeWarp::CatmullRom ? catmull_rom_at(p, x, y) : bilinear_at(p, x, y);
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

// Le mouvement d'ensemble d'un warp (spec/warp_pan.h) : g, flot au milieu en pixels de la fenêtre.
struct Pan {
  bool fade = false;  // pan_lock hors garder les poses : le fondu sous un demi-pixel
  bool on = false;    // et l'estimation est sûre : le flot remplacé
  std::array<float, 4> g{};
};

// Le flot lu f remplacé par g là où il est petit et proche.
void pan_substitute(std::array<float, 4>& f, const Pan& pan) {
  if (!pan.on) return;
  float m = 0.0F;
  float e = 0.0F;
  for (std::size_t c = 0; c < 4; ++c) {
    m = std::fmax(m, std::fabs(f[c]));
    e = std::fmax(e, std::fabs(f[c] - pan.g[c]));
  }
  if (m < kPanFlowLimit && e < kPanAgreement) f = pan.g;
}

std::array<float, 4> flow_at(const Motion& m, float u, float v, const Pan& pan) {
  std::array<float, 4> f{};
  for (std::size_t c = 0; c < 4; ++c) f[c] = bilinear_at(m.flow[c], m.area, u, v);
  pan_substitute(f, pan);
  return f;
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
using DisplacementAt = Displacement (*)(const Motion&, const Motion*, const Pan&, float, float, float);

// L'instant que RIFE a estimé : mouvement et masque tels quels (mouvement d'ensemble : flot remplacé, masque
// ramené vers le fondu sous un demi-pixel).
Displacement estimated(const Motion& m, const Motion* /*previous*/, const Pan& pan, float u, float v,
                       float t) {
  const std::array<float, 4> f = flow_at(m, u, v, pan);
  float weight = bilinear_at(m.mask, m.area, u, v);
  if (pan.fade) {
    const float mm = std::fmin(std::fmax(weight, kMaskEpsilon), 1.0F - kMaskEpsilon);
    const float w = slow_fade_weight(std::hypot(f[2] - f[0], f[3] - f[1]));
    const float z = w * std::log(mm / (1.0F - mm)) + (1.0F - w) * std::log((1.0F - t) / t);
    weight = 1.0F / (1.0F + std::exp(-z));
  }
  return {.ax = f[0], .ay = f[1], .bx = f[2], .by = f[3], .weight = weight};
}

// Un autre instant depuis le mouvement du milieu : ligne droite et loi de mélange ajustée.
Displacement from_middle(const Motion& m, const Motion* /*previous*/, const Pan& pan, float u, float v,
                         float t) {
  const std::array<float, 4> f = flow_at(m, u, v, pan);
  const float u2 = u - ((1.0F - t) * f[0] + t * f[2]);
  const float v2 = v - ((1.0F - t) * f[1] + t * f[3]);
  const std::array<float, 4> f2 = flow_at(m, u2, v2, pan);
  const float tx = f2[2] - f2[0];
  const float ty = f2[3] - f2[1];
  const float mm =
      std::fmin(std::fmax(bilinear_at(m.mask, m.area, u2, v2), kMaskEpsilon), 1.0F - kMaskEpsilon);
  float z = kBlendLawA * std::log(mm / (1.0F - mm)) + kBlendLawB * std::log((1.0F - t) / t) + kBlendLawC;
  if (pan.fade) {
    const float w = slow_fade_weight(std::hypot(tx, ty));
    z = w * z + (1.0F - w) * std::log((1.0F - t) / t);
  }
  return {.ax = -t * tx,
          .ay = -t * ty,
          .bx = (1.0F - t) * tx,
          .by = (1.0F - t) * ty,
          .weight = 1.0F / (1.0F + std::exp(-z))};
}

// L'instant réglé du pixel (u, v) puis from_middle à cet instant (spec/poses.h).
Displacement keeping_poses(const Motion& m, const Motion* previous, const Pan& /*pan*/, float u, float v,
                           float t) {
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
  return from_middle(m, previous, Pan{}, u, v, tp);
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
  Pan pan;
  if (motion.pan_lock && previous == nullptr && at != keeping_poses) {
    const PanEstimate e = pan_estimate(a, b);
    pan.fade = true;
    pan.on = e.sure;
    // L'instant du flot lu : le milieu, ou t quand le mouvement est celui de la sortie.
    const float s = motion.at_instant ? t : 0.5F;
    pan.g = {-s * e.dx / sx, -s * e.dy / sy, (1.0F - s) * e.dx / sx, (1.0F - s) * e.dy / sy};
  }
  RgbImage out(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float fx = static_cast<float>(x);
      const float fy = static_cast<float>(y);
      const Displacement d = at(motion, previous, pan, (fx + 0.5F) / sx - 0.5F, (fy + 0.5F) / sy - 0.5F, t);
      for (std::size_t c = 0; c < out.planes.size(); ++c) {
        const float va = image_at(a.planes[c], motion.warp, fx + sx * d.ax, fy + sy * d.ay);
        const float vb = image_at(b.planes[c], motion.warp, fx + sx * d.bx, fy + sy * d.by);
        out.planes[c].at(x, y) = va * d.weight + vb * (1.0F - d.weight);
      }
    }
  }
  return out;
}

}  // namespace

float catmull_rom_sample(const PlaneF& p, float x, float y) {
  return catmull_rom_at(p, x, y);
}

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
  return blend(a, b, motion, nullptr, t, motion.at_instant || t == 0.5F ? estimated : from_middle);
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
  return flow_percentile(
      bins.data(), static_cast<unsigned long long>(a.width) * static_cast<unsigned long long>(a.height));
}

}  // namespace kirari::kernels::ref
