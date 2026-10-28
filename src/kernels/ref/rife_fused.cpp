#include "kernels/ref/rife_fused.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace ajp::kernels::ref {
namespace {

using Plane = std::vector<double>;

std::size_t at(int y, int x, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

// Échantillon bilinéaire d'un plan h × w en (fy, fx), coordonnées ramenées dans le plan (bords répétés).
double sample(const double* plane, int height, int width, double fy, double fx) {
  fy = std::clamp(fy, 0.0, static_cast<double>(height - 1));
  fx = std::clamp(fx, 0.0, static_cast<double>(width - 1));
  const int y0 = static_cast<int>(std::floor(fy));
  const int x0 = static_cast<int>(std::floor(fx));
  const int y1 = std::min(y0 + 1, height - 1);
  const int x1 = std::min(x0 + 1, width - 1);
  const double wy = fy - y0;
  const double wx = fx - x0;
  return (plane[at(y0, x0, width)] * (1.0 - wx) + plane[at(y0, x1, width)] * wx) * (1.0 - wy) +
         (plane[at(y1, x0, width)] * (1.0 - wx) + plane[at(y1, x1, width)] * wx) * wy;
}

// Resize linéaire, coordonnées demi-pixel, d'un plan h × w vers oh × ow (ONNX Resize, mode linear).
Plane resize(const double* plane, int height, int width, int out_height, int out_width) {
  Plane out(static_cast<std::size_t>(out_height) * static_cast<std::size_t>(out_width));
  const double sy = static_cast<double>(height) / out_height;
  const double sx = static_cast<double>(width) / out_width;
  for (int y = 0; y < out_height; ++y) {
    for (int x = 0; x < out_width; ++x)
      out[at(y, x, out_width)] = sample(plane, height, width, (y + 0.5) * sy - 0.5, (x + 0.5) * sx - 0.5);
  }
  return out;
}

// Les canaux first .. first + count - 1 de Z_k = DepthToSpace(Y_k) (CRD, bloc 2), ramenés en pleine
// résolution par le Resize de sortie du bloc (facteur d_k ; aucun pour d_k = 1).
std::vector<Plane> upscaled_z(const RifeFusedState& s, int k, int first, int count) {
  const int d = s.plan.reduce[static_cast<std::size_t>(k)];
  const int yh = s.height / (2 * d);
  const int yw = s.width / (2 * d);
  const int zh = 2 * yh;
  const int zw = 2 * yw;
  const Plane& y = s.y[static_cast<std::size_t>(k)];
  std::vector<Plane> planes;
  for (int c = first; c < first + count; ++c) {
    Plane z(static_cast<std::size_t>(zh) * static_cast<std::size_t>(zw));
    for (int zy = 0; zy < zh; ++zy) {
      for (int zx = 0; zx < zw; ++zx) {
        const int channel = 4 * c + 2 * (zy % 2) + (zx % 2);
        z[at(zy, zx, zw)] = y[(static_cast<std::size_t>(channel) * static_cast<std::size_t>(yh) +
                               static_cast<std::size_t>(zy / 2)) *
                                  static_cast<std::size_t>(yw) +
                              static_cast<std::size_t>(zx / 2)];
      }
    }
    planes.push_back(d == 1 ? std::move(z) : resize(z.data(), zh, zw, s.height, s.width));
  }
  return planes;
}

// Le flot accumulé des blocs 0 .. blocks-1 : Σ d_i × flot de Z_i, en pleine résolution (4 plans).
std::vector<Plane> accumulated_flow(const RifeFusedState& s, int blocks) {
  const std::size_t area = static_cast<std::size_t>(s.height) * static_cast<std::size_t>(s.width);
  std::vector<Plane> flow(static_cast<std::size_t>(kFusedFlowChannels), Plane(area, 0.0));
  for (int i = 0; i < blocks; ++i) {
    const double scale = s.plan.reduce[static_cast<std::size_t>(i)];
    const std::vector<Plane> z = upscaled_z(s, i, 0, kFusedFlowChannels);
    for (std::size_t c = 0; c < flow.size(); ++c) {
      for (std::size_t p = 0; p < area; ++p) flow[c][p] += z[c][p] * scale;
    }
  }
  return flow;
}

// GridSample (coins alignés, bords répétés) d'un plan h × w par le flot (pixels), la grille normalisée étant
// celle de l'entrée (plans 7 à 10 : maillage et pas).
Plane warp(const RifeFusedState& s, const double* plane, const Plane& flow_x, const Plane& flow_y) {
  const std::size_t area = static_cast<std::size_t>(s.height) * static_cast<std::size_t>(s.width);
  const double* mesh_x = s.input.data() + 7 * area;
  const double* mesh_y = s.input.data() + 8 * area;
  const double* step_x = s.input.data() + 9 * area;
  const double* step_y = s.input.data() + 10 * area;
  Plane out(area);
  for (int y = 0; y < s.height; ++y) {
    for (int x = 0; x < s.width; ++x) {
      const std::size_t i = at(y, x, s.width);
      const double gx = mesh_x[i] + flow_x[i] * step_x[i];
      const double gy = mesh_y[i] + flow_y[i] * step_y[i];
      const double ix = (gx + 1.0) * 0.5 * (s.width - 1);
      const double iy = (gy + 1.0) * 0.5 * (s.height - 1);
      out[i] = sample(plane, s.height, s.width, iy, ix);
    }
  }
  return out;
}

}  // namespace

std::vector<double> rife_fused_block_input(const RifeFusedState& s, int k) {
  const std::size_t area = static_cast<std::size_t>(s.height) * static_cast<std::size_t>(s.width);
  const int d = s.plan.reduce[static_cast<std::size_t>(k)];
  std::vector<Plane> planes;  // le Concat plein résolution, un plan par canal
  const auto plane_of = [](const std::vector<double>& v, std::size_t index, std::size_t size) {
    return Plane(v.begin() + static_cast<std::ptrdiff_t>(index * size),
                 v.begin() + static_cast<std::ptrdiff_t>((index + 1) * size));
  };
  if (k == 0) {
    for (std::size_t c = 0; c < 6; ++c) planes.push_back(plane_of(s.input, c, area));
    for (std::size_t c = 0; c < static_cast<std::size_t>(kFusedEncodePlanes); ++c)
      planes.push_back(plane_of(s.encode, c, area));
    planes.push_back(plane_of(s.input, 6, area));
  } else {
    const std::vector<Plane> flow = accumulated_flow(s, k);
    for (std::size_t n = 0; n < 2; ++n) {  // image n puis, plus loin, son encodage
      const Plane& fx = flow[2 * n];
      const Plane& fy = flow[2 * n + 1];
      for (std::size_t c = 0; c < 3; ++c)
        planes.push_back(warp(s, s.input.data() + (3 * n + c) * area, fx, fy));
    }
    for (std::size_t n = 0; n < 2; ++n) {
      for (std::size_t c = 0; c < 4; ++c)
        planes.push_back(warp(s, s.encode.data() + (4 * n + c) * area, flow[2 * n], flow[2 * n + 1]));
    }
    planes.push_back(plane_of(s.input, 6, area));
    for (Plane& p : upscaled_z(s, k - 1, kFusedFlowChannels, kFusedAttributeChannels))
      planes.push_back(std::move(p));
    for (const Plane& p : flow) {
      Plane scaled(area);
      for (std::size_t i = 0; i < area; ++i) scaled[i] = p[i] / d;
      planes.push_back(std::move(scaled));
    }
  }
  const int oh = s.height / d;
  const int ow = s.width / d;
  std::vector<double> out;
  out.reserve(planes.size() * static_cast<std::size_t>(oh) * static_cast<std::size_t>(ow));
  for (const Plane& p : planes) {
    const Plane small = d == 1 ? p : resize(p.data(), s.height, s.width, oh, ow);
    out.insert(out.end(), small.begin(), small.end());
  }
  return out;
}

RifeFusedResult rife_fused_finish(const RifeFusedState& s) {
  const std::size_t area = static_cast<std::size_t>(s.height) * static_cast<std::size_t>(s.width);
  const int last = s.plan.blocks - 1;
  std::vector<Plane> flow = accumulated_flow(s, last);
  const std::vector<Plane> z = upscaled_z(s, last, 0, 5);  // flot (4) et masque (1) du dernier bloc
  for (std::size_t c = 0; c < flow.size(); ++c) {
    for (std::size_t p = 0; p < area; ++p) flow[c][p] += z[c][p];
  }
  RifeFusedResult result;
  result.mask.resize(area);
  for (std::size_t p = 0; p < area; ++p) result.mask[p] = 1.0 / (1.0 + std::exp(-z[4][p]));
  result.image.resize(3 * area);
  for (std::size_t c = 0; c < 3; ++c) {
    const Plane a = warp(s, s.input.data() + c * area, flow[0], flow[1]);
    const Plane b = warp(s, s.input.data() + (3 + c) * area, flow[2], flow[3]);
    for (std::size_t p = 0; p < area; ++p)
      result.image[c * area + p] = a[p] * result.mask[p] + b[p] * (1.0 - result.mask[p]);
  }
  for (const Plane& p : flow) result.flow.insert(result.flow.end(), p.begin(), p.end());
  return result;
}

}  // namespace ajp::kernels::ref
