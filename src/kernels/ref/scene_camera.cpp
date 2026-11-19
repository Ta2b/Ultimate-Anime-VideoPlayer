// Référence de l'affinage de la caméra de la carte (kernels/ref/scene_camera.h).
#include "kernels/ref/scene_camera.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <utility>
#include <vector>

#include "kernels/ref/check.h"
#include "kernels/ref/scene_fit.h"

namespace kirari::kernels::ref {
namespace {

std::size_t index(int x, int y, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

// La boîte (2r + 1)², bords répétés : la somme de la ligne, puis celle de la colonne, puis × 1/(2r + 1)².
std::vector<float> box_blur(const Plane<std::uint8_t>& p, int r) {
  const int w = p.width;
  const int h = p.height;
  std::vector<float> rows(p.area());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float sum = 0.0F;
      for (int k = -r; k <= r; ++k)
        sum += static_cast<float>(p.data[index(std::clamp(x + k, 0, w - 1), y, w)]);
      rows[index(x, y, w)] = sum;
    }
  }
  const float area = 1.0F / static_cast<float>((2 * r + 1) * (2 * r + 1));
  std::vector<float> out(p.area());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float sum = 0.0F;
      for (int k = -r; k <= r; ++k) sum += rows[index(x, std::clamp(y + k, 0, h - 1), w)];
      out[index(x, y, w)] = sum * area;
    }
  }
  return out;
}

struct Lumas {
  std::vector<float> a;
  std::vector<float> b;
  int width = 0;
  int height = 0;
  int cols = 0;
};

bool in_decor(const SceneCameraRequest& request, const Lumas& l, int i, int j) {
  return request.decor[index(i / kSceneTileFlowPixels, j / kSceneTileFlowPixels, l.cols)] != 0U;
}

// Un pas de Gauss-Newton (spec/scene_camera.h) : faux si H est mal conditionné, p inchangé.
bool refine_step(const Lumas& l, const SceneCameraRequest& request, std::array<double, 6>& p) {
  const SceneCameraFrame frame = scene_camera_frame(l.width, l.height);
  std::array<float, 6> pf{};
  for (std::size_t k = 0; k < 6; ++k) pf[k] = static_cast<float>(p[k]);
  std::array<unsigned, scene::kRefineBins> histogram{};
  unsigned count = 0;
  const auto each = [&](auto&& use) {
    for (int j = 0; j < l.height; j += scene::kRefineStride) {
      for (int i = 0; i < l.width; i += scene::kRefineStride) {
        if (!in_decor(request, l, i, j)) continue;
        const SceneModelAt m = scene_model_at(pf.data(), frame, i, j);
        const SceneWarpTerm t = scene_warp_term(l.a.data(), l.b.data(), l.width, l.height, i, j, m.dx, m.dy);
        if (t.valid) use(t, m);
      }
    }
  };
  each([&](const SceneWarpTerm& t, const SceneModelAt&) {
    ++histogram[static_cast<std::size_t>(scene_refine_bin(std::abs(t.e)))];
    ++count;
  });
  const float median = scene_refine_median(histogram.data(), count);
  std::array<double, scene::kRefineSums> sums{};
  std::array<float, scene::kRefineSums> terms{};
  each([&](const SceneWarpTerm& t, const SceneModelAt& m) {
    scene_refine_terms(t, m, scene_huber_weight(t.e, median), terms.data());
    for (std::size_t k = 0; k < sums.size(); ++k) sums[k] += terms[k];
  });
  return scene_refine_solve(sums.data(), p.data());
}

// La vérification d'une tuile (spec/scene_camera.h), sur les lumas du dernier flou.
SceneTileCheck check_tile(const Lumas& l, const std::array<double, 6>& p, int col, int row) {
  const SceneCameraFrame frame = scene_camera_frame(l.width, l.height);
  const float slack = scene_moving_slack(l.height);
  std::array<float, 6> pf{};
  for (std::size_t k = 0; k < 6; ++k) pf[k] = static_cast<float>(p[k]);
  SceneTileCheck out;
  const int x1 = std::min((col + 1) * kSceneTileFlowPixels, l.width);
  const int y1 = std::min((row + 1) * kSceneTileFlowPixels, l.height);
  for (int step = 0; step < scene::kCheckSteps; ++step) {
    SceneCheckSums s;
    for (int j = row * kSceneTileFlowPixels; j < y1; ++j) {
      for (int i = col * kSceneTileFlowPixels; i < x1; ++i) {
        const SceneModelAt m = scene_model_at(pf.data(), frame, i, j);
        const SceneWarpTerm t =
            scene_warp_term(l.a.data(), l.b.data(), l.width, l.height, i, j, m.dx + out.sx, m.dy + out.sy);
        if (!t.valid) continue;
        s.xx += t.gx * t.gx;
        s.xy += t.gx * t.gy;
        s.yy += t.gy * t.gy;
        s.bx += t.gx * t.e;
        s.by += t.gy * t.e;
        s.pixels += 1.0F;
        s.error += std::abs(t.e);
        const bool moving = scene_pixel_moving(t, slack);
        s.moving += moving ? 1.0F : 0.0F;
        s.brighter += moving && t.e < 0.0F ? 1.0F : 0.0F;
        if (step == 0 && moving)
          s.sharp += scene_change_sharp(l.a.data(), l.width, l.height, i, j, t) ? 1.0F : 0.0F;
      }
    }
    if (step == 0) scene_check_seen(s, out);
    if (step == scene::kCheckSteps - 1) scene_check_settled(s, out);
    scene_check_step(s, out);
  }
  return out;
}

}  // namespace

Result<SceneCameraRefined> scene_camera_refine(const Plane<std::uint8_t>& a, const Plane<std::uint8_t>& b,
                                               const SceneCameraRequest& request) {
  if (auto ok = check_plane(a, "scene_camera_refine : a"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_same_size(a, b, "scene_camera_refine : a et b"); !ok)
    return std::unexpected(ok.error());
  const int cols = scene_flow_tiles(a.width);
  const int rows = scene_flow_tiles(a.height);
  if (request.decor.size() != static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows)) {
    return fail(ErrorCode::SizeMismatch, std::format("scene_camera_refine : {} tuiles de décor pour {}x{}",
                                                     request.decor.size(), cols, rows));
  }
  SceneCameraRefined out{.model = request.model, .steps = 0, .tiles = {}};
  Lumas l{.a = {}, .b = {}, .width = a.width, .height = a.height, .cols = cols};
  bool stopped = false;
  for (const int radius : {scene::kRefineCoarseRadius, scene::kRefineFineRadius}) {
    l.a = box_blur(a, radius);
    l.b = box_blur(b, radius);
    for (int step = 0; step < scene::kRefineSteps && !stopped; ++step) {
      stopped = !refine_step(l, request, out.model);
      out.steps += stopped ? 0 : 1;
    }
  }
  out.tiles.resize(request.decor.size());
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      out.tiles[index(col, row, cols)] = check_tile(l, out.model, col, row);
    }
  }
  return out;
}

std::optional<SceneCameraRequest> scene_camera_request(const SceneCamera& camera) {
  if (!camera.fit || camera.decor_pixels < scene::kRefineMinPixels) return std::nullopt;
  return SceneCameraRequest{.model = camera.model, .decor = camera.decor};
}

SceneCameraWork scene_camera_work(const SceneCamera& camera, int flow_width, int flow_height, int divisor) {
  SceneCameraWork out;
  if (!camera.fit) return out;
  // d(x) = c0 + c1 (x − cx)/s + c2 (y − cy)/s, en flux (le repère de scene_camera_frame) ; en travail :
  // X = k x, d_w = k d.
  const double cx = 0.5 * flow_width;
  const double cy = 0.5 * flow_height;
  const double s = std::max(1.0, 0.5 * std::max(flow_width, flow_height));
  const auto& c = camera.model;
  const double k = divisor;
  const double a1 = c[1] / s;
  const double a2 = c[2] / s;
  const double b1 = c[4] / s;
  const double b2 = c[5] / s;
  out.camera = {k * (c[0] - a1 * cx - a2 * cy), 1.0 + a1, a2, k * (c[3] - b1 * cx - b2 * cy), b1, 1.0 + b2};
  out.fit = true;
  out.refined = camera.refined;
  out.residual = (camera.check >= 0.0 ? camera.check : camera.residual) * k;
  out.residual_tiles = camera.check >= 0.0 ? camera.check_tiles : 0;
  return out;
}

double scene_camera_gap(const std::array<double, 6>& a, const std::array<double, 6>& b, int flow_width,
                        int flow_height) {
  const double s = std::max(1.0, 0.5 * std::max(flow_width, flow_height));
  const double x = 0.5 * flow_width / s;
  const double y = 0.5 * flow_height / s;
  double gap = 0.0;
  for (const double sx : {-x, x}) {
    for (const double sy : {-y, y}) {
      const double dx = (a[0] - b[0]) + (a[1] - b[1]) * sx + (a[2] - b[2]) * sy;
      const double dy = (a[3] - b[3]) + (a[4] - b[4]) * sx + (a[5] - b[5]) * sy;
      gap = std::max(gap, std::hypot(dx, dy));
    }
  }
  return gap;
}

void scene_camera_apply(SceneCamera& camera, const SceneCameraRefined& refined) {
  if (refined.steps > 0) {
    camera.model = refined.model;
    camera.refined = true;
  }
  camera.checks = refined.tiles;
  constexpr float kArea = static_cast<float>(kSceneTileFlowPixels * kSceneTileFlowPixels);
  std::vector<double> both;
  std::vector<double> one;
  for (std::size_t i = 0; i < refined.tiles.size(); ++i) {
    const SceneTileCheck& t = refined.tiles[i];
    if (i >= camera.decor.size() || camera.decor[i] == 0U || t.pixels < scene::kCheckCover * kArea) continue;
    const float floor = scene::kCheckFloor * std::max(t.pixels, 1.0F);
    const double off = std::hypot(static_cast<double>(t.sx), static_cast<double>(t.sy));
    if (t.l2 >= floor) both.push_back(off);
    if (t.l1 >= floor) one.push_back(off);
  }
  const std::size_t needed = scene::kCheckMinTiles;
  std::vector<double>& used = both.size() >= needed ? both : one;
  if (used.size() < needed) return;
  camera.check_tiles = static_cast<int>(used.size());
  camera.check = scene_fit::median_of(std::move(used));
}

}  // namespace kirari::kernels::ref
