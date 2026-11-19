// Référence de la carte de la scène (kernels/ref/scene_flow.h).
#include "kernels/ref/scene_flow.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <utility>
#include <vector>

#include "kernels/ref/check.h"
#include "kernels/spec/color.h"

namespace kirari::kernels::ref {
namespace {

constexpr float kVectorUnit = 32.0F;  // S10.5

std::size_t at(int x, int y, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

// Une tuile lue sur l'hôte : ses moyennes.
struct TileView {
  float cx = 0.0F;  // centre, pixels du flux
  float cy = 0.0F;
  float mx = 0.0F;  // vecteur moyen, pixels du flux
  float my = 0.0F;
  float spread = 0.0F;   // écart type des vecteurs
  float matched = 0.0F;  // part de vecteurs sûrs
  float texture = 0.0F;  // par pixel
  float change = 0.0F;   // |b − a| moyen
  float shift = 0.0F;    // (b − a) moyen
  bool valid = false;
};

TileView view_of(const SceneTileSums& t, int col, int row, int flow_width, int flow_height) {
  TileView v;
  if (t.cells <= 0.0F || t.pixels <= 0.0F) return v;
  const float x0 = static_cast<float>(col * kSceneTileFlowPixels);
  const float y0 = static_cast<float>(row * kSceneTileFlowPixels);
  const float x1 = std::min(x0 + static_cast<float>(kSceneTileFlowPixels), static_cast<float>(flow_width));
  const float y1 = std::min(y0 + static_cast<float>(kSceneTileFlowPixels), static_cast<float>(flow_height));
  v.cx = 0.5F * (x0 + x1);
  v.cy = 0.5F * (y0 + y1);
  v.mx = t.sum_x / t.cells;
  v.my = t.sum_y / t.cells;
  const float var =
      std::max(0.0F, t.sum_xx / t.cells - v.mx * v.mx) + std::max(0.0F, t.sum_yy / t.cells - v.my * v.my);
  v.spread = std::sqrt(var);
  v.matched = 1.0F - t.unmatched / t.cells;
  v.texture = t.texture / t.pixels;
  v.change = t.diff / t.pixels;
  v.shift = t.signed_diff / t.pixels;
  v.valid = true;
  return v;
}

bool fit_candidate(const TileView& v) {
  return v.valid && v.matched >= scene::kFitMatched && v.texture >= scene::kTextureFloor &&
         v.spread <= scene::kSpreadMax;
}

float texture_weight(const TileView& v) {
  return std::clamp(v.texture / scene::kTextureFloor, 0.0F, 1.0F);
}

// Le déplacement du modèle en (x, y) : d = (c0 + c1 x + c2 y, c3 + c4 x + c5 y), pixels du flux.
struct Displacement {
  std::array<double, 6> c{};
  double dx(double x, double y) const { return c[0] + c[1] * x + c[2] * y; }
  double dy(double x, double y) const { return c[3] + c[4] * x + c[5] * y; }
};

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

double weighted_median(std::vector<std::pair<double, double>> values) {
  if (values.empty()) return 0.0;
  std::ranges::sort(values);
  double total = 0.0;
  for (const auto& [value, weight] : values) total += weight;
  double run = 0.0;
  for (const auto& [value, weight] : values) {
    run += weight;
    if (run >= 0.5 * total) return value;
  }
  return values.back().first;
}

struct CameraFit {
  bool fit = false;
  Displacement model;  // en coordonnées centrées et réduites
  double residual = 0.0;
};

// La caméra, sur les tuiles sûres, texturées et cohérentes, avec rejet itératif des aberrants (les dessins).
CameraFit fit_camera(const std::vector<TileView>& views, double cx, double cy, double scale) {
  std::vector<FitPoint> points;
  for (const TileView& v : views) {
    if (!fit_candidate(v)) continue;
    points.push_back({.x = (v.cx - cx) / scale,
                      .y = (v.cy - cy) / scale,
                      .dx = v.mx,
                      .dy = v.my,
                      .w = static_cast<double>(v.matched * texture_weight(v))});
  }
  const auto minimum =
      static_cast<std::size_t>(std::max(static_cast<double>(scene::kFitMinTiles),
                                        std::ceil(scene::kFitMinShare * static_cast<double>(views.size()))));
  CameraFit result;
  if (points.size() < minimum) return result;
  std::vector<std::uint8_t> keep(points.size(), 1U);
  std::vector<std::pair<double, double>> residuals;
  for (int round = 0; round < scene::kFitRounds; ++round) {
    if (!fit_affine(points, keep, result.model)) return {};
    residuals.clear();
    for (std::size_t i = 0; i < points.size(); ++i) {
      if (!keep[i]) continue;
      const FitPoint& p = points[i];
      residuals.emplace_back(std::hypot(p.dx - result.model.dx(p.x, p.y), p.dy - result.model.dy(p.x, p.y)),
                             p.w);
    }
    const double bound = std::max(static_cast<double>(scene::kBackgroundFloor),
                                  scene::kFitInlierScale * weighted_median(residuals));
    std::size_t kept = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
      const FitPoint& p = points[i];
      const bool inlier =
          std::hypot(p.dx - result.model.dx(p.x, p.y), p.dy - result.model.dy(p.x, p.y)) <= bound;
      keep[i] = inlier ? 1U : 0U;
      kept += keep[i];
    }
    if (kept < minimum) return {};
  }
  if (!fit_affine(points, keep, result.model)) return {};
  double sum = 0.0;
  double weight = 0.0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (!keep[i]) continue;
    const FitPoint& p = points[i];
    sum += p.w * std::hypot(p.dx - result.model.dx(p.x, p.y), p.dy - result.model.dy(p.x, p.y));
    weight += p.w;
  }
  result.residual = weight > 0.0 ? sum / weight : 0.0;
  result.fit = true;
  return result;
}

struct Classified {
  SceneClass kind = SceneClass::Uncertain;
  float confidence = 0.0F;
};

Classified classify(const TileView& v, const CameraFit& camera, double cx, double cy, double scale) {
  if (!v.valid) return {};
  if (v.matched < scene::kMatchedFloor) {
    if (v.change >= scene::kEffectDiff)
      return {.kind = SceneClass::Effect, .confidence = std::clamp(1.0F - v.matched, 0.0F, 1.0F)};
    return {};
  }
  const float weight = v.matched * texture_weight(v);
  const Classified unsure{.kind = SceneClass::Uncertain, .confidence = weight};
  // Un aplat : NVOF n'y voit rien (son vecteur vient des bords voisins), mais rien n'y bouge de visible tant
  // qu'il ne change guère. Il devient décor s'il correspond et ne contredit pas la caméra : pleinement sûr
  // sous kStillDiff d'écart |b − a|, plus du tout à kChangeDiff. Un vecteur qui la contredit n'y prouve pas
  // un dessin.
  const bool flat = v.texture < scene::kTextureFloor;
  if (flat &&
      (v.matched < scene::kFitMatched || v.change >= scene::kChangeDiff || v.spread > scene::kSpreadMax))
    return unsure;
  const float still =
      std::clamp((scene::kChangeDiff - v.change) / (scene::kChangeDiff - scene::kStillDiff), 0.0F, 1.0F);
  const Classified decor{.kind = SceneClass::Background, .confidence = flat ? v.matched * still : weight};
  if (v.spread > scene::kSpreadMax) return {.kind = SceneClass::Drawing, .confidence = 0.5F * weight};
  if (!camera.fit) return std::hypot(v.mx, v.my) < scene::kStillMotion ? decor : unsure;
  const double x = (v.cx - cx) / scale;
  const double y = (v.cy - cy) / scale;
  const double cam_x = camera.model.dx(x, y);
  const double cam_y = camera.model.dy(x, y);
  const double off = std::hypot(v.mx - cam_x, v.my - cam_y);
  const double bound = std::max(static_cast<double>(scene::kBackgroundFloor),
                                scene::kBackgroundShare * std::hypot(cam_x, cam_y));
  if (off < bound) return decor;
  return flat ? unsure : Classified{.kind = SceneClass::Drawing, .confidence = weight};
}

// Le lissage d'une tuile (SceneHistory). La première paire part de sa classe brute. Une paire sans preuve
// (idle : la tuile ne change pas sous une caméra tenue, où un dessin tenu se confond avec le décor, comme les
// images doublées d'une animation en deux ou en trois) garde la classe tenue et ne compte pas. Sinon, une
// classe nouvelle ne tient que vue deux paires de suite ; d'ici là, la classe tenue n'est plus sûre.
Classified smoothed(SceneHistory& history, std::size_t i, const Classified& raw, bool fresh, bool idle) {
  if (!fresh && idle) return {.kind = history.held[i], .confidence = raw.confidence};
  Classified kept = raw;
  if (!fresh && raw.kind != history.held[i] && raw.kind != history.seen[i])
    kept = {.kind = history.held[i], .confidence = scene::kContradicted * raw.confidence};
  history.held[i] = kept.kind;
  history.seen[i] = raw.kind;
  return kept;
}

// La tuile bouge hors de la caméra : un dessin, un effet ; sans caméra ajustée, toute tuile qui change (son
// mouvement ne peut pas être rendu à la caméra : ce n'est pas une preuve de dessin tenu).
bool moves_off_camera(SceneClass kind, bool changed, bool camera_fit) {
  return kind == SceneClass::Drawing || kind == SceneClass::Effect || (!camera_fit && changed);
}

bool tile_changed(const TileView& v) {
  if (!v.valid) return false;
  return std::hypot(v.mx, v.my) >= scene::kStillMotion || v.spread > scene::kSpreadMax ||
         1.0F - v.matched >= scene::kChangeUnmatched || v.change >= scene::kChangeDiff;
}

// Un fondu ou un flash : la luminosité glisse partout dans le même sens, et l'écart moyen n'est guère plus
// que ce glissement.
bool is_fade(const std::vector<TileView>& views) {
  double shift = 0.0;
  double change = 0.0;
  int count = 0;
  for (const TileView& v : views) {
    if (!v.valid) continue;
    shift += v.shift;
    change += v.change;
    ++count;
  }
  if (count == 0) return false;
  shift /= count;
  change /= count;
  if (std::abs(shift) < scene::kFadeShift || change > 1.5 * std::abs(shift) + scene::kStillDiff) return false;
  int agree = 0;
  for (const TileView& v : views) {
    if (v.valid && v.shift * shift > 0.0 && std::abs(v.shift) >= 0.5F * scene::kFadeShift) ++agree;
  }
  return static_cast<float>(agree) >= scene::kFadeAgree * static_cast<float>(count);
}

// Le modèle centré et réduit (pixels du flux) en caméra de travail : x' = x + d(x).
std::array<double, 6> camera_of(const Displacement& d, double cx, double cy, double scale, int divisor) {
  // d(x) = c0 + c1 (x − cx)/s + c2 (y − cy)/s, en flux ; en travail : X = k x, d_w = k d.
  const double k = divisor;
  const double a1 = d.c[1] / scale;
  const double a2 = d.c[2] / scale;
  const double b1 = d.c[4] / scale;
  const double b2 = d.c[5] / scale;
  const double a0 = d.c[0] - a1 * cx - a2 * cy;
  const double b0 = d.c[3] - b1 * cx - b2 * cy;
  return {k * a0, 1.0 + a1, a2, k * b0, b1, 1.0 + b2};
}

}  // namespace

Result<Plane<std::uint8_t>> scene_luma(const RgbImage& rgb, int divisor) {
  if (auto ok = check_rgb(rgb, "scene_luma"); !ok) return std::unexpected(ok.error());
  if (divisor < 1)
    return fail(ErrorCode::ParameterOutOfRange, std::format("scene_luma : diviseur {}", divisor));
  const int width = rgb.width() / divisor;
  const int height = rgb.height() / divisor;
  if (width < kSceneFlowMinSide || height < kSceneFlowMinSide) {
    return fail(ErrorCode::DimensionOutOfRange,
                std::format("scene_luma : flux {}x{} trop petit", width, height));
  }
  const Coefficients k = coefficients(Matrix::Bt709);
  const float kg = 1.0F - k.kr - k.kb;
  const float area = 1.0F / static_cast<float>(divisor * divisor);
  Plane<std::uint8_t> out(width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0F;
      for (int j = 0; j < divisor; ++j) {
        for (int i = 0; i < divisor; ++i) {
          const Rgb c = rgb.at(x * divisor + i, y * divisor + j);
          sum += k.kr * c.r + kg * c.g + k.kb * c.b;
        }
      }
      const float luma = std::clamp(sum * area, 0.0F, 1.0F);
      out.at(x, y) = static_cast<std::uint8_t>(static_cast<int>(255.0F * luma + 0.5F));
    }
  }
  return out;
}

Result<SceneFlowTiles> scene_tile_sums(const Plane<std::uint8_t>& a, const Plane<std::uint8_t>& b,
                                       const Plane<FlowVector>& flow, const Plane<std::uint8_t>& cost,
                                       int divisor) {
  if (auto ok = check_plane(a, "scene_tile_sums : a"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_same_size(a, b, "scene_tile_sums : a et b"); !ok) return std::unexpected(ok.error());
  const int cells_x = scene_flow_cells(a.width);
  const int cells_y = scene_flow_cells(a.height);
  if (flow.width != cells_x || flow.height != cells_y || cost.width != cells_x || cost.height != cells_y ||
      flow.data.size() != flow.area() || cost.data.size() != cost.area()) {
    return fail(ErrorCode::SizeMismatch,
                std::format("scene_tile_sums : flux {}x{} et coût {}x{} pour {}x{} vecteurs", flow.width,
                            flow.height, cost.width, cost.height, cells_x, cells_y));
  }
  SceneFlowTiles out{.divisor = divisor,
                     .flow_width = a.width,
                     .flow_height = a.height,
                     .cols = scene_flow_tiles(a.width),
                     .rows = scene_flow_tiles(a.height),
                     .tiles = {}};
  out.tiles.resize(static_cast<std::size_t>(out.cols) * static_cast<std::size_t>(out.rows));
  for (int row = 0; row < out.rows; ++row) {
    for (int col = 0; col < out.cols; ++col) {
      // Sommes entières, comme le noyau : le résultat ne dépend pas de l'ordre.
      std::int64_t cells = 0;
      std::int64_t sx = 0;
      std::int64_t sy = 0;
      std::int64_t sxx = 0;
      std::int64_t syy = 0;
      std::int64_t sc = 0;
      std::int64_t bad = 0;
      for (int j = row * kSceneTileCells; j < std::min((row + 1) * kSceneTileCells, cells_y); ++j) {
        for (int i = col * kSceneTileCells; i < std::min((col + 1) * kSceneTileCells, cells_x); ++i) {
          const FlowVector v = flow.at(i, j);
          const std::uint8_t c = cost.at(i, j);
          ++cells;
          sx += v.x;
          sy += v.y;
          sxx += std::int64_t{v.x} * v.x;
          syy += std::int64_t{v.y} * v.y;
          sc += c;
          bad += c >= kSceneCostHigh ? 1 : 0;
        }
      }
      std::int64_t pixels = 0;
      std::int64_t texture = 0;
      std::int64_t diff = 0;
      std::int64_t signed_diff = 0;
      const int y1 = std::min((row + 1) * kSceneTileFlowPixels, a.height);
      const int x1 = std::min((col + 1) * kSceneTileFlowPixels, a.width);
      for (int y = row * kSceneTileFlowPixels; y < y1; ++y) {
        for (int x = col * kSceneTileFlowPixels; x < x1; ++x) {
          const int va = a.data[at(x, y, a.width)];
          const int vb = b.data[at(x, y, b.width)];
          ++pixels;
          if (x + 1 < a.width) texture += std::abs(int{a.data[at(x + 1, y, a.width)]} - va);
          if (y + 1 < a.height) texture += std::abs(int{a.data[at(x, y + 1, a.width)]} - va);
          diff += std::abs(vb - va);
          signed_diff += vb - va;
        }
      }
      out.tiles[at(col, row, out.cols)] =
          SceneTileSums{.cells = static_cast<float>(cells),
                        .sum_x = static_cast<float>(sx) / kVectorUnit,
                        .sum_y = static_cast<float>(sy) / kVectorUnit,
                        .sum_xx = static_cast<float>(sxx) / (kVectorUnit * kVectorUnit),
                        .sum_yy = static_cast<float>(syy) / (kVectorUnit * kVectorUnit),
                        .cost = static_cast<float>(sc),
                        .unmatched = static_cast<float>(bad),
                        .pixels = static_cast<float>(pixels),
                        .texture = static_cast<float>(texture),
                        .diff = static_cast<float>(diff),
                        .signed_diff = static_cast<float>(signed_diff)};
    }
  }
  return out;
}

SceneMotion scene_motion_of(const SceneFlowTiles& tiles, SceneHistory& history) {
  SceneMotion out;
  out.cols = tiles.cols;
  out.rows = tiles.rows;
  out.tile_px = kSceneTileFlowPixels * tiles.divisor;
  const std::size_t count = static_cast<std::size_t>(tiles.cols) * static_cast<std::size_t>(tiles.rows);
  if (tiles.tiles.size() != count || count == 0) {
    history.clear();
    out.cols = 0;
    out.rows = 0;
    return out;
  }
  std::vector<TileView> views(count);
  double cells = 0.0;
  double unmatched = 0.0;
  for (int row = 0; row < tiles.rows; ++row) {
    for (int col = 0; col < tiles.cols; ++col) {
      const SceneTileSums& t = tiles.tiles[at(col, row, tiles.cols)];
      views[at(col, row, tiles.cols)] = view_of(t, col, row, tiles.flow_width, tiles.flow_height);
      cells += t.cells;
      unmatched += t.unmatched;
    }
  }
  out.unmatched = cells > 0.0 ? unmatched / cells : 0.0;
  out.fade = is_fade(views);
  out.cut_likely = !out.fade && out.unmatched >= scene::kCutUnmatched;

  const double cx = 0.5 * tiles.flow_width;
  const double cy = 0.5 * tiles.flow_height;
  const double scale = std::max(1.0, 0.5 * std::max(tiles.flow_width, tiles.flow_height));
  const CameraFit camera = out.cut_likely ? CameraFit{} : fit_camera(views, cx, cy, scale);
  out.camera_fit = camera.fit;
  if (camera.fit) {
    out.camera = camera_of(camera.model, cx, cy, scale, tiles.divisor);
    out.residual = camera.residual * tiles.divisor;
    const auto& c = out.camera;
    out.background_held = std::hypot(c[0], c[3]) < scene::kHeldCamera &&
                          std::abs(c[1] - 1.0) < scene::kHeldLinear && std::abs(c[2]) < scene::kHeldLinear &&
                          std::abs(c[4]) < scene::kHeldLinear && std::abs(c[5] - 1.0) < scene::kHeldLinear;
  }
  const bool camera_moves = camera.fit && !out.background_held;

  const bool same_grid =
      history.cols == tiles.cols && history.rows == tiles.rows && history.held.size() == count;
  if (!same_grid) {
    history.cols = tiles.cols;
    history.rows = tiles.rows;
    history.held.assign(count, SceneClass::Uncertain);
    history.seen.assign(count, SceneClass::Uncertain);
  }
  out.tiles.resize(count);
  std::size_t moving = 0;  // tuiles qui bougent hors de la caméra (moves_off_camera)
  const auto divisor = static_cast<float>(tiles.divisor);
  for (std::size_t i = 0; i < count; ++i) {
    const TileView& v = views[i];
    const Classified raw = classify(v, camera, cx, cy, scale);
    const bool changed = tile_changed(v);
    if (moves_off_camera(raw.kind, changed, camera.fit)) ++moving;
    const Classified kept = smoothed(history, i, raw, !same_grid, !changed && !camera_moves);
    out.tiles[i] = SceneTile{.kind = kept.kind,
                             .confidence = kept.confidence,
                             .dx = v.mx * divisor,
                             .dy = v.my * divisor,
                             .change = v.change,
                             .shift = v.shift,
                             .changed = changed};
  }
  out.drawing_held = !out.cut_likely && !out.fade &&
                     static_cast<float>(moving) <= scene::kDrawingHeldShare * static_cast<float>(count);
  if (out.cut_likely) history.clear();  // le plan suivant repart de ses propres classes
  return out;
}

}  // namespace kirari::kernels::ref
