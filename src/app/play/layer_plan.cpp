#include "app/play/layer_plan.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace kirari::app::play {
namespace {

using kernels::SceneClass;

bool sure_decor(const kernels::SceneTile& tile) {
  return (tile.kind == SceneClass::Background && tile.confidence >= kLayerConfidence) ||
         (tile.kind == SceneClass::Uncertain && !tile.changed);
}

// Une tuile qui peut porter un dessin ou un effet : ses voisines ne passent pas par la caméra.
bool keeps_rife(const kernels::SceneTile& tile) {
  return tile.kind == SceneClass::Drawing || tile.kind == SceneClass::Effect ||
         (tile.kind == SceneClass::Uncertain && tile.changed);
}

// Un déplacement en tuiles, arrondi ; un déplacement non fini (tuile sans vecteur) compte pour nul.
int tiles_moved(float d, double px) {
  const double tiles = static_cast<double>(d) / px;
  return std::isfinite(tiles) ? static_cast<int>(std::lround(std::clamp(tiles, -1e4, 1e4))) : 0;
}

// Les tuiles qui gardent RIFE : chaque dessin ou effet balaie le rectangle de tuiles entre sa place dans a et
// sa place dans b (son déplacement dx, dy), élargi d'une tuile. Un personnage rapide, qui traverse plus d'une
// tuile d'une image à l'autre, reste à RIFE sur tout son chemin : aux instants intermédiaires, a et b
// déplacés par la caméra y montrent tous deux le décor, la garde ne le verrait pas.
std::vector<std::uint8_t> rife_zone(const kernels::SceneMotion& s) {
  std::vector<std::uint8_t> zone(s.tiles.size(), 0);
  const double px = static_cast<double>(s.tile_px);
  for (int r = 0; r < s.rows; ++r) {
    for (int c = 0; c < s.cols; ++c) {
      const kernels::SceneTile& tile = s.at(c, r);
      if (!keeps_rife(tile)) continue;
      const int to_c = c + tiles_moved(tile.dx, px);
      const int to_r = r + tiles_moved(tile.dy, px);
      const int c0 = std::max(std::min(c, to_c) - 1, 0);
      const int c1 = std::min(std::max(c, to_c) + 1, s.cols - 1);
      const int r0 = std::max(std::min(r, to_r) - 1, 0);
      const int r1 = std::min(std::max(r, to_r) + 1, s.rows - 1);
      for (int rr = r0; rr <= r1; ++rr) {
        for (int cc = c0; cc <= c1; ++cc)
          zone[static_cast<std::size_t>(rr) * static_cast<std::size_t>(s.cols) +
               static_cast<std::size_t>(cc)] = 1;
      }
    }
  }
  return zone;
}

}  // namespace

std::string_view name(LayerSkip skip) {
  switch (skip) {
    case LayerSkip::NoMap:
      return "sans carte";
    case LayerSkip::NoCamera:
      return "sans caméra";
    case LayerSkip::Cut:
      return "coupe ou fondu";
    case LayerSkip::Residual:
      return "décor hors du modèle de caméra";
    case LayerSkip::Warp:
      return "mouvement non inversible";
    case LayerSkip::FewTiles:
      return "trop peu de décor";
    case LayerSkip::Grid:
      return "grille trop grande";
  }
  return "?";
}

std::expected<PairLayers, LayerSkip> plan_layers(const std::optional<kernels::SceneMotion>& scene,
                                                 int work_width, int work_height) {
  if (!scene) return std::unexpected(LayerSkip::NoMap);
  const kernels::SceneMotion& s = *scene;
  if (s.cut_likely || s.fade) return std::unexpected(LayerSkip::Cut);
  if (!s.camera_fit) return std::unexpected(LayerSkip::NoCamera);
  if (!(s.residual <= kLayerResidualMax)) return std::unexpected(LayerSkip::Residual);
  const kernels::Affine camera{s.camera[0], s.camera[1], s.camera[2], s.camera[3], s.camera[4], s.camera[5]};
  if (!kernels::inverse(camera)) return std::unexpected(LayerSkip::Warp);
  const kernels::LayerGrid limits;
  if (!limits.fits(s.cols, s.rows) || s.tile_px <= 0 || work_width <= 0 || work_height <= 0 ||
      s.tiles.size() != static_cast<std::size_t>(s.cols) * static_cast<std::size_t>(s.rows))
    return std::unexpected(LayerSkip::Grid);
  PairLayers layers{.camera = camera,
                    .work_width = work_width,
                    .work_height = work_height,
                    .tile_px = s.tile_px,
                    .cols = s.cols,
                    .rows = s.rows,
                    .decor = std::vector<std::uint8_t>(s.tiles.size(), 0)};
  const std::vector<std::uint8_t> rife = rife_zone(s);
  std::size_t kept = 0;
  for (std::size_t i = 0; i < s.tiles.size(); ++i) {
    if (rife[i] != 0 || !sure_decor(s.tiles[i])) continue;
    layers.decor[i] = 1;
    ++kept;
  }
  layers.share = static_cast<double>(kept) / static_cast<double>(s.tiles.size());
  if (layers.share < kLayerMinShare) return std::unexpected(LayerSkip::FewTiles);
  return layers;
}

kernels::LayerGrid layer_grid(const PairLayers& layers, int out_width, int out_height) {
  kernels::LayerGrid grid{
      .cols = layers.cols,
      .rows = layers.rows,
      .tile_w = static_cast<float>(static_cast<double>(layers.tile_px) * out_width / layers.work_width),
      .tile_h = static_cast<float>(static_cast<double>(layers.tile_px) * out_height / layers.work_height)};
  for (int r = 0; r < layers.rows; ++r) {
    for (int c = 0; c < layers.cols; ++c) {
      if (layers.decor[static_cast<std::size_t>(r) * static_cast<std::size_t>(layers.cols) +
                       static_cast<std::size_t>(c)] != 0)
        grid.set(c, r);
    }
  }
  return grid;
}

std::optional<kernels::LayerParams> pair_params(const PairLayers& layers, int out_width, int out_height,
                                                double t, bool cubic) {
  const kernels::Affine camera =
      kernels::to_output(layers.camera, layers.work_width, layers.work_height, out_width, out_height);
  const auto shifts = kernels::layer_shifts(camera, t);
  if (!shifts) return std::nullopt;
  kernels::LayerParams params{.grid = layer_grid(layers, out_width, out_height), .count = 2, .cubic = cubic};
  params.sources[0] = {.shift = kernels::to_float(shifts->a), .weight = static_cast<float>(1.0 - t)};
  params.sources[1] = {.shift = kernels::to_float(shifts->b), .weight = static_cast<float>(t)};
  return params;
}

double plate_cover(const kernels::Affine& key_to_b, int width, int height) {
  const std::optional<kernels::Affine> back = kernels::inverse(key_to_b);
  if (!back) return 0.0;
  constexpr int kAcross = 16;
  constexpr int kDown = 9;
  int inside = 0;
  for (int j = 0; j < kDown; ++j) {
    for (int i = 0; i < kAcross; ++i) {
      const double x = (i + 0.5) * width / kAcross;
      const double y = (j + 0.5) * height / kDown;
      const double kx = (*back)[0] + (*back)[1] * x + (*back)[2] * y;
      const double ky = (*back)[3] + (*back)[4] * x + (*back)[5] * y;
      if (kx >= 0.0 && ky >= 0.0 && kx <= width - 1.0 && ky <= height - 1.0) ++inside;
    }
  }
  return static_cast<double>(inside) / (kAcross * kDown);
}

PlateStep PlateTrack::step(const std::expected<PairLayers, LayerSkip>& pair, bool continues) {
  if (!continues) forget();
  if (!pair) {
    // Sans caméra sûre, le mouvement cumulé depuis la clé est perdu.
    forget();
    return {};
  }
  if (has_key_) {
    key_to_b_ = kernels::compose(pair->camera, key_to_b_);
    ++age_;
    const double zoom = std::abs(kernels::determinant(key_to_b_) - 1.0);
    if (age_ <= kPlateMaxAge && zoom <= kPlateMaxZoom &&
        plate_cover(key_to_b_, pair->work_width, pair->work_height) >= kPlateMinCover)
      return {.action = PlateAction::Use, .key_to_b = key_to_b_};
  }
  // La première paire sûre du plan, ou une plaque usée : b devient la clé. Ses tuiles décor sont celles de la
  // paire qui l'amène (sur la grille de a ; le chemin des dessins jusqu'à b en est retiré).
  has_key_ = true;
  key_to_b_ = kernels::kIdentityAffine;
  age_ = 0;
  key_ = *pair;
  return {.action = PlateAction::Refresh};
}

std::optional<kernels::LayerParams> plate_params(const PairLayers& key, const PairLayers& b_layers,
                                                 const kernels::Affine& key_to_b, int out_width,
                                                 int out_height, bool cubic) {
  const kernels::Affine moved =
      kernels::to_output(key_to_b, key.work_width, key.work_height, out_width, out_height);
  const std::optional<kernels::Affine> shift = kernels::plate_shift(moved);
  if (!shift) return std::nullopt;
  kernels::LayerParams params{.grid = layer_grid(b_layers, out_width, out_height),
                              .count = 1,
                              .source_grid = layer_grid(key, out_width, out_height),
                              .guard_low = kernels::kPlateGuardLow,
                              .guard_high = kernels::kPlateGuardHigh,
                              .cubic = cubic};
  params.sources[0] = {.shift = kernels::to_float(*shift), .weight = 1.0F};
  return params;
}

}  // namespace kirari::app::play
