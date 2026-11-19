// Les tuiles qui bougent (upscale_tiles.h).
#include "app/play/upscale_tiles.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <utility>

namespace kirari::app::play {
namespace {

// Un rectangle de pixels de travail, [x0, x1) × [y0, y1).
struct Rect {
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  Rect moved(float dx, float dy) const {
    // Arrondi vers l'extérieur : un mouvement fractionnaire touche les deux tuiles voisines.
    const auto lo = [](int v, float d) { return static_cast<int>(std::floor(static_cast<float>(v) + d)); };
    const auto hi = [](int v, float d) { return static_cast<int>(std::ceil(static_cast<float>(v) + d)); };
    return Rect{lo(x0, dx), lo(y0, dy), hi(x1, dx), hi(y1, dy)};
  }
};

float norm2(float x, float y) {
  return x * x + y * y;
}

}  // namespace

void SpanTiles::set_grid(const kernels::SpanTileGrid& grid) {
  grid_ = grid;
  tiles_.assign(grid.usable() ? static_cast<std::size_t>(grid.count()) : 0U, Tile{});
  since_full_ = 0;
  known_ = false;
}

template <class Visit>
void SpanTiles::scene_tiles_under(const kernels::SceneMotion& scene, int col, int row, Visit&& visit) const {
  if (scene.tile_px <= 0) return;
  const int x0 = col * grid_.tile_w, x1 = x0 + grid_.tile_w;
  const int y0 = row * grid_.tile_h, y1 = y0 + grid_.tile_h;
  const int c0 = std::max(0, x0 / scene.tile_px), c1 = std::min(scene.cols - 1, (x1 - 1) / scene.tile_px);
  const int r0 = std::max(0, y0 / scene.tile_px), r1 = std::min(scene.rows - 1, (y1 - 1) / scene.tile_px);
  for (int r = r0; r <= r1; ++r)
    for (int c = c0; c <= c1; ++c) visit(scene.at(c, r));
}

TilePlan SpanTiles::full() {
  for (Tile& tile : tiles_) tile = Tile{};
  since_full_ = 0;
  known_ = true;
  return TilePlan{};
}

bool SpanTiles::camera_shift(const kernels::SceneMotion& scene, int& shift_x, int& shift_y) const {
  shift_x = 0;
  shift_y = 0;
  if (!scene.camera_fit || scene.residual > kShiftResidual) return false;
  const auto& c = scene.camera;
  const double linear =
      std::max({std::abs(c[1] - 1.0), std::abs(c[2]), std::abs(c[4]), std::abs(c[5] - 1.0)});
  if (linear > kernels::scene::kHeldLinear) return false;
  const double step = grid_.shift_step;
  const double sx = std::round(c[0] / step) * step, sy = std::round(c[3] / step) * step;
  if (std::abs(c[0] - sx) > kShiftSnap || std::abs(c[3] - sy) > kShiftSnap) return false;
  shift_x = static_cast<int>(sx);
  shift_y = static_cast<int>(sy);
  return shift_x != 0 || shift_y != 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TilePlan SpanTiles::plan(const TileFacts& facts) {
  if (!active()) return TilePlan{};
  ++planned_;
  ++since_full_;
  const kernels::SceneMotion* scene = facts.scene;
  if (scene == nullptr) ++no_map_;
  if (!facts.fresh || !known_ || scene == nullptr || scene->cut_likely || scene->fade ||
      scene->tile_px <= 0 || since_full_ >= kFullEvery)
    return full();
  int sx = 0, sy = 0;
  const bool shifting = facts.can_shift && camera_shift(*scene, sx, sy);
  if (!shifting && !scene->background_held) {
    // La caméra bouge sans décalage entier possible : tout le décor change.
    if (scene->camera_fit) return full();
  }
  const int cols = grid_.cols, rows = grid_.rows;
  const auto index = [cols](int c, int r) {
    return static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(c);
  };
  // Les tuiles du moteur que touche un rectangle de pixels de travail.
  const auto mark = [&](std::vector<std::uint8_t>& set, const Rect& rect) {
    const int c0 = std::max(0, rect.x0 / grid_.tile_w), c1 = std::min(cols - 1, (rect.x1 - 1) / grid_.tile_w);
    const int r0 = std::max(0, rect.y0 / grid_.tile_h), r1 = std::min(rows - 1, (rect.y1 - 1) / grid_.tile_h);
    if (rect.x1 <= 0 || rect.y1 <= 0) return;
    for (int r = r0; r <= r1; ++r)
      for (int c = c0; c <= c1; ++c) set[index(c, r)] = 1;
  };
  std::vector<std::uint8_t> dirty(tiles_.size(), 0);
  // La dérive de chaque tuile du moteur à cette paire : la plus grande des tuiles de la carte qui la
  // couvrent.
  std::vector<std::pair<float, float>> step(tiles_.size(), {0.0F, 0.0F});
  const float fx = static_cast<float>(sx), fy = static_cast<float>(sy);
  for (int r = 0; r < scene->rows; ++r) {
    for (int c = 0; c < scene->cols; ++c) {
      const kernels::SceneTile& s = scene->at(c, r);
      const Rect at{c * scene->tile_px, r * scene->tile_px, (c + 1) * scene->tile_px,
                    (r + 1) * scene->tile_px};
      const float ex = s.dx - fx, ey = s.dy - fy;
      const bool follows = shifting ? s.kind == kernels::SceneClass::Background && s.confidence >= kSure &&
                                          norm2(ex, ey) <= kFollow * kFollow
                                    : !s.changed;
      if (!follows) {
        // Là où était le contenu, là où il va avec la caméra, et là où il va de lui-même.
        mark(dirty, at);
        mark(dirty, at.moved(fx, fy));
        mark(dirty, at.moved(s.dx, s.dy));
        continue;
      }
      // Le contenu suit : sa dérive va aux tuiles où il arrive.
      const Rect to = at.moved(fx, fy);
      const int c0 = std::max(0, to.x0 / grid_.tile_w), c1 = std::min(cols - 1, (to.x1 - 1) / grid_.tile_w);
      const int r0 = std::max(0, to.y0 / grid_.tile_h), r1 = std::min(rows - 1, (to.y1 - 1) / grid_.tile_h);
      if (to.x1 <= 0 || to.y1 <= 0) continue;
      for (int y = r0; y <= r1; ++y)
        for (int x = c0; x <= c1; ++x) {
          auto& d = step[index(x, y)];
          if (norm2(ex, ey) > norm2(d.first, d.second)) d = {ex, ey};
        }
    }
  }
  // L'état de chaque tuile après le décalage : celui des tuiles d'où vient son contenu (la pire), ou à
  // recalculer si ce contenu sort de la grille.
  std::vector<Tile> moved(tiles_.size());
  std::vector<std::uint8_t> core(tiles_.size(), 0);
  const int max_x = (cols - 1) * grid_.tile_w, max_y = (rows - 1) * grid_.tile_h;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const std::size_t i = index(c, r);
      const int x0 = c * grid_.tile_w - sx, y0 = r * grid_.tile_h - sy;
      const int x1 = x0 + grid_.tile_w, y1 = y0 + grid_.tile_h;
      // Un contenu qui entre par la droite ou par le bas viendrait de la dernière colonne ou ligne, tronquée
      // : au-delà du bord réel, le décalage n'a pas de source.
      const bool shifted_in =
          shifting && (x0 < 0 || y0 < 0 || (sx < 0 && x1 > max_x) || (sy < 0 && y1 > max_y));
      if (shifted_in) {
        core[i] = 1;
        continue;
      }
      Tile from;
      if (shifting) {
        const int c0 = x0 / grid_.tile_w, c1 = (x1 - 1) / grid_.tile_w;
        const int r0 = y0 / grid_.tile_h, r1 = (y1 - 1) / grid_.tile_h;
        for (int y = r0; y <= r1; ++y)
          for (int x = c0; x <= c1; ++x) {
            const Tile& t = tiles_[index(x, y)];
            from.age = std::max(from.age, t.age);
            if (norm2(t.drift_x, t.drift_y) > norm2(from.drift_x, from.drift_y)) {
              from.drift_x = t.drift_x;
              from.drift_y = t.drift_y;
            }
          }
      } else {
        from = tiles_[i];
      }
      from.age += 1;
      from.drift_x += step[i].first;
      from.drift_y += step[i].second;
      moved[i] = from;
      if (dirty[i] != 0 || from.age > kMaxAge || norm2(from.drift_x, from.drift_y) >= kDriftMax * kDriftMax)
        core[i] = 1;
    }
  }
  const std::vector<std::uint8_t> wide =
      kernels::dilate_span_tiles(core, cols, rows, grid_.reach_cols, grid_.reach_rows);
  const int wide_count = kernels::count_span_tiles(wide);
  if (static_cast<double>(wide_count) > kMaxShare * static_cast<double>(tiles_.size())) {
    ++too_wide_;
    return full();
  }
  for (std::size_t i = 0; i < tiles_.size(); ++i) tiles_[i] = core[i] != 0 ? Tile{} : moved[i];
  ++tiled_;
  if (shifting) ++shifted_;
  tiles_done_ += static_cast<std::uint64_t>(kernels::count_span_tiles(core));
  tiles_wide_ += static_cast<std::uint64_t>(wide_count);
  tiles_seen_ += tiles_.size();
  return TilePlan{.tiled = true, .shift_x = sx, .shift_y = sy, .core = std::move(core)};
}

void SpanTiles::held(const kernels::SceneMotion* scene) {
  if (!active() || !known_) return;
  ++since_full_;
  for (int r = 0; r < grid_.rows; ++r) {
    for (int c = 0; c < grid_.cols; ++c) {
      Tile& tile = tiles_[static_cast<std::size_t>(r) * static_cast<std::size_t>(grid_.cols) +
                          static_cast<std::size_t>(c)];
      ++tile.age;
      if (scene == nullptr) continue;
      float wx = 0.0F, wy = 0.0F;
      scene_tiles_under(*scene, c, r, [&](const kernels::SceneTile& s) {
        if (norm2(s.dx, s.dy) > norm2(wx, wy)) {
          wx = s.dx;
          wy = s.dy;
        }
      });
      tile.drift_x += wx;
      tile.drift_y += wy;
    }
  }
}

void SpanTiles::refused() {
  ++refused_;
  if (tiled_ > 0) --tiled_;
  full();
}

std::string SpanTiles::summary() const {
  if (tiled_ == 0) return {};
  const auto share = [](std::uint64_t part, std::uint64_t whole) {
    return whole > 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
  };
  return std::format(
      "SPAN par tuiles : {} images sur {} calculées ({:.1f} %), {:.1f} % des tuiles recalculées ({:.1f} % "
      "avec le halo), {} décalées sur un panoramique ; images entières : {} sans carte, {} région trop "
      "large, {} refus du moteur",
      tiled_, planned_, share(tiled_, planned_), share(tiles_done_, tiles_seen_),
      share(tiles_wide_, tiles_seen_), shifted_, no_map_, too_wide_, refused_);
}

}  // namespace kirari::app::play
