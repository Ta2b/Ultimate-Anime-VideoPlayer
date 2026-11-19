// Les tuiles qui bougent (upscale_tiles.h).
#include "app/play/upscale_tiles.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <utility>

#include "kernels/spec/detect.h"

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

// Les tuiles de la grille que touche un rectangle de pixels de travail, par leur indice.
template <class F>
void for_tiles(const kernels::SpanTileGrid& g, const Rect& rect, F&& f) {
  if (rect.x1 <= 0 || rect.y1 <= 0) return;
  const int c0 = std::max(0, rect.x0 / g.tile_w), c1 = std::min(g.cols - 1, (rect.x1 - 1) / g.tile_w);
  const int r0 = std::max(0, rect.y0 / g.tile_h), r1 = std::min(g.rows - 1, (rect.y1 - 1) / g.tile_h);
  for (int r = r0; r <= r1; ++r)
    for (int c = c0; c <= c1; ++c)
      f(static_cast<std::size_t>(r) * static_cast<std::size_t>(g.cols) + static_cast<std::size_t>(c));
}

// Ce que la carte dit à chaque tuile du moteur : à recalculer (dirty), ou la dérive et le glissement de
// luminosité reçus à cette paire (les plus grands des tuiles de la carte qui y arrivent).
struct SceneRead {
  std::vector<std::uint8_t> dirty;
  std::vector<std::pair<float, float>> step;
  std::vector<float> bias;
};

// Le plus grand en valeur absolue.
float wider(float a, float b) {
  return std::abs(b) > std::abs(a) ? b : a;
}

SceneRead read_scene(const kernels::SpanTileGrid& g, const kernels::SceneMotion& scene, int sx, int sy,
                     bool shifting) {
  const auto count = static_cast<std::size_t>(g.count());
  SceneRead read{.dirty = std::vector<std::uint8_t>(count, 0),
                 .step = std::vector<std::pair<float, float>>(count, {0.0F, 0.0F}),
                 .bias = std::vector<float>(count, 0.0F)};
  const float fx = static_cast<float>(sx), fy = static_cast<float>(sy);
  const auto dirty = [&read](std::size_t i) { read.dirty[i] = 1; };
  for (int r = 0; r < scene.rows; ++r) {
    for (int c = 0; c < scene.cols; ++c) {
      const kernels::SceneTile& s = scene.at(c, r);
      const int px = scene.tile_px;
      const Rect at{c * px, r * px, (c + 1) * px, (r + 1) * px};
      const float ex = s.dx - fx, ey = s.dy - fy;
      // Sous un décalage, seul le décor sûr qui le suit se reprend ; sinon, ce que la carte dit inchangé.
      const bool follows = shifting ? s.kind == kernels::SceneClass::Background &&
                                          s.confidence >= SpanTiles::kSure &&
                                          norm2(ex, ey) <= SpanTiles::kFollow * SpanTiles::kFollow
                                    : !s.changed;
      if (!follows) {
        // Là où était le contenu, là où il va avec la caméra, et là où il va de lui-même.
        for_tiles(g, at, dirty);
        for_tiles(g, at.moved(fx, fy), dirty);
        for_tiles(g, at.moved(s.dx, s.dy), dirty);
        continue;
      }
      for_tiles(g, at.moved(fx, fy), [&](std::size_t i) {
        auto& d = read.step[i];
        if (norm2(ex, ey) > norm2(d.first, d.second)) d = {ex, ey};
        read.bias[i] = wider(read.bias[i], s.shift);
      });
    }
  }
  return read;
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

std::uint64_t* SpanTiles::whole_reason(const TileFacts& facts, bool shifting, bool new_recipe) {
  if (facts.stale != TileStale::None) return &stale_[static_cast<std::size_t>(facts.stale)];
  if (!known_) return &stale_[static_cast<std::size_t>(TileStale::Restart)];
  if (new_recipe) return &recipe_;
  const kernels::SceneMotion* scene = facts.scene;
  if (scene == nullptr) return &no_map_;
  if (since_full_ >= kFullEvery) return &refresh_;
  if (scene->cut_likely || scene->fade || scene->tile_px <= 0) return &cut_;
  // La caméra bouge sans décalage entier possible : tout le décor change.
  if (!shifting && !scene->background_held && scene->camera_fit) return &camera_;
  return nullptr;
}

SpanTiles::Tile SpanTiles::carried(int col, int row, int sx, int sy) const {
  if (sx == 0 && sy == 0) return tiles_[at(col, row)];
  // Le contenu vient de la tuile décalée : la pire des tuiles qu'elle touche.
  const int x0 = col * grid_.tile_w - sx, y0 = row * grid_.tile_h - sy;
  Tile from;
  for_tiles(grid_, Rect{x0, y0, x0 + grid_.tile_w, y0 + grid_.tile_h}, [&](std::size_t i) {
    const Tile& t = tiles_[i];
    from.age = std::max(from.age, t.age);
    if (norm2(t.drift_x, t.drift_y) > norm2(from.drift_x, from.drift_y)) {
      from.drift_x = t.drift_x;
      from.drift_y = t.drift_y;
    }
    from.bias = wider(from.bias, t.bias);
  });
  return from;
}

bool SpanTiles::enters(int col, int row, int sx, int sy) const {
  if (sx == 0 && sy == 0) return false;
  const int x0 = col * grid_.tile_w - sx, y0 = row * grid_.tile_h - sy;
  const int x1 = x0 + grid_.tile_w, y1 = y0 + grid_.tile_h;
  // Un contenu qui entre par la droite ou par le bas viendrait de la dernière colonne ou ligne, tronquée :
  // au-delà du bord réel, le décalage n'a pas de source.
  const int max_x = (grid_.cols - 1) * grid_.tile_w, max_y = (grid_.rows - 1) * grid_.tile_h;
  return x0 < 0 || y0 < 0 || (sx < 0 && x1 > max_x) || (sy < 0 && y1 > max_y);
}

std::vector<std::uint8_t> SpanTiles::reach_of(const std::vector<std::uint8_t>& raw, int spread_px) const {
  return kernels::dilate_span_tiles(raw, grid_.cols, grid_.rows,
                                    grid_.reach_cols + kernels::span_reach_tiles(spread_px, grid_.tile_w),
                                    grid_.reach_rows + kernels::span_reach_tiles(spread_px, grid_.tile_h));
}

TilePlan SpanTiles::plan(const TileFacts& facts) {
  if (!active()) return TilePlan{};
  ++planned_;
  ++since_full_;
  const bool new_recipe = known_ && facts.input_key != input_key_;
  input_key_ = facts.input_key;
  int sx = 0, sy = 0;
  const bool shifting = facts.scene != nullptr && facts.can_shift && camera_shift(*facts.scene, sx, sy);
  if (std::uint64_t* reason = whole_reason(facts, shifting, new_recipe); reason != nullptr) {
    ++*reason;
    return full();
  }
  const SceneRead read = read_scene(grid_, *facts.scene, sx, sy, shifting);
  // L'état de chaque tuile après le décalage, et ce qui a changé : ce qui entre, les bords de l'axe décalé,
  // ce que la carte dit changé, ce qui a trop vieilli, dérivé ou glissé.
  std::vector<Tile> moved(tiles_.size());
  std::vector<std::uint8_t> raw(tiles_.size(), 0);
  for (int r = 0; r < grid_.rows; ++r) {
    for (int c = 0; c < grid_.cols; ++c) {
      const std::size_t i = at(c, r);
      const bool edge =
          (sx != 0 && (c == 0 || c == grid_.cols - 1)) || (sy != 0 && (r == 0 || r == grid_.rows - 1));
      if (edge || enters(c, r, sx, sy)) {
        raw[i] = 1;
        continue;
      }
      Tile from = carried(c, r, sx, sy);
      from.age += 1;
      from.drift_x += read.step[i].first;
      from.drift_y += read.step[i].second;
      from.bias += read.bias[i];
      moved[i] = from;
      const bool stale = from.age > kMaxAge || norm2(from.drift_x, from.drift_y) >= kDriftMax * kDriftMax ||
                         std::abs(from.bias) >= kBiasMax;
      raw[i] = read.dirty[i] != 0 || stale ? 1 : 0;
    }
  }
  blocks_into(raw, facts.blocks, shifting);
  // Le cœur : ce qui a changé, et toute sortie qui le voit ; la région : le cœur, et tout ce qu'il lit.
  std::vector<std::uint8_t> core = reach_of(raw, facts.spread_px);
  const std::vector<std::uint8_t> wide =
      kernels::dilate_span_tiles(core, grid_.cols, grid_.rows, grid_.reach_cols, grid_.reach_rows);
  const int wide_count = kernels::count_span_tiles(wide);
  if (static_cast<double>(wide_count) > kMaxShare * static_cast<double>(tiles_.size())) {
    ++too_wide_;
    return full();
  }
  for (std::size_t i = 0; i < tiles_.size(); ++i) tiles_[i] = core[i] != 0 ? Tile{} : moved[i];
  last_ = Counted{.core = static_cast<std::uint64_t>(kernels::count_span_tiles(core)),
                  .wide = static_cast<std::uint64_t>(wide_count),
                  .shifted = shifting};
  ++tiled_;
  if (shifting) ++shifted_;
  tiles_done_ += last_.core;
  tiles_wide_ += last_.wide;
  tiles_seen_ += tiles_.size();
  return TilePlan{.tiled = true, .shift_x = sx, .shift_y = sy, .core = std::move(core)};
}

void SpanTiles::blocks_into(std::vector<std::uint8_t>& raw, const PairBlocks& blocks, bool shifting) {
  if (shifting || blocks.cols == 0 || blocks.side_px <= 0.0F) return;
  std::uint64_t added = 0;
  const auto mark = [&raw, &added](std::size_t i) {
    if (raw[i] == 0) ++added;
    raw[i] = 1;
  };
  for (std::size_t i = 0; i < blocks.deviation.size(); ++i) {
    if (kernels::is_held(blocks.deviation[i], kBlockChange)) continue;
    const std::size_t col_index = i % blocks.cols, row_index = i / blocks.cols;
    const auto col = static_cast<float>(col_index), row = static_cast<float>(row_index);
    // Arrondi vers l'extérieur, comme Rect::moved : un bloc à cheval touche les deux tuiles.
    const Rect at{static_cast<int>(std::floor(col * blocks.side_px)),
                  static_cast<int>(std::floor(row * blocks.side_px)),
                  static_cast<int>(std::ceil((col + 1.0F) * blocks.side_px)),
                  static_cast<int>(std::ceil((row + 1.0F) * blocks.side_px))};
    for_tiles(grid_, at, mark);
  }
  if (added == 0) return;
  ++net_frames_;
  net_tiles_ += added;
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
      float wx = 0.0F, wy = 0.0F, bias = 0.0F;
      scene_tiles_under(*scene, c, r, [&](const kernels::SceneTile& s) {
        if (norm2(s.dx, s.dy) > norm2(wx, wy)) {
          wx = s.dx;
          wy = s.dy;
        }
        bias = wider(bias, s.shift);
      });
      tile.drift_x += wx;
      tile.drift_y += wy;
      tile.bias += bias;
    }
  }
}

void SpanTiles::refused() {
  ++refused_;
  // L'image se calcule entière : elle ne compte plus parmi celles par tuiles.
  if (tiled_ > 0) {
    --tiled_;
    tiles_done_ -= last_.core;
    tiles_wide_ -= last_.wide;
    tiles_seen_ -= tiles_.size();
    if (last_.shifted) --shifted_;
  }
  last_ = Counted{};
  full();
}

void SpanTiles::checked(std::uint32_t worst_block, double mean_levels) {
  ++checks_;
  check_worst_ = std::max(check_worst_, worst_block);
  check_mean_ = std::max(check_mean_, mean_levels);
}

std::string SpanTiles::summary() const {
  if (!active()) return {};
  const auto share = [](std::uint64_t part, std::uint64_t whole) {
    return whole > 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
  };
  std::string line = std::format("SPAN par tuiles : {} images sur {} calculées ({:.1f} %)", tiled_, planned_,
                                 share(tiled_, planned_));
  if (tiled_ > 0)
    line += std::format(
        ", {:.1f} % des tuiles recalculées ({:.1f} % avec le halo), {} décalées sur un panoramique",
        share(tiles_done_, tiles_seen_), share(tiles_wide_, tiles_seen_), shifted_);
  if (net_frames_ > 0)
    line += std::format(", le filet des blocs a vu {} tuiles que la carte ne disait pas changées, sur {} images",
                        net_tiles_, net_frames_);
  const auto stale = [this](TileStale why) { return stale_[static_cast<std::size_t>(why)]; };
  const std::array<std::pair<std::uint64_t, const char*>, 11> reasons{{
      {no_map_, "sans carte"},
      {too_wide_, "région trop large"},
      {refused_, "refus du moteur"},
      {stale(TileStale::Graded), "sortie d'avant gradée sur place"},
      {stale(TileStale::Overwritten), "sortie d'avant écrasée (image créée, source sautée)"},
      {stale(TileStale::Chain), "chaîne sans tuiles (export, YUV, TemporalFix, Smooth, plaque)"},
      {stale(TileStale::Restart), "début, saut ou image abîmée"},
      {recipe_, "entrée de SPAN changée"},
      {refresh_, "rafraîchissement"},
      {cut_, "coupe ou fondu"},
      {camera_, "caméra sans décalage entier"},
  }};
  std::string whole;
  for (const auto& [count, why] : reasons) {
    if (count == 0) continue;
    whole += std::format("{}{} {}", whole.empty() ? "" : ", ", count, why);
  }
  line += " ; images entières : " + (whole.empty() ? std::string("aucune") : whole);
  if (checks_ > 0)
    line += std::format(
        " ; contrôle sur {} images contre un calcul complet : pire bloc {:.2f} niveau, écart moyen {:.3f} "
        "niveau au plus",
        checks_, static_cast<double>(check_worst_) / 256.0, check_mean_);
  return line;
}

}  // namespace kirari::app::play
