// Essai de tuiles_rejeu : les tuiles décidées par l'écart à l'ancre composée (ancre_tuiles.h).
#include "ancre_tuiles.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <format>

#include "app/play/upscale_reuse.h"
#include "kernels/spec/detect.h"

namespace kirari::bench {

using app::play::SpanReuse;
using app::play::SpanTiles;

BlockGap block_gap(const kernels::Plane<std::uint8_t>& a, const kernels::Plane<std::uint8_t>& b, int side) {
  BlockGap gap;
  gap.cols = (a.width + side - 1) / side;
  const int rows = (a.height + side - 1) / side;
  gap.deviation.assign(static_cast<std::size_t>(gap.cols) * static_cast<std::size_t>(rows), 0U);
  for (int by = 0; by < rows; ++by) {
    for (int bx = 0; bx < gap.cols; ++bx) {
      std::uint64_t sum = 0, n = 0;
      for (int y = by * side; y < std::min(a.height, (by + 1) * side); ++y) {
        for (int x = bx * side; x < std::min(a.width, (bx + 1) * side); ++x) {
          sum +=
              static_cast<std::uint64_t>(std::abs(int{a.data[a.index(x, y)]} - int{b.data[b.index(x, y)]}));
          ++n;
        }
      }
      // Des codes 8 bits de la plage de la source : 256 par niveau (spec/detect.h, pair_levels).
      const auto dev = static_cast<std::uint32_t>(sum * 256U / n);
      gap.deviation[static_cast<std::size_t>(by) * static_cast<std::size_t>(gap.cols) +
                    static_cast<std::size_t>(bx)] = dev;
      gap.worst = std::max(gap.worst, dev);
    }
  }
  return gap;
}

AnchorTiles::AnchorTiles(const kernels::SpanTileGrid& grid, int divisor, int side, int spread_px,
                         std::uint32_t threshold, Bound bound, const kernels::Plane<std::uint8_t>& first)
    : grid_(grid),
      divisor_(divisor),
      side_(side),
      spread_px_(spread_px),
      threshold_(threshold),
      bound_(bound),
      composed_(first),
      previous_(first),
      anchor_(first),
      age_(static_cast<std::size_t>(grid.count()), 0) {}

char AnchorTiles::full(const kernels::Plane<std::uint8_t>& frame, std::uint64_t& reason) {
  ++reason;
  composed_ = frame;
  anchor_ = frame;
  std::ranges::fill(sums_, 0U);
  std::ranges::fill(frozen_, 0U);
  std::ranges::fill(age_, 0);
  since_full_ = 0;
  in_row_ = 0;
  core_.assign(age_.size(), 1);
  return 'C';
}

BlockGap AnchorTiles::gap_of(const kernels::Plane<std::uint8_t>& frame) {
  if (bound_ == Bound::Composed) return block_gap(frame, composed_, side_);
  BlockGap pair = block_gap(frame, previous_, side_);
  previous_ = frame;
  if (sums_.empty()) sums_.assign(pair.deviation.size(), 0U);
  if (frozen_.empty()) frozen_.assign(pair.deviation.size(), 0U);
  const BlockGap anchored = bound_ == Bound::Anchored ? block_gap(frame, anchor_, side_) : BlockGap{};
  pair.worst = 0;
  for (std::size_t b = 0; b < sums_.size(); ++b) {
    sums_[b] += pair.deviation[b];
    if (bound_ == Bound::Anchored) sums_[b] = std::min(sums_[b], frozen_[b] + anchored.deviation[b]);
    pair.deviation[b] = sums_[b];
    pair.worst = std::max(pair.worst, sums_[b]);
  }
  return pair;
}

void AnchorTiles::computed(std::size_t i, const kernels::Plane<std::uint8_t>& frame) {
  age_[i] = 0;
  const int c = static_cast<int>(i % static_cast<std::size_t>(grid_.cols));
  const int r = static_cast<int>(i / static_cast<std::size_t>(grid_.cols));
  const int x0 = c * grid_.tile_w / divisor_, x1 = std::min(frame.width, (c + 1) * grid_.tile_w / divisor_);
  const int y0 = r * grid_.tile_h / divisor_, y1 = std::min(frame.height, (r + 1) * grid_.tile_h / divisor_);
  for (int y = y0; y < y1; ++y)
    for (int x = x0; x < x1; ++x) composed_.data[composed_.index(x, y)] = frame.data[frame.index(x, y)];
  if (sums_.empty()) return;
  // Les blocs de la tuile (un bloc de 16 pixels de travail n'est jamais à cheval sur deux tuiles).
  const int cols = (frame.width + side_ - 1) / side_;
  for (int by = y0 / side_; by * side_ < y1; ++by)
    for (int bx = x0 / side_; bx * side_ < x1; ++bx)
      sums_[static_cast<std::size_t>(by) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(bx)] =
          0U;
}

// Les tuiles du moteur dont un bloc de la mesure passe le seuil, ou trop vieilles.
std::vector<std::uint8_t> AnchorTiles::raw_of(const BlockGap& gap) const {
  std::vector<std::uint8_t> raw(age_.size(), 0);
  const int block = side_ * divisor_;  // côté d'un bloc, pixels de travail
  const int rows = static_cast<int>(gap.deviation.size()) / gap.cols;
  for (int r = 0; r < grid_.rows; ++r) {
    for (int c = 0; c < grid_.cols; ++c) {
      const std::size_t i =
          static_cast<std::size_t>(r) * static_cast<std::size_t>(grid_.cols) + static_cast<std::size_t>(c);
      bool dirty = age_[i] + 1 > SpanTiles::kMaxAge;
      const int bx0 = c * grid_.tile_w / block,
                bx1 = std::min(gap.cols - 1, ((c + 1) * grid_.tile_w - 1) / block);
      const int by0 = r * grid_.tile_h / block,
                by1 = std::min(rows - 1, ((r + 1) * grid_.tile_h - 1) / block);
      for (int by = by0; by <= by1 && !dirty; ++by)
        for (int bx = bx0; bx <= bx1 && !dirty; ++bx)
          dirty = !kernels::is_held(
              gap.deviation[static_cast<std::size_t>(by) * static_cast<std::size_t>(gap.cols) +
                            static_cast<std::size_t>(bx)],
              threshold_);
      raw[i] = dirty ? 1 : 0;
    }
  }
  return raw;
}

char AnchorTiles::next(const kernels::Plane<std::uint8_t>& frame, const kernels::SceneMotion& scene) {
  ++seen_;
  ++since_full_;
  if (scene.cut_likely || scene.fade) return full(frame, cut_);
  const BlockGap gap = gap_of(frame);
  if (kernels::is_held(gap.worst, threshold_) && in_row_ < SpanReuse::kMaxInRow) {
    ++in_row_;
    ++reused_;
    for (int& a : age_) ++a;
    core_.assign(age_.size(), 0);
    return 'R';
  }
  if (in_row_ >= SpanReuse::kMaxInRow) ++limit_;
  in_row_ = 0;
  if (since_full_ >= SpanTiles::kFullEvery) return full(frame, refresh_);
  const std::vector<std::uint8_t> raw = raw_of(gap);
  const std::vector<std::uint8_t> core = kernels::dilate_span_tiles(
      raw, grid_.cols, grid_.rows, grid_.reach_cols + kernels::span_reach_tiles(spread_px_, grid_.tile_w),
      grid_.reach_rows + kernels::span_reach_tiles(spread_px_, grid_.tile_h));
  const std::vector<std::uint8_t> wide =
      kernels::dilate_span_tiles(core, grid_.cols, grid_.rows, grid_.reach_cols, grid_.reach_rows);
  const int wide_count = kernels::count_span_tiles(wide);
  if (static_cast<double>(wide_count) > SpanTiles::kMaxShare * static_cast<double>(grid_.count()))
    return full(frame, too_wide_);
  // Les tuiles recalculées recopient leurs pixels dans R ; les autres vieillissent.
  for (int r = 0; r < grid_.rows; ++r) {
    for (int c = 0; c < grid_.cols; ++c) {
      const std::size_t i =
          static_cast<std::size_t>(r) * static_cast<std::size_t>(grid_.cols) + static_cast<std::size_t>(c);
      if (core[i] == 0) {
        ++age_[i];
        continue;
      }
      computed(i, frame);
    }
  }
  anchor_ = frame;
  frozen_ = sums_;
  ++tiled_;
  tiles_core_ += static_cast<std::uint64_t>(kernels::count_span_tiles(core));
  core_ = core;
  tiles_wide_ += static_cast<std::uint64_t>(wide_count);
  return 'T';
}

std::string AnchorTiles::summary() const {
  const auto count = static_cast<double>(grid_.count());
  const auto share = [](double part, double whole) { return whole > 0.0 ? 100.0 * part / whole : 0.0; };
  const std::uint64_t whole = seen_ - reused_ - tiled_;
  // Le corps de SPAN calcule la région dilatée : son coût contre une image entière à chaque image.
  const double cost = share(static_cast<double>(whole) * count + static_cast<double>(tiles_wide_),
                            static_cast<double>(seen_) * count);
  return std::format(
      "{} (essai, seuil {:.2f} niveau) : sur {} images, {} reprises entières, {} par tuiles "
      "({:.1f} % "
      "des tuiles recalculées, {:.1f} % avec le halo), {} entières ({} région trop large, {} coupe ou fondu, "
      "{} "
      "rafraîchissement) ; arrêts à la limite {} ; corps de SPAN : {:.1f} % du calcul de chaque image "
      "entière",
      bound_ == Bound::Composed ? "Ancre composée"
      : bound_ == Bound::Summed ? "Somme des paires"
                                : "Somme bornée par l'ancre",
      static_cast<double>(threshold_) / 100.0, seen_, reused_, tiled_,
      share(static_cast<double>(tiles_core_), static_cast<double>(tiled_) * count),
      share(static_cast<double>(tiles_wide_), static_cast<double>(tiled_) * count), whole, too_wide_, cut_,
      refresh_, limit_, cost);
}

}  // namespace kirari::bench
