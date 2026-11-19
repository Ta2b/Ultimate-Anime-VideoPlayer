#include "render/effect_mask.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace kirari::render {
namespace {

constexpr std::int64_t kDefaultSpanUs = 41'708;  // une image à 23,976 i/s

std::size_t at(int col, int row, int cols) {
  return (static_cast<std::size_t>(row) * static_cast<std::size_t>(cols)) + static_cast<std::size_t>(col);
}

// La plus forte preuve d'effet des voisines (3 × 3, la tuile comprise), kEffectUnsure si aucune ne tranche.
float neighbour_effect(const EffectTiles& tiles, int col, int row) {
  float best = kEffectUnsure;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const int c = col + dx;
      const int r = row + dy;
      if (c < 0 || r < 0 || c >= tiles.cols || r >= tiles.rows) continue;
      best = std::max(best, tiles.effect[at(c, r, tiles.cols)]);
    }
  }
  return best;
}

}  // namespace

float effect_step(std::int64_t span_us, double tau_us) {
  const double span = static_cast<double>(span_us > 0 ? span_us : kDefaultSpanUs);
  return static_cast<float>(1.0 - std::exp(-span / std::max(tau_us, 1.0)));
}

void EffectMask::reset() {
  cols_ = 0;
  rows_ = 0;
  level_.clear();
  shown_.clear();
  ++serial_;
}

void EffectMask::update(const EffectTiles& tiles) {
  const auto count =
      static_cast<std::size_t>(std::max(tiles.cols, 0)) * static_cast<std::size_t>(std::max(tiles.rows, 0));
  if (count == 0 || tiles.effect.size() != count) {
    // Sans carte : une coupe ne laisse rien du plan d'avant (le grade d'aujourd'hui), sinon retour lent.
    if (tiles.cut) {
      if (!level_.empty()) reset();
      return;
    }
    relax(tiles.span_us);
    return;
  }
  // Une autre grille (autre taille de travail) : on repart du neutre.
  if (tiles.cols != cols_ || tiles.rows != rows_) {
    cols_ = tiles.cols;
    rows_ = tiles.rows;
    level_.assign(count, 1.0F);
  }
  cover_x_ = std::max(tiles.cover_x, 1.0);
  cover_y_ = std::max(tiles.cover_y, 1.0);
  if (tiles.hold && !tiles.cut) return;  // rien à apprendre : le masque reste tel quel
  const float rise = effect_step(tiles.span_us, kEffectRiseUs);
  const float fall = effect_step(tiles.span_us, kEffectFallUs);
  for (int row = 0; row < rows_; ++row) {
    for (int col = 0; col < cols_; ++col) {
      const std::size_t i = at(col, row, cols_);
      const float own = tiles.effect[i];
      const float near = neighbour_effect(tiles, col, row);
      float& level = level_[i];
      if (tiles.cut) {
        // Une coupe : le masque du nouveau plan d'un coup ; une tuile qui ne tranche pas repart du neutre.
        level = own >= 0.0F ? std::clamp(std::max(own, near), 0.0F, 1.0F) : 1.0F;
        continue;
      }
      // Une tuile sans verdict ne fait que monter vers un effet voisin.
      const float target = own >= 0.0F ? std::max(own, near) : near;
      if (target < 0.0F || (own < 0.0F && target <= level)) continue;
      level += (target - level) * (target > level ? rise : fall);
    }
  }
  publish();
}

void EffectMask::relax(std::int64_t span_us) {
  if (level_.empty()) return;
  const float fall = effect_step(span_us, kEffectFallUs);
  bool all = true;
  for (float& level : level_) {
    level += (1.0F - level) * fall;
    if (1.0F - level >= kEffectNeutral) all = false;
  }
  if (all) {
    reset();
    return;
  }
  publish();
}

// Le flou 3 × 3 (1 2 1), borné aux tuiles de l'image.
void EffectMask::publish() {
  constexpr std::array<float, 3> kWeights{1.0F, 2.0F, 1.0F};
  shown_.assign(level_.size(), 1.0F);
  for (int row = 0; row < rows_; ++row) {
    for (int col = 0; col < cols_; ++col) {
      float sum = 0.0F;
      float weight = 0.0F;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int c = col + dx;
          const int r = row + dy;
          if (c < 0 || r < 0 || c >= cols_ || r >= rows_) continue;
          const int wx = dx + 1;
          const int wy = dy + 1;
          const float w = kWeights[static_cast<std::size_t>(wx)] * kWeights[static_cast<std::size_t>(wy)];
          sum += w * level_[at(c, r, cols_)];
          weight += w;
        }
      }
      shown_[at(col, row, cols_)] = std::clamp(sum / weight, 0.0F, 1.0F);
    }
  }
  ++serial_;
}

bool EffectMask::neutral() const {
  return std::ranges::all_of(shown_, [](float v) { return 1.0F - v < kEffectNeutral; });
}

}  // namespace kirari::render
