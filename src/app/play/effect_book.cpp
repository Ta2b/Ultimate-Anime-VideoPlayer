#include "app/play/effect_book.h"

#include <algorithm>
#include <utility>

namespace kirari::app::play {

render::EffectTiles effect_tiles_of(const kernels::SceneMotion& scene, int work_width, int work_height,
                                    std::int64_t span_us, bool cut) {
  render::EffectTiles out;
  out.span_us = span_us;
  out.cut = cut || scene.cut_likely;
  out.hold = scene.drawing_held || scene.fade;
  const auto count =
      static_cast<std::size_t>(std::max(scene.cols, 0)) * static_cast<std::size_t>(std::max(scene.rows, 0));
  if (count == 0 || scene.tiles.size() != count || scene.tile_px <= 0 || work_width <= 0 || work_height <= 0)
    return out;  // vide : le masque revient vers le neutre
  out.cols = scene.cols;
  out.rows = scene.rows;
  out.cover_x = static_cast<double>(scene.cols * scene.tile_px) / work_width;
  out.cover_y = static_cast<double>(scene.rows * scene.tile_px) / work_height;
  out.effect.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    const kernels::SceneTile& tile = scene.tiles[i];
    const float sure = std::clamp(tile.confidence, 0.0F, 1.0F);
    switch (tile.kind) {
      case kernels::SceneClass::Effect:
        out.effect[i] = 1.0F;
        break;
      case kernels::SceneClass::Background:
      case kernels::SceneClass::Drawing:
        out.effect[i] = 1.0F - sure;
        break;
      case kernels::SceneClass::Uncertain:
        out.effect[i] = tile.changed ? render::kEffectUnsure : 0.0F;
        break;
    }
  }
  return out;
}

void EffectBook::add(EffectNote note) {
  const std::scoped_lock lock(mutex_);
  notes_[next_] = std::move(note);
  next_ = (next_ + 1) % kKept;
  count_ = std::min(count_ + 1, kKept);
}

std::optional<EffectNote> EffectBook::at(std::uint32_t epoch, std::int64_t pts_us) const {
  const std::scoped_lock lock(mutex_);
  for (std::size_t k = 1; k <= count_; ++k) {
    const EffectNote& note = notes_[(next_ + kKept - k) % kKept];
    if (note.epoch == epoch && pts_us > note.from_us && pts_us <= note.to_us) return note;
  }
  return std::nullopt;
}

void EffectBook::clear() {
  const std::scoped_lock lock(mutex_);
  for (EffectNote& note : notes_) note.tiles.reset();
  count_ = 0;
  next_ = 0;
}

bool EffectFollower::show(const std::optional<EffectNote>& note, std::int64_t pts_us) {
  if (!note || !note->tiles) {
    // Pas de carte pour l'image montrée : vers le neutre, au temps écoulé entre deux images montrées (une
    // image représentée ne compte pas ; un pts inconnu ou qui recule, comme une image à 23,976 i/s).
    seen_ = false;
    if (mask_.neutral()) {
      last_pts_ = pts_us;
      return false;
    }
    if (last_pts_ && *last_pts_ == pts_us) return false;
    const std::int64_t span = last_pts_ && pts_us > *last_pts_ ? pts_us - *last_pts_ : 0;
    last_pts_ = pts_us;
    const std::uint64_t before = mask_.serial();
    mask_.relax(span);
    return mask_.serial() != before;
  }
  last_pts_.reset();
  // Les images d'une même paire (interpolées, représentées) ne la relisent pas.
  if (seen_ && epoch_ == note->epoch && from_us_ == note->from_us && to_us_ == note->to_us) return false;
  if (epoch_ && *epoch_ != note->epoch)
    mask_.reset();  // saut ou rechargement : le masque d'avant ne dit rien
  epoch_ = note->epoch;
  from_us_ = note->from_us;
  to_us_ = note->to_us;
  seen_ = true;
  const std::uint64_t before = mask_.serial();
  mask_.update(*note->tiles);
  return mask_.serial() != before;
}

void EffectFollower::reset() {
  mask_.reset();
  epoch_.reset();
  seen_ = false;
  last_pts_.reset();
}

}  // namespace kirari::app::play
