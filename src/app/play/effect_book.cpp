#include "app/play/effect_book.h"

#include <algorithm>
#include <cstdlib>
#include <utility>

#include "core/frame.h"

namespace kirari::app::play {

render::EffectTiles effect_tiles_of(const kernels::SceneMotion& scene, int work_width, int work_height,
                                    std::int64_t span_us, bool cut) {
  render::EffectTiles out;
  out.span_us = span_us;
  out.cut = cut || scene.cut_likely;
  out.hold = scene.drawing_held || scene.fade;
  // Une paire qui enjambe une coupe compare deux plans : NVOF y voit des effets là où il n'y en a pas (un
  // amas de 243 tuiles dans un fond sombre). Sans tuiles, la coupe remet le masque au neutre.
  if (out.cut) return out;
  const auto count =
      static_cast<std::size_t>(std::max(scene.cols, 0)) * static_cast<std::size_t>(std::max(scene.rows, 0));
  if (count == 0 || scene.tiles.size() != count || scene.tile_px <= 0 || work_width <= 0 || work_height <= 0)
    return out;  // vide : le masque revient vers le neutre
  out.cols = scene.cols;
  out.rows = scene.rows;
  out.cover_x = static_cast<double>(scene.cols * scene.tile_px) / work_width;
  out.cover_y = static_cast<double>(scene.rows * scene.tile_px) / work_height;
  out.effect.assign(count, 1.0F);
  out.sure.assign(count, 0.0F);
  for (std::size_t i = 0; i < count; ++i)
    if (scene.tiles[i].kind == kernels::SceneClass::Effect) out.sure[i] = 1.0F;
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

std::optional<EffectNote> EffectBook::for_source(std::uint32_t epoch, std::int64_t pts_us) const {
  const std::scoped_lock lock(mutex_);
  for (std::size_t k = 1; k <= count_; ++k) {
    const EffectNote& note = notes_[(next_ + kKept - k) % kKept];
    if (note.epoch != epoch) continue;
    if (pts_us != kNoPts && pts_us - note.to_us > kEffectStaleUs) return std::nullopt;
    return note;
  }
  return std::nullopt;
}

void EffectBook::clear() {
  const std::scoped_lock lock(mutex_);
  for (EffectNote& note : notes_) note.tiles.reset();
  count_ = 0;
  next_ = 0;
}

EffectSummary effect_summary(const render::EffectMask& mask, bool on) {
  EffectSummary out{.on = on && !mask.values().empty(), .cols = mask.cols(), .rows = mask.rows()};
  if (!out.on) return out;
  for (const float v : mask.values()) out.held += 1.0 - static_cast<double>(v);
  const auto tiles = static_cast<double>(mask.values().size());
  out.held /= tiles;
  out.sure =
      static_cast<double>(std::ranges::count_if(mask.sure(), [](float v) { return v > 0.5F; })) / tiles;
  return out;
}

void EffectTally::count(bool noted, bool on, const render::EffectMask& mask, std::int64_t pts_us) {
  ++images_;
  if (noted) ++noted_;
  const EffectSummary seen = effect_summary(mask, on);
  if (!seen.on) return;
  ++masked_;
  held_sum_ += seen.held;
  if (pts_us != kNoPts && seen.sure >= kTallyEffectShare)
    keep(Moment{.pts_us = pts_us, .effect = seen.sure, .held = seen.held});
}

void EffectTally::keep(Moment moment) {
  // Un moment plus fort à moins de 10 s l'emporte ; sinon celui-ci remplace les plus faibles de son
  // voisinage, puis le plus faible de tous.
  const auto near = [&](const Moment& kept) {
    return kept.effect > 0.0 && std::abs(kept.pts_us - moment.pts_us) < kTallySpacingUs;
  };
  for (const Moment& kept : moments_) {
    if (near(kept) && kept.effect >= moment.effect) return;
  }
  for (Moment& kept : moments_) {
    if (near(kept)) kept = Moment{};
  }
  std::ranges::sort(moments_, std::ranges::greater{}, &Moment::effect);
  if (moment.effect > moments_.back().effect) moments_.back() = moment;
  std::ranges::sort(moments_, std::ranges::greater{}, &Moment::effect);
}

bool EffectFollower::show(const std::optional<EffectNote>& note, std::int64_t pts_us,
                          std::optional<std::uint32_t> frame_epoch) {
  const std::uint64_t before = mask_.serial();
  if (frame_epoch) {
    if (epoch_ && *epoch_ != *frame_epoch) reset();
    epoch_ = frame_epoch;
  }
  if (!note || !note->tiles || (frame_epoch && note->epoch != *frame_epoch)) {
    // Pas de carte pour l'image montrée : vers le neutre, au temps écoulé entre deux images montrées (une
    // image représentée ne compte pas ; un pts inconnu ou qui recule, comme une image à 23,976 i/s).
    seen_ = false;
    if (mask_.neutral()) {
      last_pts_ = pts_us;
      return mask_.serial() != before;
    }
    if (last_pts_ && *last_pts_ == pts_us) return false;
    const std::int64_t span = last_pts_ && pts_us > *last_pts_ ? pts_us - *last_pts_ : 0;
    last_pts_ = pts_us;
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
  mask_.update(*note->tiles);
  return mask_.serial() != before;
}

bool EffectFollower::source(std::uint32_t epoch, const std::optional<EffectNote>& note) {
  const std::uint64_t before = mask_.serial();
  if (epoch_ && *epoch_ != epoch) reset();  // saut ou rechargement : le masque d'avant ne dit rien
  epoch_ = epoch;
  if (note && note->tiles && note->epoch == epoch) {
    show(note, note->to_us);
  } else {
    // Pas de note : une source de plus sans carte (la durée d'une image à 23,976 i/s, effect_step).
    seen_ = false;
    if (!mask_.neutral()) mask_.relax(0);
  }
  return mask_.serial() != before;
}

void EffectFollower::reset() {
  mask_.reset();
  epoch_.reset();
  seen_ = false;
  last_pts_.reset();
}

}  // namespace kirari::app::play
