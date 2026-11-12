#include "media/shot_timeline.h"

#include <algorithm>
#include <cstddef>
#include <iterator>

namespace ajp::media {
namespace {

bool overlaps(const Shot& a, const Shot& b) {
  return a.start_us < b.end_us && b.start_us < a.end_us;
}

float lerp(float a, float b, float t) {
  return a + (b - a) * t;
}

ShotStats blend(const ShotStats& a, const ShotStats& b, float t) {
  return ShotStats{.mean = lerp(a.mean, b.mean, t),
                   .p05 = lerp(a.p05, b.p05, t),
                   .p50 = lerp(a.p50, b.p50, t),
                   .p95 = lerp(a.p95, b.p95, t),
                   .p995 = lerp(a.p995, b.p995, t),
                   .dark = lerp(a.dark, b.dark, t),
                   .bright = lerp(a.bright, b.bright, t),
                   .blue = lerp(a.blue, b.blue, t),
                   .chroma = lerp(a.chroma, b.chroma, t),
                   .lit = lerp(a.lit, b.lit, t),
                   .apl_nits = lerp(a.apl_nits, b.apl_nits, t)};
}

}  // namespace

void ShotTimeline::add(const Shot& found) {
  constexpr std::int64_t kEdgeUs = 250'000;
  const std::scoped_lock lock(mutex_);
  // Un plan qui déborde de quelques images sur un voisin déjà trouvé (l'analyse arrêtée à l'entrée d'une
  // zone connue) s'arrête à lui au lieu de le remplacer.
  Shot shot = found;
  for (const Shot& s : shots_) {
    if (!overlaps(s, shot) || (s.start_us >= shot.start_us && s.end_us <= shot.end_us)) continue;
    if (std::min(s.end_us, shot.end_us) - std::max(s.start_us, shot.start_us) >= kEdgeUs) continue;
    if (s.start_us > shot.start_us) {
      shot.end_us = s.start_us;
    } else {
      shot.start_us = s.end_us;
    }
  }
  if (shot.end_us <= shot.start_us) return;
  // Un plan coupé par un départ en cours de vidéo ne remplace pas un plan entier déjà trouvé : il n'en garde
  // que la part qui ne recouvre rien (A42-03 : jeté entier, il laissait pour toujours le trou d'un plan clos
  // à la frontière d'un saut). Un plan entier remplace ceux qu'il recouvre (la même zone, analysée à
  // nouveau après un saut).
  if (!shot.cut_before) {
    for (const Shot& s : shots_) {
      if (!overlaps(s, shot)) continue;
      if (s.start_us <= shot.start_us) {
        shot.start_us = s.end_us;
      } else {
        shot.end_us = s.start_us;
      }
      if (shot.end_us <= shot.start_us) return;
    }
  }
  for (std::size_t i = shots_.size(); i-- > 0;) {
    if (overlaps(shots_.at(i), shot)) {
      shots_.erase(shots_.begin() + static_cast<std::ptrdiff_t>(i));
      serials_.erase(serials_.begin() + static_cast<std::ptrdiff_t>(i));
    }
  }
  const auto at = std::ranges::upper_bound(shots_, shot.start_us, {}, &Shot::start_us);
  const auto index = at - shots_.begin();
  shots_.insert(at, shot);
  serials_.insert(serials_.begin() + index, next_serial_++);
  ++version_;
}

void ShotTimeline::add_flash(FlashSpan span) {
  if (span.until_us <= span.from_us) return;
  const std::scoped_lock lock(mutex_);
  for (std::size_t i = flashes_.size(); i-- > 0;) {
    const FlashSpan& known = flashes_.at(i);
    if (known.until_us < span.from_us || span.until_us < known.from_us) continue;
    if (known.from_us <= span.from_us && known.until_us >= span.until_us) return;  // déjà connue
    span = {.from_us = std::min(span.from_us, known.from_us),
            .until_us = std::max(span.until_us, known.until_us)};
    flashes_.erase(flashes_.begin() + static_cast<std::ptrdiff_t>(i));
  }
  flashes_.insert(std::ranges::upper_bound(flashes_, span.from_us, {}, &FlashSpan::from_us), span);
  ++version_;
}

bool ShotTimeline::flashing(std::int64_t pts_us) const {
  const std::scoped_lock lock(mutex_);
  const auto after = std::ranges::upper_bound(flashes_, pts_us, {}, &FlashSpan::from_us);
  return after != flashes_.begin() && pts_us < std::prev(after)->until_us;
}

std::vector<FlashSpan> ShotTimeline::flashes() const {
  const std::scoped_lock lock(mutex_);
  return flashes_;
}

std::uint64_t ShotTimeline::version() const {
  const std::scoped_lock lock(mutex_);
  return version_;
}

std::optional<ShotAt> ShotTimeline::at(std::int64_t pts_us) const {
  const std::scoped_lock lock(mutex_);
  const auto after = std::ranges::upper_bound(shots_, pts_us, {}, &Shot::start_us);
  if (after == shots_.begin()) return std::nullopt;
  const auto index = static_cast<std::size_t>(after - shots_.begin()) - 1;
  const Shot& shot = shots_.at(index);
  if (pts_us >= shot.end_us) return std::nullopt;
  ShotAt found{.start_us = shot.start_us,
               .end_us = shot.end_us,
               .serial = serials_.at(index),
               .stats = shot.first,
               .fade = shot.fade,
               .cut_before = shot.cut_before,
               .skins = shot.meta.skins};
  if (shot.fade) {
    // La part où la lumière change (B42-03) ; tout le plan si elle n'est pas connue.
    const bool ramp = shot.ramp_to_us > shot.ramp_from_us;
    const std::int64_t from = ramp ? shot.ramp_from_us : shot.start_us;
    const std::int64_t to = ramp ? shot.ramp_to_us : shot.end_us;
    const auto t =
        static_cast<float>(pts_us - from) / static_cast<float>(std::max<std::int64_t>(to - from, 1));
    found.stats = blend(shot.first, shot.last, std::clamp(t, 0.0F, 1.0F));
  }
  return found;
}

std::int64_t ShotTimeline::covered_us() const {
  const std::scoped_lock lock(mutex_);
  std::int64_t sum = 0;
  for (const Shot& s : shots_) sum += s.end_us - s.start_us;
  return sum;
}

std::int64_t ShotTimeline::analyzed_until(std::int64_t pts_us) const {
  const std::scoped_lock lock(mutex_);
  std::int64_t until = pts_us;
  for (const Shot& s : shots_) {
    if (s.end_us <= until) continue;
    if (s.start_us > until) break;
    until = s.end_us;
  }
  return until;
}

std::int64_t ShotTimeline::first_gap_us() const {
  constexpr std::int64_t kSlackUs = 100'000;
  const std::scoped_lock lock(mutex_);
  std::int64_t until = origin_us_;
  for (const Shot& s : shots_) {
    if (s.start_us > until + kSlackUs) break;
    until = std::max(until, s.end_us);
  }
  return until;
}

void ShotTimeline::set_origin_us(std::int64_t pts_us) {
  const std::scoped_lock lock(mutex_);
  origin_us_ = std::max<std::int64_t>(pts_us, 0);
}

std::int64_t ShotTimeline::end_us() const {
  const std::scoped_lock lock(mutex_);
  return shots_.empty() ? 0 : shots_.back().end_us;
}

std::size_t ShotTimeline::size() const {
  const std::scoped_lock lock(mutex_);
  return shots_.size();
}

std::vector<Shot> ShotTimeline::shots() const {
  const std::scoped_lock lock(mutex_);
  return shots_;
}

void ShotTimeline::clear() {
  const std::scoped_lock lock(mutex_);
  shots_.clear();
  serials_.clear();
  flashes_.clear();
  ++version_;
}

}  // namespace ajp::media
