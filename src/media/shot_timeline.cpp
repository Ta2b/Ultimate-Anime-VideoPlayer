#include "media/shot_timeline.h"

#include <algorithm>
#include <cstddef>

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
                   .chroma = lerp(a.chroma, b.chroma, t)};
}

}  // namespace

void ShotTimeline::add(const Shot& shot) {
  const std::scoped_lock lock(mutex_);
  // Un plan coupé par un départ en cours de vidéo ne remplace pas un plan entier déjà trouvé ; un plan
  // entier remplace ceux qu'il recouvre (la même zone, analysée à nouveau après un saut).
  const bool covered = std::ranges::any_of(shots_, [&](const Shot& s) { return overlaps(s, shot); });
  if (covered && !shot.cut_before) return;
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
               .cut_before = shot.cut_before};
  if (shot.fade) {
    const auto t =
        static_cast<float>(pts_us - shot.start_us) / static_cast<float>(shot.end_us - shot.start_us);
    found.stats = blend(shot.first, shot.last, std::clamp(t, 0.0F, 1.0F));
  }
  return found;
}

std::optional<double> ShotTimeline::mean_luma() const {
  const std::scoped_lock lock(mutex_);
  double sum = 0.0, weight = 0.0;
  for (const Shot& s : shots_) {
    const auto d = static_cast<double>(s.end_us - s.start_us);
    sum += d * 0.5 * static_cast<double>(s.first.mean + s.last.mean);
    weight += d;
  }
  if (weight <= 0.0) return std::nullopt;
  return sum / weight;
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

std::size_t ShotTimeline::size() const {
  const std::scoped_lock lock(mutex_);
  return shots_.size();
}

void ShotTimeline::clear() {
  const std::scoped_lock lock(mutex_);
  shots_.clear();
  serials_.clear();
}

}  // namespace ajp::media
