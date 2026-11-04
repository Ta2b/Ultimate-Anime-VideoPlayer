#include "app/play/cut_book.h"

#include <algorithm>

namespace ajp::app::play {
namespace {

const media::Shot* shot_of(std::span<const media::Shot> shots, std::int64_t pts_us) {
  const auto after = std::ranges::upper_bound(shots, pts_us, {}, &media::Shot::start_us);
  if (after == shots.begin()) return nullptr;
  const media::Shot& shot = *(after - 1);
  return pts_us < shot.end_us ? &shot : nullptr;
}

}  // namespace

std::optional<bool> known_cut(std::span<const media::Shot> shots, std::int64_t a_us, std::int64_t b_us) {
  if (b_us <= a_us) return std::nullopt;
  const media::Shot* a = shot_of(shots, a_us);
  const media::Shot* b = shot_of(shots, b_us);
  if (a == nullptr || b == nullptr) return std::nullopt;
  if (a == b) return false;
  // Entre a et b, chaque plan doit commencer à une coupe connue : la dernière suffit à dire « coupe », mais
  // un plan intermédiaire sans coupe connue laisse un trou dans la chronologie.
  for (const media::Shot* s = a + 1; s <= b; ++s) {
    if (s->start_us != (s - 1)->end_us || !s->cut_before) return std::nullopt;
  }
  return true;
}

void CutBook::set_shots(std::span<const media::Shot> shots) {
  const std::scoped_lock lock(mutex_);
  shots_.assign(shots.begin(), shots.end());
}

std::optional<bool> CutBook::cut_between(std::int64_t a_us, std::int64_t b_us) const {
  const std::scoped_lock lock(mutex_);
  return known_cut(shots_, a_us, b_us);
}

bool CutBook::active() const {
  const std::scoped_lock lock(mutex_);
  return !shots_.empty();
}

}  // namespace ajp::app::play
