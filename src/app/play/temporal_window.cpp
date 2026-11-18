#include "app/play/temporal_window.h"

#include <algorithm>

namespace ajp::app::play {

std::optional<std::size_t> TemporalWindow::find(const DecodedFrame& frame,
                                                const gpu::Reduction& reduce) const {
  for (std::size_t i = 0; i < raws_.size(); ++i) {
    const Raw& r = raws_[i];
    if (r.filled && r.id == frame.id && r.epoch == frame.epoch && r.reduce == reduce) return i;
  }
  return std::nullopt;
}

std::size_t TemporalWindow::claim(const DecodedFrame& frame, const gpu::Reduction& reduce, bool debanded) {
  for (Raw& r : raws_) {
    if (r.filled && r.epoch != frame.epoch) r.filled = false;  // un seek : l'époque d'avant ne sert plus
  }
  std::size_t chosen = raws_.size();
  for (std::size_t i = 0; i < raws_.size(); ++i) {
    if (!raws_[i].filled || !(raws_[i].reduce == reduce)) {
      chosen = i;
      break;
    }
    if (chosen == raws_.size() || raws_[i].pts_us < raws_[chosen].pts_us) chosen = i;
  }
  raws_[chosen] = Raw{.id = frame.id,
                      .epoch = frame.epoch,
                      .pts_us = frame.pts_us,
                      .reduce = reduce,
                      .debanded = debanded,
                      .filled = true};
  return chosen;
}

std::array<std::size_t, kTemporalFixFrames> TemporalWindow::window(std::size_t center, bool* mirrored) const {
  const Raw& middle = raws_[center];
  // Les voisines de la même époque et de la même réduction, de part et d'autre du pts du milieu, les plus
  // proches d'abord.
  std::vector<std::size_t> before;
  std::vector<std::size_t> after;
  for (std::size_t i = 0; i < raws_.size(); ++i) {
    const Raw& r = raws_[i];
    if (i == center || !r.filled || r.epoch != middle.epoch || !(r.reduce == middle.reduce)) continue;
    if (r.pts_us < middle.pts_us) before.push_back(i);
    if (r.pts_us > middle.pts_us) after.push_back(i);
  }
  std::ranges::sort(before, [&](std::size_t a, std::size_t b) { return raws_[a].pts_us > raws_[b].pts_us; });
  std::ranges::sort(after, [&](std::size_t a, std::size_t b) { return raws_[a].pts_us < raws_[b].pts_us; });
  constexpr std::size_t kRadius = kTemporalFixRadius;
  std::array<std::optional<std::size_t>, kRadius> past{};
  std::array<std::optional<std::size_t>, kRadius> next{};
  for (std::size_t k = 0; k < kRadius; ++k) {
    if (k < before.size()) past[k] = before[k];
    if (k < after.size()) next[k] = after[k];
  }
  bool reflected = false;
  // Le reflet de vs_temporalfix : la voisine k manquante d'un côté est la voisine k de l'autre côté ; sans
  // elle non plus, la voisine k - 1 de son propre côté (déjà comblée), et le milieu pour la première.
  const auto fill = [&](std::array<std::optional<std::size_t>, kRadius>& side,
                        const std::array<std::optional<std::size_t>, kRadius>& other) {
    std::array<std::optional<std::size_t>, kRadius> filled = side;
    for (std::size_t k = 0; k < kRadius; ++k) {
      if (filled[k]) continue;
      reflected = true;
      filled[k] = other[k] ? other[k] : k > 0 ? filled[k - 1] : std::optional(center);
    }
    return filled;
  };
  const auto past_filled = fill(past, next);
  const auto next_filled = fill(next, past);
  std::array<std::size_t, kTemporalFixFrames> out{};
  for (std::size_t k = 0; k < kRadius; ++k) {
    out[kRadius - 1 - k] = *past_filled[k];
    out[kRadius + 1 + k] = *next_filled[k];
  }
  out[kRadius] = center;
  if (mirrored != nullptr) *mirrored = reflected;
  return out;
}

void TemporalWindow::clear() {
  for (Raw& r : raws_) r.filled = false;
}

}  // namespace ajp::app::play
