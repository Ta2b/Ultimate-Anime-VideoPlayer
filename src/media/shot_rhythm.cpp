#include "media/shot_rhythm.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace kirari::media {

RhythmPair with_flow(RhythmPair pair, const FlowSummary& flow, int height) {
  if (height < 1 || !std::isfinite(flow.shift_x) || !std::isfinite(flow.shift_y) ||
      !std::isfinite(flow.residual) || !std::isfinite(flow.unmatched))
    return pair;
  const auto lines = static_cast<float>(height);
  pair.flow = true;
  pair.camera = std::hypot(flow.shift_x, flow.shift_y) / lines;
  pair.drawing = std::max(flow.residual, 0.0F) / lines;
  pair.unmatched = std::clamp(flow.unmatched, 0.0F, 1.0F);
  return pair;
}

std::optional<RhythmPair> rhythm_pair_of(std::span<const float> previous, std::span<const float> luma,
                                         int width, int height) {
  if (width < 1 || height < 1) return std::nullopt;
  const auto points = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (previous.size() != points || luma.size() != points) return std::nullopt;
  constexpr int kSide = rhythm::kBlockPoints;
  const int cols = (width + kSide - 1) / kSide;
  const int rows = (height + kSide - 1) / kSide;
  int moving = 0;
  int blocks = 0;
  double grain = 0.0;
  for (int by = 0; by < rows; ++by) {
    for (int bx = 0; bx < cols; ++bx) {
      double sum = 0.0;
      int count = 0;
      for (int y = by * kSide; y < std::min(height, (by + 1) * kSide); ++y) {
        for (int x = bx * kSide; x < std::min(width, (bx + 1) * kSide); ++x) {
          const auto at = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) +
                          static_cast<std::size_t>(x);
          sum += std::abs(static_cast<double>(luma[at]) - static_cast<double>(previous[at]));
          ++count;
        }
      }
      const double codes = 255.0 * sum / static_cast<double>(count);
      ++blocks;
      if (codes >= rhythm::kMoveCodes) {
        ++moving;
      } else {
        grain += codes;
      }
    }
  }
  RhythmPair pair;
  pair.changed = static_cast<float>(moving) / static_cast<float>(blocks);
  const int still = blocks - moving;
  pair.grain = still > 0 ? static_cast<float>(grain / still) : 0.0F;
  return pair;
}

bool held_pair(const RhythmPair& pair) {
  if (pair.flow) return pair.drawing < rhythm::kHeldDrawing && pair.camera < rhythm::kHeldCamera;
  return pair.changed < rhythm::kHeldChanged;
}

bool strong_pair(const RhythmPair& pair) {
  if (held_pair(pair)) return false;
  if (pair.flow && (pair.drawing >= rhythm::kStrongDrawing || pair.unmatched >= rhythm::kStrongUnmatched))
    return true;
  // Sans le flux, un grand changement ; avec lui, un panoramique change toute l'image sans être fort : seuls
  // les dessins comptent.
  return !pair.flow && pair.changed >= rhythm::kStrongChanged;
}

float drawing_cadence(const ShotRhythm& rhythm) {
  if (rhythm.pairs == 0) return 0.0F;
  const std::uint32_t drawn = rhythm.pairs - std::min(rhythm.held, rhythm.pairs);
  // Aucun dessin nouveau sur tout le plan (un plan fixe) : autant d'images que de paires pour un dessin.
  return static_cast<float>(rhythm.pairs) / static_cast<float>(std::max<std::uint32_t>(drawn, 1));
}

void RhythmTotals::add(const RhythmPair& pair) {
  ++pairs_;
  const bool held = held_pair(pair);
  if (held) ++held_;
  if (strong_pair(pair)) ++strong_;
  if (!held) changed_ += pair.changed;
  grain_ += pair.grain;
  still_ += 1.0 - pair.changed;
  if (pair.flow) {
    ++flow_;
    camera_ += pair.camera;
    drawing_ += pair.drawing;
    unmatched_ += pair.unmatched;
  }
}

ShotRhythm RhythmTotals::rhythm() const {
  ShotRhythm out{.pairs = pairs_, .held = held_, .strong = strong_, .flow_pairs = flow_};
  if (pairs_ == 0) return out;
  const auto n = static_cast<double>(pairs_);
  const std::uint32_t drawn = pairs_ - held_;
  out.changed = drawn > 0 ? static_cast<float>(changed_ / drawn) : 0.0F;
  out.grain = static_cast<float>(grain_ / n);
  out.still = static_cast<float>(still_ / n);
  if (flow_ > 0) {
    const auto f = static_cast<double>(flow_);
    out.camera = static_cast<float>(camera_ / f);
    out.drawing = static_cast<float>(drawing_ / f);
    out.unmatched = static_cast<float>(unmatched_ / f);
  }
  return out;
}

std::optional<RhythmPair> RhythmProbe::push(std::vector<float> luma, int width, int height) {
  std::optional<RhythmPair> pair;
  if (width == width_ && height == height_) pair = rhythm_pair_of(previous_, luma, width, height);
  previous_ = std::move(luma);
  width_ = width;
  height_ = height;
  return pair;
}

}  // namespace kirari::media
