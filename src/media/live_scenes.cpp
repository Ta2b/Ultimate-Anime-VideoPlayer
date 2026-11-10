#include "media/live_scenes.h"

#include <algorithm>
#include <iterator>

namespace ajp::media {

void LiveScenes::push(std::int64_t pts_us, const kernels::FrameStats& stats) {
  const ShotFrame frame = shot_frame_of(pts_us, stats);
  const std::scoped_lock lock(mutex_);
  const bool jump = previous_ && pts_us <= previous_->pts_us;
  if (!previous_ || jump || frame_distance(*previous_, frame) > kLiveCut) {
    ++serial_;
    frames_ = 0;
    histogram_ = {};
    light_ = {};
    mean_ = blue_ = chroma_ = 0.0F;
  }
  if (jump) kept_.clear();
  const bool flashing = flashes_.push(pts_us, zone_nits_of(frame));
  previous_ = frame;
  // Moyenne cumulée de la scène : chaque image y entre avec le poids 1 / (images vues).
  ++frames_;
  const float w = 1.0F / static_cast<float>(frames_);
  const auto mix = [w](float& into, float value) { into += w * (value - into); };
  for (std::size_t i = 0; i < histogram_.size(); ++i) mix(histogram_.at(i), frame.histogram.at(i));
  mix(mean_, frame.mean);
  mix(blue_, frame.blue);
  mix(chroma_, frame.chroma);
  for (std::size_t i = 0; i < light_.maxrgb_nits.size(); ++i)
    mix(light_.maxrgb_nits.at(i), frame.light.maxrgb_nits.at(i));
  mix(light_.apl_nits, frame.light.apl_nits);
  mix(light_.maxrgb_mean_nits, frame.light.maxrgb_mean_nits);
  mix(light_.bright, frame.light.bright);
  mix(light_.highlights, frame.light.highlights);
  mix(light_.skin, frame.light.skin);
  for (std::size_t i = 0; i < light_.lum_hist.size(); ++i)
    mix(light_.lum_hist.at(i), frame.light.lum_hist.at(i));
  ShotStats scene = stats_of(histogram_, mean_, blue_, chroma_);
  scene.apl_nits = light_.apl_nits;
  kept_.push_back({.pts_us = pts_us,
                   .scene = {.stats = scene, .light = light_, .serial = serial_, .flashing = flashing}});
  while (kept_.size() > kLiveKept) kept_.pop_front();
}

std::optional<LiveScene> LiveScenes::at(std::int64_t pts_us, std::int64_t max_gap_us) const {
  const std::scoped_lock lock(mutex_);
  const auto after = std::ranges::upper_bound(kept_, pts_us, {}, &Entry::pts_us);
  if (after == kept_.begin()) return std::nullopt;
  const Entry& found = *std::prev(after);
  if (pts_us - found.pts_us > max_gap_us) return std::nullopt;
  return found.scene;
}

void LiveScenes::missed(std::int64_t pts_us) {
  const std::scoped_lock lock(mutex_);
  if (flashes_.missed(pts_us) && !kept_.empty()) kept_.back().scene.flashing = true;
}

void LiveScenes::set_flash_gain(double gain) {
  const std::scoped_lock lock(mutex_);
  flashes_.set_gain(gain);
}

void LiveScenes::clear() {
  const std::scoped_lock lock(mutex_);
  kept_.clear();
  previous_.reset();
  flashes_.reset();
  frames_ = 0;
  ++serial_;
}

}  // namespace ajp::media
