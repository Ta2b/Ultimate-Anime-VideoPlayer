#include "audio/shaper.h"

#include <algorithm>
#include <cstddef>

#include "audio/converter.h"

namespace ajp::audio {

void OutputShaper::process(std::span<float> samples, const Pass& pass) {
  if (pass.channels <= 0 || pass.sample_rate <= 0) return;
  const auto channels = static_cast<std::size_t>(pass.channels);
  const std::size_t frames = samples.size() / channels;
  const auto ramp = static_cast<std::size_t>(std::max(1, pass.sample_rate * kRampMs / 1000));
  // Une nouvelle cible repart du gain atteint, pas de l'ancienne cible ; un fondu demandé recommence.
  if (pass.gain != target_) {
    ramp_from_ = applied_;
    target_ = pass.gain;
    ramp_done_ = 0;
  }
  if (pass.fade_in) {
    fading_ = true;
    fade_done_ = 0;
  }
  const bool ramping = ramp_done_ < ramp && applied_ != target_;
  const bool limit = pass.force_limit || pass.gain > 1.0F || applied_ > 1.0F;
  // Le cas courant : gain 1 établi, rien à faire.
  if (!limit && !fading_ && !ramping && target_ == 1.0F && applied_ == 1.0F) return;
  for (std::size_t frame = 0; frame < frames; ++frame) {
    float gain = target_;
    if (ramp_done_ < ramp) {
      ++ramp_done_;
      gain = ramp_done_ == ramp ? target_
                                : ramp_from_ + ((target_ - ramp_from_) * static_cast<float>(ramp_done_) /
                                                static_cast<float>(ramp));
    }
    float fade = 1.0F;
    if (fading_) {
      fade = static_cast<float>(fade_done_ + 1) / static_cast<float>(ramp);
      if (++fade_done_ >= ramp) fading_ = false;
    }
    applied_ = gain;
    for (std::size_t channel = 0; channel < channels; ++channel) {
      float& sample = samples[(frame * channels) + channel];
      sample *= gain * fade;
      if (limit) sample = soft_limit(sample);
    }
  }
}

}  // namespace ajp::audio
