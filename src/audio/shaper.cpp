#include "audio/shaper.h"

#include <algorithm>
#include <cstddef>

#include "audio/converter.h"

namespace ajp::audio {

void OutputShaper::process(std::span<float> samples, const Pass& pass) {
  if (pass.channels <= 0 || pass.sample_rate <= 0) return;
  const auto channels = static_cast<std::size_t>(pass.channels);
  const std::size_t frames = samples.size() / channels;
  const bool limit = pass.force_limit || pass.gain > 1.0F || applied_ > 1.0F;
  // Le cas courant : gain 1 établi, rien à faire.
  if (!limit && !pass.fade_in && pass.gain == applied_ && pass.gain == 1.0F) return;
  const auto ramp = static_cast<std::size_t>(std::max(1, pass.sample_rate * kRampMs / 1000));
  const float from = applied_;
  for (std::size_t frame = 0; frame < frames; ++frame) {
    const float progress = frame < ramp ? static_cast<float>(frame + 1) / static_cast<float>(ramp) : 1.0F;
    const float gain = from + ((pass.gain - from) * progress);
    const float fade = pass.fade_in ? progress : 1.0F;
    for (std::size_t channel = 0; channel < channels; ++channel) {
      float& sample = samples[(frame * channels) + channel];
      sample *= gain * fade;
      if (limit) sample = soft_limit(sample);
    }
  }
  applied_ = pass.gain;
}

}  // namespace ajp::audio
