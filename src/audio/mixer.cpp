#include "audio/mixer.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace ajp::audio {
namespace {

constexpr float kPi = std::numbers::pi_v<float>;
constexpr float kHeadRadiusM = 0.0875F;
constexpr float kSoundSpeed = 343.0F;
constexpr float kCenterLift = 1.26F;  // +2 dB sur le centre : les voix
constexpr float kDownmixScale = 0.6F;
constexpr float kCrossfeedGain = 0.32F;
constexpr float kCrossfeedCutoffHz = 700.0F;
constexpr float kCrossfeedDelayMs = 0.25F;
constexpr float kLfeGain = 0.3F;

// Les haut-parleurs dans l'ordre de SDL3 : azimut en degrés (négatif à gauche) ; le LFE n'a pas de place.
struct Speaker {
  float azimuth_deg;
  bool lfe;
};
constexpr std::array<Speaker, 6> k51{{{-30, false}, {30, false}, {0, false}, {0, true}, {-110, false}, {110, false}}};
constexpr std::array<Speaker, 8> k71{
    {{-30, false}, {30, false}, {0, false}, {0, true}, {-150, false}, {150, false}, {-90, false}, {90, false}}};

float one_pole(float cutoff_hz, int rate) {
  return 1.0F - std::exp(-2.0F * kPi * cutoff_hz / static_cast<float>(rate));
}

float soft_limit(float x) {
  const float magnitude = std::fabs(x);
  if (magnitude <= 0.9F) return x;
  return std::copysign(0.9F + 0.1F * std::tanh((magnitude - 0.9F) / 0.1F), x);
}

// Un échantillon par la ligne à retard puis le passe-bas, gain compris.
float run(MixPath& path, float input) {
  auto& line = path.ear.line;
  const auto size = static_cast<int>(line.size());
  line[static_cast<std::size_t>(path.ear.head)] = input;
  const int read = (path.ear.head - path.delay + size) % size;
  path.ear.head = (path.ear.head + 1) % size;
  const float delayed = line[static_cast<std::size_t>(read)];
  path.ear.low += path.alpha * (delayed - path.ear.low);
  return path.gain * path.ear.low;
}

}  // namespace

Mixer::Mixer(MixConfig config) : config_(config) {
  const int delay = std::clamp(static_cast<int>(std::lround(kCrossfeedDelayMs * 0.001F * static_cast<float>(config_.sample_rate))), 0,
                               kMixDelayMax - 1);
  for (MixPath& path : feed_) {
    path.delay = delay;
    path.gain = kCrossfeedGain;
    path.alpha = one_pole(kCrossfeedCutoffHz, config_.sample_rate);
  }
}

int Mixer::decoder_channels(int source_channels) const {
  const bool processed = !config_.faithful && config_.listen != Listen::Cinema && config_.sink_channels == 2 &&
                         (source_channels == 6 || source_channels == 8);
  return processed ? source_channels : config_.sink_channels;
}

void Mixer::reset() {
  for (auto& pair : paths_)
    for (MixPath& path : pair) path.ear = MixEar{};
  for (MixPath& path : feed_) path.ear = MixEar{};
}

void Mixer::set_faithful(bool faithful) {
  config_.faithful = faithful;
  reset();
}

void Mixer::build_binaural(int channels) {
  paths_.assign(static_cast<std::size_t>(channels), {});
  binaural_channels_ = channels;
  const Speaker* speakers = channels == 6 ? k51.data() : k71.data();
  for (int c = 0; c < channels; ++c) {
    const Speaker& speaker = speakers[c];
    for (int ear = 0; ear < 2; ++ear) {
      MixPath& path = paths_[static_cast<std::size_t>(c)][static_cast<std::size_t>(ear)];
      if (speaker.lfe) {
        path.gain = kLfeGain * 0.7071F;
        path.alpha = 1.0F;
        continue;
      }
      const float theta = speaker.azimuth_deg * kPi / 180.0F;
      const float side = std::sin(theta);  // > 0 : à droite
      const bool far = (ear == 0 && side > 0.0F) || (ear == 1 && side < 0.0F);
      const float shadow = far ? std::fabs(side) : 0.0F;
      // Woodworth : retard de l'oreille éloignée seulement.
      const float itd_s = far ? kHeadRadiusM / kSoundSpeed * (std::fabs(theta) <= kPi / 2 ? std::fabs(theta) + std::fabs(side)
                                                                                           : kPi - std::fabs(theta) + std::fabs(side))
                              : 0.0F;
      path.delay = std::clamp(static_cast<int>(std::lround(itd_s * static_cast<float>(config_.sample_rate))), 0,
                              kMixDelayMax - 1);
      path.gain = (far ? 1.0F - 0.35F * shadow : 1.0F) * 0.5F;
      path.alpha = far ? one_pole(6000.0F - 4500.0F * shadow, config_.sample_rate) : 1.0F;
    }
  }
}

PcmBlock Mixer::downmix(const PcmBlock& in) const {
  PcmBlock out{.pts_us = in.pts_us, .channels = 2};
  const auto frames = static_cast<std::size_t>(in.frames());
  const auto n = static_cast<std::size_t>(in.channels);
  out.samples.resize(frames * 2);
  const float centre = 0.7071F * kCenterLift;
  for (std::size_t f = 0; f < frames; ++f) {
    const float* x = &in.samples[f * n];
    float left = x[0] + centre * x[2] + kLfeGain * x[3];
    float right = x[1] + centre * x[2] + kLfeGain * x[3];
    if (n == 6) {  // FL FR FC LFE SL SR
      left += 0.7071F * x[4];
      right += 0.7071F * x[5];
    } else {  // FL FR FC LFE BL BR SL SR
      left += 0.5F * x[4] + 0.5F * x[6];
      right += 0.5F * x[5] + 0.5F * x[7];
    }
    out.samples[f * 2] = soft_limit(left * kDownmixScale);
    out.samples[f * 2 + 1] = soft_limit(right * kDownmixScale);
  }
  return out;
}

void Mixer::binaural(const PcmBlock& in, PcmBlock& out) {
  if (binaural_channels_ != in.channels) build_binaural(in.channels);
  const auto frames = static_cast<std::size_t>(in.frames());
  const auto n = static_cast<std::size_t>(in.channels);
  out.channels = 2;
  out.pts_us = in.pts_us;
  out.samples.assign(frames * 2, 0.0F);
  for (std::size_t f = 0; f < frames; ++f) {
    float left = 0.0F;
    float right = 0.0F;
    for (std::size_t c = 0; c < n; ++c) {
      const float sample = in.samples[f * n + c];
      left += run(paths_[c][0], sample);
      right += run(paths_[c][1], sample);
    }
    out.samples[f * 2] = soft_limit(left);
    out.samples[f * 2 + 1] = soft_limit(right);
  }
}

void Mixer::crossfeed(PcmBlock& block) {
  const auto frames = static_cast<std::size_t>(block.frames());
  constexpr float kNormalize = 1.0F / (1.0F + kCrossfeedGain);
  for (std::size_t f = 0; f < frames; ++f) {
    const float left = block.samples[f * 2];
    const float right = block.samples[f * 2 + 1];
    const float from_right = run(feed_[0], right);
    const float from_left = run(feed_[1], left);
    block.samples[f * 2] = (left + from_right) * kNormalize;
    block.samples[f * 2 + 1] = (right + from_left) * kNormalize;
  }
}

PcmBlock Mixer::process(PcmBlock block) {
  if (config_.faithful || config_.sink_channels != 2) return block;
  if (block.channels == 6 || block.channels == 8) {
    if (config_.listen == Listen::Cinema) return block;
    if (config_.listen == Listen::Headphones) {
      PcmBlock out;
      binaural(block, out);
      return out;
    }
    return downmix(block);
  }
  if (block.channels == 2 && config_.listen == Listen::Headphones) crossfeed(block);
  return block;  // mono ou autre : jamais spatialisé
}

}  // namespace ajp::audio
