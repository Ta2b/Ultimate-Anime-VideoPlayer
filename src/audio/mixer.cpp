#include "audio/mixer.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace ajp::audio {
namespace {

constexpr float kPi = std::numbers::pi_v<float>;
constexpr float kSoundSpeed = 343.0F;

// Les haut-parleurs dans l'ordre de SDL3 : azimut en degrés (négatif à gauche) ; le LFE n'a pas de place.
struct Speaker {
  float azimuth_deg;
  bool lfe;
};
constexpr std::array<Speaker, 6> k51{
    {{-30, false}, {30, false}, {0, false}, {0, true}, {-110, false}, {110, false}}};
constexpr std::array<Speaker, 8> k71{{{-30, false},
                                      {30, false},
                                      {0, false},
                                      {0, true},
                                      {-150, false},
                                      {150, false},
                                      {-90, false},
                                      {90, false}}};

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

const std::array<MixPreset, 6> kPresets{{
    {"natural", "Naturel", "Le réglage d'origine, équilibré.", true, true, MixTuning{}},
    {"soft", "Doux", "Casque : séparation presque intacte, juste de quoi adoucir les écoutes longues.", true,
     false,
     MixTuning{.crossfeed_gain = 0.2F,
               .crossfeed_cutoff_hz = 800.0F,
               .crossfeed_delay_ms = 0.2F,
               .head_shadow = 0.25F}},
    {"strong", "Enceintes virtuelles", "Casque : comme des enceintes devant soi, le plus loin de la tête.",
     true, false,
     MixTuning{.crossfeed_gain = 0.5F,
               .crossfeed_cutoff_hz = 650.0F,
               .crossfeed_delay_ms = 0.3F,
               .head_shadow = 0.45F,
               .surround_gain = 1.1F}},
    {"wide", "Large", "Casque : tête plus grande, arrière plus présent (5.1 et 7.1).", true, false,
     MixTuning{.crossfeed_gain = 0.28F, .head_radius_m = 0.10F, .head_shadow = 0.5F, .surround_gain = 1.2F}},
    {"voices", "Voix claires", "Enceintes : centre rehaussé, arrière et basses discrets.", false, true,
     MixTuning{.surround_gain = 0.85F, .center_lift = 1.6F, .lfe_gain = 0.2F}},
    {"night", "Nuit", "Enceintes : voix devant, arrière et basses très réduits.", false, true,
     MixTuning{.surround_gain = 0.7F, .center_lift = 1.5F, .lfe_gain = 0.12F, .downmix_scale = 0.55F}},
}};

}  // namespace

std::array<float, kMixTuningFields> to_array(const MixTuning& t) {
  return {t.crossfeed_gain, t.crossfeed_cutoff_hz, t.crossfeed_delay_ms, t.head_radius_m, t.head_shadow,
          t.surround_gain,  t.center_lift,         t.lfe_gain,           t.downmix_scale};
}

MixTuning tuning_from(const std::array<float, kMixTuningFields>& f) {
  return clamped(MixTuning{.crossfeed_gain = f[0],
                           .crossfeed_cutoff_hz = f[1],
                           .crossfeed_delay_ms = f[2],
                           .head_radius_m = f[3],
                           .head_shadow = f[4],
                           .surround_gain = f[5],
                           .center_lift = f[6],
                           .lfe_gain = f[7],
                           .downmix_scale = f[8]});
}

MixTuning clamped(MixTuning t) {
  t.crossfeed_gain = std::clamp(t.crossfeed_gain, 0.0F, 0.7F);
  t.crossfeed_cutoff_hz = std::clamp(t.crossfeed_cutoff_hz, 300.0F, 1500.0F);
  t.crossfeed_delay_ms = std::clamp(t.crossfeed_delay_ms, 0.0F, 0.6F);
  t.head_radius_m = std::clamp(t.head_radius_m, 0.07F, 0.11F);
  t.head_shadow = std::clamp(t.head_shadow, 0.0F, 0.7F);
  t.surround_gain = std::clamp(t.surround_gain, 0.5F, 1.6F);
  t.center_lift = std::clamp(t.center_lift, 1.0F, 2.0F);
  t.lfe_gain = std::clamp(t.lfe_gain, 0.0F, 0.6F);
  t.downmix_scale = std::clamp(t.downmix_scale, 0.4F, 0.9F);
  return t;
}

std::span<const MixPreset> mix_presets() {
  return kPresets;
}

const MixPreset* find_preset(std::string_view key) {
  const auto found = std::ranges::find(kPresets, key, &MixPreset::key);
  return found == kPresets.end() ? nullptr : &*found;
}

Mixer::Mixer(MixConfig config) : config_(config) {
  config_.tuning = clamped(config_.tuning);
  const MixTuning& tune = config_.tuning;
  const int delay = std::clamp(static_cast<int>(std::lround(tune.crossfeed_delay_ms * 0.001F *
                                                            static_cast<float>(config_.sample_rate))),
                               0, kMixDelayMax - 1);
  for (MixPath& path : feed_) {
    path.delay = delay;
    path.gain = tune.crossfeed_gain;
    path.alpha = one_pole(tune.crossfeed_cutoff_hz, config_.sample_rate);
  }
}

int Mixer::decoder_channels(int source_channels) const {
  const bool processed = !config_.faithful && config_.listen != Listen::Cinema &&
                         config_.sink_channels == 2 && (source_channels == 6 || source_channels == 8);
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
  const MixTuning& tune = config_.tuning;
  const Speaker* speakers = channels == 6 ? k51.data() : k71.data();
  for (int c = 0; c < channels; ++c) {
    const Speaker& speaker = speakers[c];
    for (int ear = 0; ear < 2; ++ear) {
      MixPath& path = paths_[static_cast<std::size_t>(c)][static_cast<std::size_t>(ear)];
      if (speaker.lfe) {
        path.gain = tune.lfe_gain * 0.7071F;
        path.alpha = 1.0F;
        continue;
      }
      const float theta = speaker.azimuth_deg * kPi / 180.0F;
      const float side = std::sin(theta);  // > 0 : à droite
      const bool far = (ear == 0 && side > 0.0F) || (ear == 1 && side < 0.0F);
      const float shadow = far ? std::fabs(side) : 0.0F;
      // Woodworth : retard de l'oreille éloignée seulement.
      const float itd_s = far ? tune.head_radius_m / kSoundSpeed *
                                    (std::fabs(theta) <= kPi / 2 ? std::fabs(theta) + std::fabs(side)
                                                                 : kPi - std::fabs(theta) + std::fabs(side))
                              : 0.0F;
      path.delay = std::clamp(static_cast<int>(std::lround(itd_s * static_cast<float>(config_.sample_rate))),
                              0, kMixDelayMax - 1);
      // Les arrières et les côtés (|azimut| > 60°) prennent le niveau « surround » ; l'avant reste à 1.
      const float level = std::fabs(speaker.azimuth_deg) > 60.0F ? tune.surround_gain : 1.0F;
      path.gain = (far ? 1.0F - tune.head_shadow * shadow : 1.0F) * 0.5F * level;
      path.alpha = far ? one_pole(6000.0F - 4500.0F * shadow, config_.sample_rate) : 1.0F;
    }
  }
}

PcmBlock Mixer::downmix(const PcmBlock& in) const {
  PcmBlock out{.pts_us = in.pts_us, .channels = 2};
  const auto frames = static_cast<std::size_t>(in.frames());
  const auto n = static_cast<std::size_t>(in.channels);
  out.samples.resize(frames * 2);
  const MixTuning& tune = config_.tuning;
  const float centre = 0.7071F * tune.center_lift;
  const float side = tune.surround_gain;
  for (std::size_t f = 0; f < frames; ++f) {
    const float* x = &in.samples[f * n];
    float left = x[0] + centre * x[2] + tune.lfe_gain * x[3];
    float right = x[1] + centre * x[2] + tune.lfe_gain * x[3];
    if (n == 6) {  // FL FR FC LFE SL SR
      left += side * 0.7071F * x[4];
      right += side * 0.7071F * x[5];
    } else {  // FL FR FC LFE BL BR SL SR
      left += side * (0.5F * x[4] + 0.5F * x[6]);
      right += side * (0.5F * x[5] + 0.5F * x[7]);
    }
    out.samples[f * 2] = soft_limit(left * tune.downmix_scale);
    out.samples[f * 2 + 1] = soft_limit(right * tune.downmix_scale);
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
  const float kNormalize = 1.0F / (1.0F + config_.tuning.crossfeed_gain);
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
