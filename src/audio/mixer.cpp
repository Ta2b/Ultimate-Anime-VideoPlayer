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

struct Biquad {
  float b0 = 1.0F, b1 = 0.0F, b2 = 0.0F, a1 = 0.0F, a2 = 0.0F;
  std::array<float, 2> z1{}, z2{};  // un état par canal
  float run(int channel, float x) {
    const auto c = static_cast<std::size_t>(channel);
    const float y = b0 * x + z1[c];
    z1[c] = b1 * x - a1 * y + z2[c];
    z2[c] = b2 * x - a2 * y;
    return y;
  }
};

// Les formules de l'« Audio EQ Cookbook » de Robert Bristow-Johnson : étagère grave, étagère aiguë et cloche.
Biquad shelf(bool high, float freq, float gain_db, int rate) {
  const float a = std::pow(10.0F, gain_db / 40.0F);
  const float w = 2.0F * kPi * freq / static_cast<float>(rate);
  const float cw = std::cos(w);
  const float alpha = std::sin(w) / 2.0F * std::sqrt(2.0F);  // pente 1
  const float sq = 2.0F * std::sqrt(a) * alpha;
  float b0, b1, b2, a0, a1, a2;
  if (!high) {
    b0 = a * ((a + 1) - (a - 1) * cw + sq);
    b1 = 2 * a * ((a - 1) - (a + 1) * cw);
    b2 = a * ((a + 1) - (a - 1) * cw - sq);
    a0 = (a + 1) + (a - 1) * cw + sq;
    a1 = -2 * ((a - 1) + (a + 1) * cw);
    a2 = (a + 1) + (a - 1) * cw - sq;
  } else {
    b0 = a * ((a + 1) + (a - 1) * cw + sq);
    b1 = -2 * a * ((a - 1) + (a + 1) * cw);
    b2 = a * ((a + 1) + (a - 1) * cw - sq);
    a0 = (a + 1) - (a - 1) * cw + sq;
    a1 = 2 * ((a - 1) - (a + 1) * cw);
    a2 = (a + 1) - (a - 1) * cw - sq;
  }
  return Biquad{b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0, {}, {}};
}

Biquad peaking(float freq, float gain_db, float q, int rate) {
  const float a = std::pow(10.0F, gain_db / 40.0F);
  const float w = 2.0F * kPi * freq / static_cast<float>(rate);
  const float alpha = std::sin(w) / (2.0F * q);
  const float cw = std::cos(w);
  const float a0 = 1 + alpha / a;
  return Biquad{
      (1 + alpha * a) / a0, -2 * cw / a0, (1 - alpha * a) / a0, -2 * cw / a0, (1 - alpha / a) / a0, {}, {}};
}

}  // namespace

// Ce qui suit le mixage, en stéréo : égaliseur, pièce légère, compression « nuit ». Rien n'existe (ni coût)
// tant que le profil n'en demande pas.
struct MixPost {
  struct Comb {
    std::vector<float> line;
    std::size_t at = 0;
    float low = 0.0F;
  };
  struct Allpass {
    std::vector<float> line;
    std::size_t at = 0;
  };
  explicit MixPost(const MixTuning& t, int sample_rate)
      : tune(t),
        rate(sample_rate),
        eq_on(t.eq_bass_db != 0.0F || t.eq_voice_db != 0.0F || t.eq_treble_db != 0.0F),
        wet(std::clamp(t.room * 0.4F + t.distance * 0.2F, 0.0F, 0.5F)),
        night_on(t.night > 0.0F) {
    if (eq_on) {
      bass = shelf(false, 120.0F, t.eq_bass_db, rate);
      voice = peaking(2500.0F, t.eq_voice_db, 0.9F, rate);
      treble = shelf(true, 8000.0F, t.eq_treble_db, rate);
    }
    if (wet > 0.0F) {
      constexpr std::array<float, 4> kMs{29.7F, 37.1F, 41.1F, 43.7F};
      for (int ear = 0; ear < 2; ++ear) {
        for (std::size_t i = 0; i < kMs.size(); ++i) {
          const float ms = kMs[i] + 1.2F * static_cast<float>(ear);  // les deux oreilles décorrélées
          combs[static_cast<std::size_t>(ear)][i].line.assign(
              static_cast<std::size_t>(ms * 0.001F * static_cast<float>(rate)) + 1, 0.0F);
        }
        for (std::size_t i = 0; i < 2; ++i)
          allpass[static_cast<std::size_t>(ear)][i].line.assign(
              static_cast<std::size_t>((i == 0 ? 5.0F : 1.7F) * 0.001F * static_cast<float>(rate)) + 1, 0.0F);
      }
    }
    release = std::exp(-1.0F / (0.25F * static_cast<float>(rate)));
    attack = std::exp(-1.0F / (0.01F * static_cast<float>(rate)));
  }
  bool needed() const { return eq_on || wet > 0.0F || night_on; }

  void process(PcmBlock& block) {
    const auto frames = static_cast<std::size_t>(block.frames());
    for (std::size_t f = 0; f < frames; ++f) {
      float l = block.samples[f * 2];
      float r = block.samples[f * 2 + 1];
      if (eq_on) {
        l = treble.run(0, voice.run(0, bass.run(0, l)));
        r = treble.run(1, voice.run(1, bass.run(1, r)));
      }
      if (wet > 0.0F) {
        const float in = 0.5F * (l + r) * 0.25F;
        std::array<float, 2> tail{};
        for (std::size_t ear = 0; ear < 2; ++ear) {
          float sum = 0.0F;
          for (Comb& comb : combs[ear]) {
            const float out = comb.line[comb.at];
            comb.low += 0.35F * (out - comb.low);  // amorti : les aigus meurent plus vite
            comb.line[comb.at] = in + 0.74F * comb.low;
            comb.at = (comb.at + 1) % comb.line.size();
            sum += out;
          }
          for (Allpass& ap : allpass[ear]) {
            const float buffered = ap.line[ap.at];
            const float x = sum;
            sum = -x + buffered;
            ap.line[ap.at] = x + 0.5F * buffered;
            ap.at = (ap.at + 1) % ap.line.size();
          }
          tail[ear] = sum;
        }
        l += wet * tail[0];
        r += wet * tail[1];
      }
      if (night_on) {
        // Compression liée des deux canaux : seuil -26 dBFS, rapport jusqu'à 6:1, gain de rattrapage à
        // moitié.
        const float level = std::max(std::fabs(l), std::fabs(r));
        env = level > env ? attack * env + (1.0F - attack) * level : release * env + (1.0F - release) * level;
        const float ratio = 1.0F + 5.0F * tune.night;
        const float threshold_db = -26.0F;
        const float env_db = 20.0F * std::log10(std::max(env, 1e-6F));
        const float over = env_db - threshold_db;
        const float reduction_db = over > 0.0F ? over * (1.0F - 1.0F / ratio) : 0.0F;
        const float makeup_db = -threshold_db * (1.0F - 1.0F / ratio) * 0.5F;
        const float gain = std::pow(10.0F, (makeup_db - reduction_db) / 20.0F);
        l *= gain;
        r *= gain;
      }
      block.samples[f * 2] = soft_limit(l);
      block.samples[f * 2 + 1] = soft_limit(r);
    }
  }
  void reset() {
    bass.z1 = bass.z2 = voice.z1 = voice.z2 = treble.z1 = treble.z2 = {};
    for (auto& bank : combs)
      for (Comb& c : bank) std::ranges::fill(c.line, 0.0F);
    for (auto& bank : allpass)
      for (Allpass& a : bank) std::ranges::fill(a.line, 0.0F);
    env = 0.0F;
  }

  MixTuning tune;
  int rate;
  bool eq_on;
  float wet;
  bool night_on;
  Biquad bass, voice, treble;
  std::array<std::array<Comb, 4>, 2> combs;
  std::array<std::array<Allpass, 2>, 2> allpass;
  float env = 0.0F;
  float attack = 0.0F;
  float release = 0.0F;
};

namespace {

const std::array<MixPreset, 8> kPresets{{
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
    {"night", "Nuit",
     "Enceintes : voix devant, arrière et basses très réduits, dynamique comprimée (explosions réduites, "
     "murmures relevés).",
     false, true,
     MixTuning{.surround_gain = 0.7F,
               .center_lift = 1.5F,
               .lfe_gain = 0.12F,
               .downmix_scale = 0.55F,
               .night = 0.7F}},
    {"room", "Salon", "Casque : enceintes dans une pièce, un peu plus loin, avec une réverbération légère.",
     true, false,
     MixTuning{.crossfeed_gain = 0.4F,
               .crossfeed_cutoff_hz = 700.0F,
               .crossfeed_delay_ms = 0.3F,
               .head_shadow = 0.4F,
               .distance = 0.5F,
               .room = 0.35F}},
    {"anime", "Voix d'anime",
     "Voix plus nettes et aigus plus clairs, graves allégés : dialogues d'anime sans fatigue.", true, true,
     MixTuning{.eq_bass_db = -1.5F, .eq_voice_db = 3.0F, .eq_treble_db = 1.5F}},
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
                           .downmix_scale = f[8],
                           .distance = f[9],
                           .room = f[10],
                           .eq_bass_db = f[11],
                           .eq_voice_db = f[12],
                           .eq_treble_db = f[13],
                           .night = f[14]});
}

std::array<float, 16> to_wire(const MixTuning& t) {
  std::array<float, 16> wire{};
  const auto fields = to_array(t);
  std::ranges::copy(fields, wire.begin());
  return wire;
}

MixTuning tuning_from_wire(const std::array<float, 16>& wire) {
  std::array<float, kMixTuningFields> fields{};
  std::copy_n(wire.begin(), fields.size(), fields.begin());
  return tuning_from(fields);
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
  t.distance = std::clamp(t.distance, 0.0F, 1.0F);
  t.room = std::clamp(t.room, 0.0F, 1.0F);
  t.eq_bass_db = std::clamp(t.eq_bass_db, -8.0F, 8.0F);
  t.eq_voice_db = std::clamp(t.eq_voice_db, -8.0F, 8.0F);
  t.eq_treble_db = std::clamp(t.eq_treble_db, -8.0F, 8.0F);
  t.night = std::clamp(t.night, 0.0F, 1.0F);
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
  if (config_.sink_channels == 2) {
    auto post = std::make_shared<MixPost>(tune, config_.sample_rate);
    if (post->needed()) post_ = std::move(post);
  }
  if (config_.hrir && !config_.hrir->empty() && config_.sink_channels == 2)
    hrir_ = std::make_shared<const Hrir>(prepared(*config_.hrir, config_.sample_rate));
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
  if (post_) post_->reset();
  if (convolver_) convolver_->reset();
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
      path.gain =
          (far ? 1.0F - tune.head_shadow * shadow : 1.0F) * 0.5F * level * (1.0F - 0.2F * tune.distance);
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

PcmBlock Mixer::convolve(const PcmBlock& in) {
  if (!convolver_ || convolver_->source_channels() != in.channels)
    convolver_ = std::make_shared<HrirConvolver>(*hrir_, in.channels, config_.tuning.lfe_gain);
  PcmBlock out{.pts_us = in.pts_us, .channels = 2};
  convolver_->process(in.samples, out.samples);
  return out;
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
  PcmBlock out = process_core(std::move(block));
  if (post_ && out.channels == 2 && config_.listen != Listen::Cinema) post_->process(out);
  return out;
}

PcmBlock Mixer::process_core(PcmBlock block) {
  const bool convolved = hrir_ && config_.listen == Listen::Headphones &&
                         (block.channels == 2 || block.channels == 6 || block.channels == 8);
  if (convolved) return convolve(block);
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
