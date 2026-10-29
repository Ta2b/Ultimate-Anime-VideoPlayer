#include "audio/mixer.h"

#include <cmath>
#include <cstdio>
#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace ajp::audio;
using Catch::Approx;

namespace {

PcmBlock impulse(int channels, int hot_channel, int frames = 256) {
  PcmBlock block{.pts_us = 0, .channels = channels};
  block.samples.assign(static_cast<std::size_t>(frames * channels), 0.0F);
  block.samples[static_cast<std::size_t>(hot_channel)] = 0.5F;
  return block;
}

}  // namespace

TEST_CASE("le décodeur rend les canaux de la source quand le mixeur les traite", "[mixer]") {
  CHECK(Mixer({.listen = Listen::Speakers}).decoder_channels(6) == 6);
  CHECK(Mixer({.listen = Listen::Headphones}).decoder_channels(8) == 8);
  CHECK(Mixer({.listen = Listen::Speakers}).decoder_channels(2) == 2);
  CHECK(Mixer({.listen = Listen::Speakers}).decoder_channels(1) == 2);  // l'appareil reçoit de la stéréo
  CHECK(Mixer({.listen = Listen::Cinema}).decoder_channels(6) == 2);    // libswresample adapte
  CHECK(Mixer({.listen = Listen::Speakers, .faithful = true}).decoder_channels(6) == 2);
  CHECK(Mixer({.listen = Listen::Cinema, .sink_channels = 6}).decoder_channels(6) == 6);
}

TEST_CASE("fidèle, mono et appareil multicanal ne sont jamais traités", "[mixer]") {
  Mixer faithful({.listen = Listen::Headphones, .faithful = true});
  const PcmBlock stereo = impulse(2, 0);
  CHECK(faithful.process(stereo).samples == stereo.samples);
  Mixer mono({.listen = Listen::Headphones});
  const PcmBlock one = impulse(1, 0);
  CHECK(mono.process(one).samples == one.samples);
  Mixer wide({.listen = Listen::Headphones, .sink_channels = 6});
  const PcmBlock six = impulse(6, 0);
  CHECK(wide.process(six).samples == six.samples);
  // Enceintes : la stéréo passe telle quelle.
  Mixer speakers({.listen = Listen::Speakers});
  CHECK(speakers.process(stereo).samples == stereo.samples);
}

TEST_CASE("le downmix soigné rehausse le centre et garde la balance", "[mixer]") {
  Mixer mixer({.listen = Listen::Speakers});
  const PcmBlock centre = mixer.process(impulse(6, 2));
  REQUIRE(centre.channels == 2);
  CHECK(centre.samples[0] == Approx(centre.samples[1]));
  CHECK(centre.samples[0] > 0.5F * 0.6F * 0.7071F);  // plus fort que le -3 dB habituel
  const PcmBlock left = mixer.process(impulse(8, 0));
  CHECK(left.samples[0] > 0.0F);
  CHECK(left.samples[1] == Approx(0.0F));
}

TEST_CASE("le binaural retarde et adoucit l'oreille éloignée", "[mixer]") {
  Mixer mixer({.listen = Listen::Headphones});
  const PcmBlock out = mixer.process(impulse(6, 0));  // haut-parleur avant gauche
  REQUIRE(out.channels == 2);
  const auto first_above = [&](std::size_t ear) {
    for (std::size_t f = 0; f < 200; ++f)
      if (std::fabs(out.samples[f * 2 + ear]) > 1e-3F) return static_cast<int>(f);
    return -1;
  };
  float peak_left = 0.0F;
  float peak_right = 0.0F;
  for (std::size_t f = 0; f < 200; ++f) {
    peak_left = std::max(peak_left, std::fabs(out.samples[f * 2]));
    peak_right = std::max(peak_right, std::fabs(out.samples[f * 2 + 1]));
  }
  CHECK(first_above(0) >= 0);
  CHECK(first_above(1) > first_above(0));  // l'oreille droite entend plus tard
  CHECK(peak_right < peak_left);           // et moins fort
}

TEST_CASE("le crossfeed amène un peu du canal opposé, retardé", "[mixer]") {
  Mixer mixer({.listen = Listen::Headphones});
  const PcmBlock out = mixer.process(impulse(2, 0));  // impulsion à gauche seulement
  CHECK(out.samples[1] == Approx(0.0F));              // pas encore : le retard
  float later = 0.0F;
  for (std::size_t f = 1; f < 100; ++f) later = std::max(later, std::fabs(out.samples[f * 2 + 1]));
  CHECK(later > 0.01F);
  CHECK(later < 0.5F);
  // Après reset, l'état est oublié : le même bloc donne la même sortie.
  mixer.reset();
  CHECK(mixer.process(impulse(2, 0)).samples == out.samples);
}

TEST_CASE("le limiteur garde la sortie sous 1", "[mixer]") {
  Mixer mixer({.listen = Listen::Speakers});
  PcmBlock loud{.pts_us = 0, .channels = 6};
  loud.samples.assign(6 * 64, 1.0F);
  const PcmBlock out = mixer.process(loud);
  for (const float sample : out.samples) CHECK(std::fabs(sample) < 1.0F);
}

TEST_CASE("les profils : Naturel est le mixage d'origine, les autres le changent", "[mixer][profils]") {
  CHECK(ajp::audio::MixTuning{} == ajp::audio::find_preset("natural")->tuning);
  CHECK(ajp::audio::find_preset("inconnu") == nullptr);
  // Au casque, un crossfeed fort amène plus du canal opposé qu'un doux.
  const auto leak = [](const char* key) {
    Mixer mixer(MixConfig{.listen = Listen::Headphones, .tuning = ajp::audio::find_preset(key)->tuning});
    PcmBlock block{.pts_us = 0, .channels = 2};
    block.samples.assign(512, 0.0F);
    for (std::size_t f = 0; f < 256; ++f) block.samples[f * 2] = 0.5F;  // gauche seule
    const PcmBlock out = mixer.process(std::move(block));
    return out.samples[200 * 2 + 1];
  };
  CHECK(leak("strong") > leak("natural"));
  CHECK(leak("natural") > leak("soft"));
  // Aux enceintes, « Voix claires » baisse l'arrière par rapport à Naturel.
  const auto rear = [](const char* key) {
    Mixer mixer(MixConfig{.listen = Listen::Speakers, .tuning = ajp::audio::find_preset(key)->tuning});
    PcmBlock block{.pts_us = 0, .channels = 6};
    block.samples.assign(6 * 64, 0.0F);
    for (std::size_t f = 0; f < 64; ++f) block.samples[f * 6 + 4] = 0.5F;  // arrière gauche
    return mixer.process(std::move(block)).samples[40 * 2];
  };
  CHECK(rear("voices") < rear("natural"));
}

TEST_CASE("les réglages hors plage sont ramenés, le tableau fait l'aller-retour", "[mixer][profils]") {
  ajp::audio::MixTuning wild;
  wild.crossfeed_gain = 9.0F;
  wild.lfe_gain = -1.0F;
  const ajp::audio::MixTuning tame = ajp::audio::clamped(wild);
  CHECK(tame.crossfeed_gain == Approx(0.7F));
  CHECK(tame.lfe_gain == Approx(0.0F));
  const auto round_trip =
      ajp::audio::tuning_from(ajp::audio::to_array(ajp::audio::find_preset("wide")->tuning));
  CHECK(round_trip == ajp::audio::find_preset("wide")->tuning);
}

namespace {

PcmBlock stereo_sine(float hz, float amplitude, int frames = 48'000) {
  PcmBlock block{.pts_us = 0, .channels = 2};
  block.samples.resize(static_cast<std::size_t>(frames) * 2);
  for (int f = 0; f < frames; ++f) {
    const float v = amplitude * std::sin(6.2831853F * hz * static_cast<float>(f) / 48'000.0F);
    block.samples[static_cast<std::size_t>(f) * 2] = v;
    block.samples[static_cast<std::size_t>(f) * 2 + 1] = v;
  }
  return block;
}

// L'énergie efficace du dernier quart du bloc (le régime établi).
float steady_rms(const PcmBlock& block) {
  const std::size_t begin = block.samples.size() * 3 / 4;
  double sum = 0.0;
  for (std::size_t i = begin; i < block.samples.size(); ++i)
    sum += static_cast<double>(block.samples[i]) * block.samples[i];
  return static_cast<float>(std::sqrt(sum / static_cast<double>(block.samples.size() - begin)));
}

float through(const ajp::audio::MixTuning& tuning, float hz, float amplitude) {
  Mixer mixer(MixConfig{.listen = Listen::Speakers, .tuning = tuning});
  return steady_rms(mixer.process(stereo_sine(hz, amplitude)));
}

}  // namespace

TEST_CASE("l'égaliseur relève les graves, les voix et les aigus à leur fréquence", "[mixer][profils][eq]") {
  const float flat = through({}, 60.0F, 0.2F);
  ajp::audio::MixTuning bass;
  bass.eq_bass_db = 6.0F;
  CHECK(through(bass, 60.0F, 0.2F) / flat == Approx(2.0F).margin(0.25F));  // +6 dB ~ x2
  ajp::audio::MixTuning voice;
  voice.eq_voice_db = 6.0F;
  CHECK(through(voice, 2500.0F, 0.2F) / through({}, 2500.0F, 0.2F) == Approx(2.0F).margin(0.25F));
  CHECK(through(voice, 100.0F, 0.2F) / through({}, 100.0F, 0.2F) ==
        Approx(1.0F).margin(0.1F));  // hors bande : intact
  ajp::audio::MixTuning treble;
  treble.eq_treble_db = -6.0F;
  CHECK(through(treble, 12'000.0F, 0.2F) / through({}, 12'000.0F, 0.2F) == Approx(0.5F).margin(0.12F));
}

TEST_CASE("la compression nuit rapproche les fortes et les faibles dynamiques", "[mixer][profils][nuit]") {
  ajp::audio::MixTuning night;
  night.night = 1.0F;
  const float loud_in = 0.8F;
  const float quiet_in = 0.02F;
  const float ratio_in = loud_in / quiet_in;
  const float ratio_out = through(night, 1000.0F, loud_in) / through(night, 1000.0F, quiet_in);
  CHECK(ratio_out < ratio_in * 0.5F);  // la dynamique est réduite de moitié au moins
  CHECK(through(night, 1000.0F, quiet_in) > through({}, 1000.0F, quiet_in));  // le murmure est relevé
  CHECK(through(night, 1000.0F, loud_in) < 1.0F);                             // jamais d'écrêtage
}

TEST_CASE("la pièce ajoute une queue de réverbération, sans pièce rien ne dure", "[mixer][profils][room]") {
  const auto tail_energy = [](const ajp::audio::MixTuning& tuning) {
    Mixer mixer(MixConfig{.listen = Listen::Speakers, .tuning = tuning});
    PcmBlock block{.pts_us = 0, .channels = 2};
    block.samples.assign(2 * 24'000, 0.0F);
    block.samples[0] = 0.9F;
    block.samples[1] = 0.9F;
    const PcmBlock out = mixer.process(std::move(block));
    double sum = 0.0;
    for (std::size_t i = 2 * 4800; i < out.samples.size(); ++i)
      sum += static_cast<double>(out.samples[i]) * out.samples[i];  // après 100 ms
    return sum;
  };
  ajp::audio::MixTuning room;
  room.room = 1.0F;
  CHECK(tail_energy(room) > 1e-6);
  CHECK(tail_energy({}) == 0.0);
}

TEST_CASE("les nouveaux profils existent et le tableau des réglages fait l'aller-retour",
          "[mixer][profils]") {
  for (const char* key : {"room", "anime", "night"}) CHECK(ajp::audio::find_preset(key) != nullptr);
  CHECK(ajp::audio::find_preset("anime")->tuning.eq_voice_db > 0.0F);
  const auto tuning = ajp::audio::find_preset("room")->tuning;
  const auto back = ajp::audio::to_array(ajp::audio::tuning_from_wire(ajp::audio::to_wire(tuning)));
  const auto original = ajp::audio::to_array(tuning);
  for (std::size_t i = 0; i < back.size(); ++i) {
    INFO("champ " << i);
    CHECK(back[i] == original[i]);
  }
}

TEST_CASE("aux enceintes, une piste stéréo change avec Voix claires, Nuit et Voix d'anime",
          "[mixer][profils]") {
  // Un son de voix (2,5 kHz, le centre de la cloche des voix) et un grave (80 Hz) en stéréo : chaque profil
  // des enceintes doit en changer le niveau par rapport à Naturel, même sans surround.
  const auto level = [](const char* key, float hz) {
    Mixer mixer(MixConfig{.listen = Listen::Speakers, .tuning = ajp::audio::find_preset(key)->tuning});
    PcmBlock block{.pts_us = 0, .channels = 2};
    constexpr std::size_t kFrames = 9600;
    block.samples.assign(kFrames * 2, 0.0F);
    for (std::size_t f = 0; f < kFrames; ++f) {
      const float v = 0.1F * std::sin(6.2831853F * hz * static_cast<float>(f) / 48000.0F);
      block.samples[f * 2] = v;
      block.samples[f * 2 + 1] = v;
    }
    const PcmBlock out = mixer.process(std::move(block));
    float peak = 0.0F;
    for (std::size_t f = kFrames / 2; f < kFrames; ++f) peak = std::max(peak, std::abs(out.samples[f * 2]));
    return peak;
  };
  for (const char* key : {"voices", "night", "anime"}) {
    INFO(key);
    CHECK(level(key, 2500.0F) != Catch::Approx(level("natural", 2500.0F)).epsilon(0.03));
  }
  CHECK(level("voices", 80.0F) < level("natural", 80.0F));
}

TEST_CASE("tableau : gain en dB de chaque profil sur une piste stéréo", "[.profils_tableau]") {
  // Un sinus stéréo à 80 Hz, 2,5 kHz et 8 kHz (niveau 0,1), puis une salve forte suivie d'un murmure (nuit).
  const auto run = [](const char* key, Listen listen, float hz, float amp) {
    Mixer mixer(MixConfig{.listen = listen, .tuning = ajp::audio::find_preset(key)->tuning});
    PcmBlock block{.pts_us = 0, .channels = 2};
    constexpr std::size_t kFrames = 24000;
    block.samples.assign(kFrames * 2, 0.0F);
    for (std::size_t f = 0; f < kFrames; ++f) {
      const float v = amp * std::sin(6.2831853F * hz * static_cast<float>(f) / 48000.0F);
      block.samples[f * 2] = v;
      block.samples[f * 2 + 1] = 0.8F * v;
    }
    const PcmBlock out = mixer.process(std::move(block));
    double sum = 0.0;
    for (std::size_t f = kFrames / 2; f < kFrames; ++f)
      sum += static_cast<double>(out.samples[f * 2]) * static_cast<double>(out.samples[f * 2]);
    return std::sqrt(sum / static_cast<double>(kFrames / 2));
  };
  for (const Listen listen : {Listen::Speakers, Listen::Headphones}) {
    std::printf("\n%s\n%-8s %8s %8s %8s %12s\n", listen == Listen::Speakers ? "ENCEINTES" : "CASQUE",
                "profil", "80 Hz", "2,5 kHz", "8 kHz", "murmure 0,01");
    for (const auto& preset : ajp::audio::mix_presets()) {
      const std::string key(preset.key);
      if ((listen == Listen::Speakers && !preset.speakers) ||
          (listen == Listen::Headphones && !preset.headphones))
        continue;
      const auto db = [&](float hz, float amp) {
        return 20.0 * std::log10(run(key.c_str(), listen, hz, amp) / run("natural", listen, hz, amp));
      };
      std::printf("%-8s %+8.2f %+8.2f %+8.2f %+12.2f\n", key.c_str(), db(80.0F, 0.1F), db(2500.0F, 0.1F),
                  db(8000.0F, 0.1F), db(1000.0F, 0.01F));
    }
  }
}
