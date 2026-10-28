#include "audio/mixer.h"

#include <cmath>

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
