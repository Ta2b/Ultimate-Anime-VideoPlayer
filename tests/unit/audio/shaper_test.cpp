// Le dernier étage avant le matériel : rampes, fondu, limiteur doux (B16-03, B16-07, B16-08).
#include "audio/shaper.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::audio;

namespace {

std::vector<float> sine(int frames, int channels, float amplitude) {
  std::vector<float> out(static_cast<std::size_t>(frames * channels));
  for (int frame = 0; frame < frames; ++frame) {
    const float value = amplitude * std::sin(static_cast<float>(frame) * 0.05F);
    for (int channel = 0; channel < channels; ++channel)
      out[static_cast<std::size_t>((frame * channels) + channel)] = value;
  }
  return out;
}

float peak(const std::vector<float>& samples) {
  float result = 0.0F;
  for (const float sample : samples) result = std::max(result, std::abs(sample));
  return result;
}

}  // namespace

TEST_CASE("OutputShaper : gain 1 établi, le tampon passe tel quel", "[audio][shaper]") {
  OutputShaper shaper;
  std::vector<float> samples = sine(1'024, 2, 0.95F);
  const std::vector<float> before = samples;
  shaper.process(samples, {});
  CHECK(samples == before);
}

TEST_CASE("OutputShaper : au-dessus de 100 %, les crêtes sont arrondies, jamais écrêtées",
          "[audio][shaper]") {
  OutputShaper shaper;
  std::vector<float> samples = sine(4'800, 2, 0.9F);  // crêtes à 0,9
  shaper.process(samples, {.gain = 1.5F});
  CHECK(peak(samples) < 1.0F);
  CHECK(peak(samples) > 0.9F);
  // Pas de plateau : l'écrêtage dur ferait rester des dizaines d'échantillons à exactement 1.
  const auto flat =
      std::count_if(samples.begin(), samples.end(), [](float s) { return std::abs(s) >= 0.9999F; });
  CHECK(flat == 0);
  // Le gain suivant garde le limiteur : la montée n'a pas fini d'être appliquée sans lui.
  std::vector<float> next = sine(4'800, 2, 0.9F);
  shaper.process(next, {.gain = 1.5F});
  CHECK(peak(next) < 1.0F);
}

TEST_CASE("OutputShaper : un gain qui change monte par une rampe, pas par une marche", "[audio][shaper]") {
  OutputShaper shaper;
  std::vector<float> samples(4'800 * 2, 0.5F);
  shaper.process(samples, {.gain = 0.0F});  // coupure du son
  // 10 ms à 48 kHz : 480 trames de rampe, puis le silence.
  CHECK(samples[0] > 0.49F * (1.0F - 1.0F / 480.0F));  // la première trame est presque intacte
  CHECK(samples[2 * 240] < 0.3F);
  CHECK(samples[2 * 240] > 0.2F);
  CHECK(samples[2 * 480] == 0.0F);
  CHECK(samples[2 * 4'000] == 0.0F);
  for (std::size_t i = 2; i < 2 * 480; i += 2) CHECK(samples[i] <= samples[i - 2]);  // décroissante
  CHECK(shaper.applied_gain() == 0.0F);
}

TEST_CASE("OutputShaper : un son qui repart entre en fondu", "[audio][shaper]") {
  OutputShaper shaper;
  std::vector<float> samples(4'800 * 2, 0.5F);
  shaper.process(samples, {.fade_in = true});
  CHECK(samples[0] < 0.01F);
  CHECK(samples[2 * 240] > 0.2F);
  CHECK(samples[2 * 240] < 0.3F);
  CHECK(samples[2 * 480] == 0.5F);
  CHECK(samples[2 * 4'000] == 0.5F);
}

TEST_CASE("OutputShaper : un mixage vers un autre nombre de canaux se limite même à gain 1",
          "[audio][shaper]") {
  OutputShaper shaper;
  std::vector<float> samples = sine(4'800, 2, 1.4F);  // crêtes d'un 5.1 vers stéréo, au-delà de 1
  shaper.process(samples, {.force_limit = true});
  CHECK(peak(samples) < 1.0F);
}

TEST_CASE("OutputShaper : une demande vide ou incohérente ne touche à rien", "[audio][shaper]") {
  OutputShaper shaper;
  std::vector<float> samples = sine(100, 2, 1.0F);
  const std::vector<float> before = samples;
  shaper.process(samples, {.channels = 0});
  shaper.process(samples, {.sample_rate = 0});
  CHECK(samples == before);
}

// C16-03 : un tampon plus court que la rampe (5 ms à 48 kHz) ne la raccourcit pas, la rampe continue au
// suivant.
TEST_CASE("OutputShaper : la rampe de gain continue d'un tampon court au suivant", "[audio][shaper]") {
  OutputShaper shaper;
  constexpr std::size_t kFrames = 240;  // 5 ms
  float previous = 1.0F;
  float worst = 0.0F;
  for (int buffer = 0; buffer < 4; ++buffer) {
    std::vector<float> samples(kFrames, 1.0F);
    shaper.process(samples, {.channels = 1, .gain = 0.5F});
    for (const float value : samples) {
      worst = std::max(worst, std::abs(value - previous));
      previous = value;
    }
  }
  CHECK(worst < 0.005F);
  CHECK(previous == 0.5F);
  CHECK(shaper.applied_gain() == 0.5F);
}

TEST_CASE("OutputShaper : le fondu d'entrée continue d'un tampon court au suivant", "[audio][shaper]") {
  OutputShaper shaper;
  std::vector<float> first(240, 1.0F);
  std::vector<float> second(240, 1.0F);
  std::vector<float> third(240, 1.0F);
  shaper.process(first, {.channels = 1, .fade_in = true});
  shaper.process(second, {.channels = 1});
  shaper.process(third, {.channels = 1});
  CHECK(first.back() < 0.51F);
  CHECK(second.front() > first.back());  // pas de marche vers 1,0
  CHECK(second.front() - first.back() < 0.005F);
  CHECK(second.back() == 1.0F);
  CHECK(third.front() == 1.0F);
}
