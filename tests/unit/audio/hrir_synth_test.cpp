#include "audio/hrir_synth.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "audio/mixer.h"

using Catch::Approx;
using namespace ajp::audio;

namespace {

float energy(const std::vector<float>& taps, std::size_t from, std::size_t to) {
  float sum = 0.0F;
  for (std::size_t i = from; i < std::min(to, taps.size()); ++i) sum += taps[i] * taps[i];
  return sum;
}

std::size_t first_peak(const std::vector<float>& taps) {
  const auto at = std::ranges::max_element(taps, [](float a, float b) { return std::abs(a) < std::abs(b); });
  return static_cast<std::size_t>(at - taps.begin());
}

}  // namespace

TEST_CASE("hrir synthétique : sans effet, chaque enceinte va à son oreille seulement", "[hrir_synth]") {
  const Hrir hrir = synthetic_hrir({.amount = 0.0F, .room = 0.0F, .sample_rate = 48'000});
  CHECK(energy(hrir.taps[0][0], 0, 4096) > 0.4F);   // FL → oreille gauche
  CHECK(energy(hrir.taps[0][1], 0, 4096) == 0.0F);  // FL → oreille droite : rien
  CHECK(energy(hrir.taps[1][1], 0, 4096) > 0.4F);
  CHECK(energy(hrir.taps[1][0], 0, 4096) == 0.0F);
}

TEST_CASE("hrir synthétique : l'oreille éloignée arrive plus tard, plus faible et plus sombre",
          "[hrir_synth]") {
  const Hrir hrir = synthetic_hrir({.amount = 1.0F, .room = 0.0F, .sample_rate = 48'000});
  const auto& near_ear = hrir.taps[0][0];  // enceinte gauche, oreille gauche
  const auto& far_ear = hrir.taps[0][1];
  const std::size_t gap = first_peak(far_ear) - first_peak(near_ear);
  CHECK(gap >= 6);   // plus de 0,12 ms
  CHECK(gap <= 30);  // moins de 0,6 ms : le retard interaural d'une tête
  CHECK(energy(far_ear, 0, 4096) < 0.35F * energy(near_ear, 0, 4096));
  CHECK(energy(far_ear, 0, 4096) > 0.0F);
  // Symétrie : la droite est le miroir de la gauche.
  CHECK(hrir.taps[1][1] == hrir.taps[0][0]);
  CHECK(hrir.taps[1][0] == hrir.taps[0][1]);
}

TEST_CASE("hrir synthétique : la pièce ajoute des réflexions tardives, sans toucher le direct",
          "[hrir_synth]") {
  const Hrir dry = synthetic_hrir({.amount = 1.0F, .room = 0.0F, .sample_rate = 48'000});
  const Hrir wet = synthetic_hrir({.amount = 1.0F, .room = 1.0F, .sample_rate = 48'000});
  const std::size_t late = 48'000 / 100;  // après 10 ms
  CHECK(energy(dry.taps[0][0], late, 9'600) == 0.0F);
  CHECK(energy(wet.taps[0][0], late, 9'600) > 0.001F);
  CHECK(energy(wet.taps[0][0], 0, 48) == Approx(energy(dry.taps[0][0], 0, 48)).epsilon(0.05));
  for (float tap : wet.taps[0][0]) CHECK(std::isfinite(tap));
}

TEST_CASE("mixeur : les enceintes virtuelles du casque décorrèlent la stéréo et gardent le niveau",
          "[mixer][hrir_synth]") {
  const int rate = 48'000;
  // Un signal à gauche seulement : sans traitement, rien à droite.
  const auto run = [&](float virtual_speakers) {
    MixTuning tuning;
    tuning.virtual_speakers = virtual_speakers;
    tuning.room = 0.3F;
    Mixer mixer(MixConfig{.listen = Listen::Headphones, .tuning = tuning, .sample_rate = rate});
    PcmBlock block;
    block.channels = 2;
    block.samples.assign(2 * 8192, 0.0F);
    for (std::size_t f = 0; f < 8192; ++f)
      block.samples[2 * f] = 0.5F * std::sin(0.05F * static_cast<float>(f));
    PcmBlock out = mixer.process(block);
    float left = 0.0F;
    float right = 0.0F;
    for (std::size_t f = 1024; f + 1 < out.samples.size() / 2; ++f) {
      left += out.samples[2 * f] * out.samples[2 * f];
      right += out.samples[2 * f + 1] * out.samples[2 * f + 1];
    }
    return std::pair{left, right};
  };
  const auto [plain_left, plain_right] = run(0.0F);
  const auto [virt_left, virt_right] = run(1.0F);
  CHECK(virt_right > 0.05F * virt_left);   // l'oreille éloignée entend la source
  CHECK(virt_right > 3.0F * plain_right);  // bien plus que le crossfeed seul
  CHECK(virt_left > 0.2F * plain_left);    // le niveau reste du même ordre
  CHECK(virt_left < 2.0F * plain_left);
}
