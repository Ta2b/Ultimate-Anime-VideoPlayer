#include "audio/hrir_synth.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace ajp::audio {
namespace {

constexpr double kSpeedOfSound = 343.0;
constexpr std::array<double, kHrirPositions> kAzimuthDeg{-30.0,  30.0,  0.0,   0.0,
                                                         -135.0, 135.0, -90.0, 90.0};  // + : droite
constexpr float kOutputScale =
    0.72F;  // la somme des deux enceintes, des réflexions et du direct reste sous le plein niveau

// Un générateur congruentiel : la queue diffuse est la même d'un lancement à l'autre.
struct Noise {
  std::uint32_t state;
  float next() {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(static_cast<std::int32_t>(state)) / 2147483648.0F;  // [-1, 1[
  }
};

// Ajoute à h la réponse d'un filtre passe-bas à un pôle de coupure fc, de gain `gain`, retardée de `delay`
// échantillons.
void add_lowpass(std::vector<float>& h, std::size_t delay, double fc, double rate, float gain) {
  const double a = std::exp(-2.0 * std::numbers::pi * fc / rate);
  double tap = (1.0 - a);
  for (std::size_t n = 0; delay + n < h.size() && tap > 1e-5; ++n) {
    h[delay + n] += gain * static_cast<float>(tap);
    tap *= a;
  }
}

void add_reflections(std::vector<float>& h, int rate, float level, std::uint32_t seed) {
  if (level <= 0.0F) return;
  constexpr std::array<double, 5> kDelaysMs{6.2, 9.7, 13.4, 17.9, 23.1};
  constexpr std::array<float, 5> kGains{0.30F, 0.24F, 0.20F, 0.15F, 0.11F};
  for (std::size_t i = 0; i < kDelaysMs.size(); ++i) {
    const auto at = static_cast<std::size_t>(kDelaysMs[i] * 0.001 * rate);
    if (at < h.size()) h[at] += level * kGains[i] * ((seed + i) % 2 == 0 ? 1.0F : -1.0F);
  }
  // Queue diffuse : bruit adouci, décroissance exponentielle (RT60 0,4 s), à partir de 28 ms.
  Noise noise{seed * 2654435761U + 12345U};
  const std::size_t start = static_cast<std::size_t>(0.028 * rate);
  float low = 0.0F;
  const float alpha = static_cast<float>(std::exp(-2.0 * std::numbers::pi * 4500.0 / rate));
  for (std::size_t n = start; n < h.size(); ++n) {
    const double t = static_cast<double>(n - start) / rate;
    low = alpha * low + (1.0F - alpha) * noise.next();
    h[n] += level * 0.9F * static_cast<float>(std::exp(-6.9 * t / 0.4)) * low * 2.0F;
  }
}

}  // namespace

Hrir synthetic_hrir(const SyntheticHrirParams& params) {
  const int rate = std::clamp(params.sample_rate, 8'000, 192'000);
  const float amount = std::clamp(params.amount, 0.0F, 1.0F);
  const float room = std::clamp(params.room, 0.0F, 1.0F) * amount;
  const double radius = std::clamp<double>(params.head_radius_m, 0.07, 0.11);
  const std::size_t length = static_cast<std::size_t>(0.2 * rate);  // 200 ms
  constexpr std::size_t kBase = 16;  // marge avant le direct, pour le retard et les creux
  Hrir out;
  out.sample_rate = rate;
  for (std::size_t position = 0; position < kHrirPositions; ++position) {
    if (position == 3) continue;  // LFE : sans réponse, il passe directement aux deux oreilles
    const double az = kAzimuthDeg[position] * std::numbers::pi / 180.0;
    const double lateral = std::sin(az);  // + : à droite
    const double phi = std::asin(std::min(1.0, std::abs(lateral)));
    const double itd = radius / kSpeedOfSound * (phi + std::sin(phi));  // secondes (Woodworth)
    const bool rear = std::cos(az) < -0.3;
    for (std::size_t ear = 0; ear < 2; ++ear) {
      const double side = ear == 0 ? -1.0 : 1.0;
      const bool near_ear = std::abs(lateral) < 1e-6 || side * lateral > 0.0;
      std::vector<float>& h = out.taps[position][ear];
      h.assign(length, 0.0F);
      if (near_ear) {
        h[kBase] = 1.0F;
        // Un creux de pavillon léger (retard de 0,13 ms) : la source se place devant plutôt que dans la tête.
        const auto notch = static_cast<std::size_t>(0.00013 * rate);
        h[kBase + notch] -= 0.2F * amount;
      } else {
        // L'oreille éloignée : retardée de l'écart interaural, assombrie et plus faible.
        const auto delay = kBase + static_cast<std::size_t>(std::lround(itd * rate));
        const double fc = 4800.0 - 3300.0 * std::sin(phi);
        const float gain = amount * static_cast<float>(std::pow(10.0, -(2.5 + 6.0 * std::sin(phi)) / 20.0));
        add_lowpass(h, delay, fc, rate, gain);
      }
      if (rear) {  // derrière la tête : les aigus sont masqués par le pavillon
        std::vector<float> softened(length, 0.0F);
        for (std::size_t n = 0; n < length; ++n) softened[n] = h[n];
        const double a = std::exp(-2.0 * std::numbers::pi * 6500.0 / rate);
        float low = 0.0F;
        for (std::size_t n = 0; n < length; ++n) {
          low = static_cast<float>(a) * low + (1.0F - static_cast<float>(a)) * softened[n];
          h[n] = low;
        }
      }
      add_reflections(h, rate, 0.22F * room, static_cast<std::uint32_t>(position * 2 + ear + 1));
      for (float& tap : h) tap *= kOutputScale;
    }
  }
  return out;
}

}  // namespace ajp::audio
