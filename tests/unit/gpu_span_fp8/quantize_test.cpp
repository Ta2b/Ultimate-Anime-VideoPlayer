// Étape 10.3 : poids FP8 des blocs du SPAN quantifiés par canal de sortie (gpu/span_fp8/quantize.h).
#include "gpu/span_fp8/quantize.h"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/span_fp8/derive.h"  // les deux ensemble : une seule kE4m3Max (A02-09)

using namespace ajp::gpu::span_fp8;

TEST_CASE("E4M3 sur l'hôte : valeurs exactes, arrondi pair, saturation", "[span_fp8][quantize]") {
  CHECK(e4m3_round(0.0F) == 0.0F);
  CHECK(e4m3_round(1.0F) == 1.0F);
  CHECK(e4m3_round(448.0F) == 448.0F);
  CHECK(e4m3_round(1000.0F) == 448.0F);
  CHECK(e4m3_round(-1000.0F) == -448.0F);
  CHECK(e4m3_round(1.125F) == 1.125F);  // 1 + 1/8 : représentable
  CHECK(e4m3_round(1.0625F) == 1.0F);   // au milieu de 1 et 1,125 : le pair (mantisse 0)
  CHECK(e4m3_round(1.1875F) == 1.25F);  // au milieu de 1,125 et 1,25 : le pair (mantisse 2)
  CHECK(e4m3_round(-3.3F) == -3.25F);
  CHECK(e4m3_round(std::ldexp(1.0F, -9)) == std::ldexp(1.0F, -9));  // plus petit dénormal
  CHECK(e4m3_round(std::ldexp(1.0F, -11)) == 0.0F);
  CHECK(e4m3_round(240.0F) == 240.0F);
  CHECK(e4m3_round(250.0F) == 256.0F);
}

TEST_CASE("par canal, l'erreur de reconstruction ne dépasse jamais celle par tenseur",
          "[span_fp8][quantize]") {
  // 64 canaux de sortie de 9 × 64 poids, aux amplitudes très différentes : avec une échelle commune, le canal
  // fort pousse les faibles sous le plus petit normal d'E4M3 (2^−6), où le pas devient absolu (2^−9) et la
  // précision relative s'effondre. E4M3 étant un format flottant, deux canaux d'amplitudes voisines, eux, se
  // quantifient aussi bien d'une façon que de l'autre : le gain de l'étape 10.3 est là.
  constexpr std::size_t kOut = 64, kPer = 9 * 64;
  std::mt19937 rng(10);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> weights(kOut * kPer);
  for (std::size_t c = 0; c < kOut; ++c) {
    const float amplitude = c == 0 ? 4.0F : 2e-5F * static_cast<float>(1 + c % 3);
    for (std::size_t i = 0; i < kPer; ++i) weights[c * kPer + i] = amplitude * normal(rng);
  }
  const std::vector<float> per_channel = channel_scales(weights, kOut);
  const std::vector<float> per_tensor(kOut, tensor_scale(weights));
  const std::vector<float> a = reconstruct(weights, per_channel);
  const std::vector<float> b = reconstruct(weights, per_tensor);
  double total_a = 0.0, total_b = 0.0;
  for (std::size_t c = 0; c < kOut; ++c) {
    double ea = 0.0, eb = 0.0;
    for (std::size_t i = c * kPer; i < (c + 1) * kPer; ++i) {
      ea += std::pow(static_cast<double>(a[i] - weights[i]), 2);
      eb += std::pow(static_cast<double>(b[i] - weights[i]), 2);
    }
    CAPTURE(c);
    CHECK(ea <= eb * (1.0 + 1e-9) + 1e-30);
    if (c == 0) continue;  // le canal fort a la même échelle des deux façons
    total_a += ea;
    total_b += eb;
  }
  CHECK(total_a < 0.5 * total_b);  // les canaux faibles, bien mieux rendus
  // Un canal nul garde l'échelle 1 et se reconstruit nul.
  std::vector<float> zero(2 * kPer, 0.0F);
  for (std::size_t i = kPer; i < 2 * kPer; ++i) zero[i] = 0.5F;
  const std::vector<float> scales = channel_scales(zero, 2);
  CHECK(scales[0] == 1.0F);
  CHECK(scales[1] == 0.5F / kE4m3Max);
}

TEST_CASE("derive.h et quantize.h se lisent ensemble, une seule borne E4M3 (A02-09)",
          "[span_fp8][quantize]") {
  STATIC_CHECK(kE4m3Max == 448.0F);
  CHECK(e4m3_round(kE4m3Max * 2.0F) == kE4m3Max);
}
