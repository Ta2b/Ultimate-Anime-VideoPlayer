// Empaquetage de l'anneau en RGB10A2 (kernels/spec/ring_pack.h, étape 5.5b).
#include "kernels/spec/ring_pack.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::kernels;

TEST_CASE("l'empaquetage 10 bits rend k/1023 exactement et sature hors de [0, 1]", "[kernels][ring]") {
  for (std::uint32_t k = 0; k <= 1023U; ++k) {
    CAPTURE(k);
    CHECK(unorm10(static_cast<float>(k) / 1023.0F) == k);
  }
  CHECK(unorm10(-0.5F) == 0U);
  CHECK(unorm10(std::numeric_limits<float>::quiet_NaN()) == 0U);
  CHECK(unorm10(1.0F) == 1023U);
  CHECK(unorm10(7.0F) == 1023U);
  CHECK(unorm10(std::numeric_limits<float>::infinity()) == 1023U);
  // Rouge dans les bits de poids faible, alpha opaque (VK_FORMAT_A2B10G10R10_UNORM_PACK32).
  CHECK(pack_rgb10a2(1023U, 0U, 0U) == 0xC00003FFU);
  CHECK(pack_rgb10a2(0U, 1023U, 0U) == 0xC00FFC00U);
  CHECK(pack_rgb10a2(0U, 0U, 1023U) == 0xFFF00000U);
  CHECK(pack_rgb10a2(0U, 0U, 0U) == 0xC0000000U);
}

TEST_CASE("le tramage TPDF de l'anneau reste dans un code, centré, et sature aux bornes", "[kernels][ring]") {
  double sum = 0.0;
  constexpr int kSide = 256;
  for (std::uint32_t y = 0; y < kSide; ++y) {
    for (std::uint32_t x = 0; x < kSide; ++x) {
      const float noise = ring_tpdf(x, y);
      REQUIRE(noise > -1.0F);
      REQUIRE(noise < 1.0F);
      sum += noise;
      CHECK(ring_tpdf(x, y) == noise);  // fixe par pixel
      // Une valeur au milieu de deux codes donne l'un des deux ; un code exact reste à ±1.
      const std::uint32_t mid = unorm10_dithered((100.5F) / 1023.0F, noise);
      CHECK((mid == 100U || mid == 101U || mid == 102U));
      const std::uint32_t exact = unorm10_dithered(500.0F / 1023.0F, noise);
      CHECK((exact >= 499U && exact <= 501U));
      CHECK(unorm10_dithered(0.0F, noise) <= 1U);
      CHECK(unorm10_dithered(1.0F, noise) >= 1022U);
      CHECK(unorm10_dithered(-1.0F, noise) <= 1U);
      CHECK(unorm10_dithered(2.0F, noise) <= 1023U);
    }
  }
  CHECK(std::fabs(sum / (kSide * kSide)) < 0.01);
}
