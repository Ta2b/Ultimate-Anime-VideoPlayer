#include "gpu/ir/half.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::gpu::ir;

TEST_CASE("half : valeurs remarquables", "[gpu_ir][half]") {
  CHECK(half_to_float(0x3C00) == 1.0F);
  CHECK(half_to_float(0xC000) == -2.0F);
  CHECK(half_to_float(0x7BFF) == 65504.0F);
  CHECK(half_to_float(0x0400) == std::ldexp(1.0F, -14));  // plus petit normal
  CHECK(half_to_float(0x0001) == std::ldexp(1.0F, -24));  // plus petit sous-normal
  CHECK(half_to_float(0x03FF) == std::ldexp(1023.0F, -24));
  CHECK(std::signbit(half_to_float(0x8000)));
  CHECK(half_to_float(0x7C00) == std::numeric_limits<float>::infinity());
  CHECK(std::isnan(half_to_float(0x7E00)));

  CHECK(float_to_half(1.0F) == 0x3C00);
  CHECK(float_to_half(-0.0F) == 0x8000);
  CHECK(float_to_half(std::numeric_limits<float>::infinity()) == 0x7C00);
  CHECK(float_to_half(-std::numeric_limits<float>::infinity()) == 0xFC00);
  CHECK(std::isnan(half_to_float(float_to_half(std::numeric_limits<float>::quiet_NaN()))));
  // Un NaN dont seuls les bits bas de mantisse sont posés reste un NaN.
  CHECK(std::isnan(half_to_float(float_to_half(std::bit_cast<float>(0x7F800001U)))));
}

TEST_CASE("half : arrondi au plus proche, égalité vers le pair", "[gpu_ir][half]") {
  const float ulp = std::ldexp(1.0F, -10);                   // écart entre F16 voisins dans [1, 2[
  CHECK(float_to_half(1.0F + (ulp / 2)) == 0x3C00);          // égalité : vers le pair
  CHECK(float_to_half(1.0F + (3 * ulp / 2)) == 0x3C02);      // égalité : vers le pair
  CHECK(float_to_half(1.0F + (ulp / 2) + 1e-7F) == 0x3C01);  // juste au-dessus
  CHECK(float_to_half(2.0F - (ulp / 4)) == 0x4000);          // report vers l'exposant suivant
  CHECK(float_to_half(65519.0F) == 0x7BFF);
  CHECK(float_to_half(65520.0F) == 0x7C00);  // au-delà de 65504 + ulp/2 : infini
  CHECK(float_to_half(1e10F) == 0x7C00);
  CHECK(float_to_half(std::ldexp(1.0F, -25)) == 0x0000);  // égalité avec zéro : vers le pair
  CHECK(float_to_half(std::ldexp(1.5F, -25)) == 0x0001);
  CHECK(float_to_half(std::ldexp(1.0F, -26)) == 0x0000);
  CHECK(float_to_half(std::ldexp(1023.5F, -24)) == 0x0400);  // sous-normal arrondi au plus petit normal
  CHECK(float_to_half(std::numeric_limits<float>::denorm_min()) == 0x0000);
}

TEST_CASE("half : aller-retour exact sur les 65536 motifs", "[gpu_ir][half]") {
  for (std::uint32_t bits = 0; bits <= 0xFFFFU; ++bits) {
    const auto h = static_cast<std::uint16_t>(bits);
    const float f = half_to_float(h);
    if (std::isnan(f)) {
      CHECK(std::isnan(half_to_float(float_to_half(f))));
    } else if (float_to_half(f) != h) {
      FAIL("motif " << bits << " non retrouvé");
    }
  }
}

// Propriété indépendante de l'algorithme : aucun voisin F16 n'est plus proche que le résultat.
TEST_CASE("half : le résultat est le F16 le plus proche", "[gpu_ir][half]") {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> exponent(-26.0F, 16.5F);
  std::uniform_real_distribution<float> mantissa(1.0F, 2.0F);
  for (int k = 0; k < 20000; ++k) {
    const float f = std::ldexp(mantissa(rng), static_cast<int>(exponent(rng)));
    const std::uint16_t h = float_to_half(f);
    if (h == 0x7C00) {
      CHECK(f >= 65520.0F);
      continue;
    }
    const double chosen = std::abs(static_cast<double>(half_to_float(h)) - f);
    for (const int step : {-1, 1}) {
      const int neighbour = static_cast<int>(h) + step;
      if (neighbour < 0 || neighbour >= 0x7C00) continue;
      const double other =
          std::abs(static_cast<double>(half_to_float(static_cast<std::uint16_t>(neighbour))) - f);
      if (other < chosen) FAIL("f = " << f << " : voisin plus proche que " << h);
    }
  }
}
