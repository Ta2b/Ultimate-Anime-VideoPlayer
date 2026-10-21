#include "kernels/ref/half.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::kernels;

TEST_CASE("fp16 : valeurs exactes et arrondi au pair", "[half]") {
  CHECK(ref::float_to_half(0.0F) == 0x0000);
  CHECK(ref::float_to_half(-0.0F) == 0x8000);
  CHECK(ref::float_to_half(1.0F) == 0x3C00);
  CHECK(ref::float_to_half(-2.0F) == 0xC000);
  CHECK(ref::float_to_half(0.5F) == 0x3800);
  CHECK(ref::float_to_half(65504.0F) == 0x7BFF);  // plus grand fini
  CHECK(ref::round_to_half(1.0F / 3.0F) == 0.333251953125F);
  // 1 + 2^−11 : à mi-chemin de 1 et de 1 + 2^−10, va au pair (1).
  CHECK(ref::round_to_half(1.0F + 0x1p-11F) == 1.0F);
  // 1 + 3 × 2^−11 : à mi-chemin, va au pair (1 + 2^−9).
  CHECK(ref::round_to_half(1.0F + 3.0F * 0x1p-11F) == 1.0F + 0x1p-9F);
  CHECK(ref::round_to_half(1.0F + 0x1p-11F + 0x1p-20F) == 1.0F + 0x1p-10F);
  // Une retenue passe dans l'exposant : 2 − 2^−12 s'arrondit à 2.
  CHECK(ref::round_to_half(2.0F - 0x1p-12F) == 2.0F);
}

TEST_CASE("fp16 : débordement, NaN et sous-normaux", "[half]") {
  CHECK(ref::float_to_half(65519.0F) == 0x7BFF);
  CHECK(ref::float_to_half(65520.0F) == 0x7C00);  // égalité : au pair, l'infini
  CHECK(ref::float_to_half(1e9F) == 0x7C00);
  CHECK(ref::float_to_half(-std::numeric_limits<float>::infinity()) == 0xFC00);
  CHECK(std::isnan(ref::round_to_half(std::numeric_limits<float>::quiet_NaN())));
  CHECK(ref::round_to_half(0x1p-14F) == 0x1p-14F);  // plus petit normal
  CHECK(ref::float_to_half(0x1p-24F) == 0x0001);    // plus petit sous-normal
  CHECK(ref::float_to_half(0x1p-25F) == 0x0000);    // égalité avec 0 : au pair
  CHECK(ref::float_to_half(1.5F * 0x1p-25F) == 0x0001);
  CHECK(ref::float_to_half(3.0F * 0x1p-25F) == 0x0002);  // 1,5 unité : au pair, 2
  CHECK(ref::round_to_half(0x1p-20F) == 0x1p-20F);
  CHECK(ref::float_to_half(0x1p-30F) == 0x0000);
  CHECK(ref::half_to_float(0x7C00) == std::numeric_limits<float>::infinity());
  CHECK(ref::half_to_float(0x8001) == -0x1p-24F);
}

TEST_CASE("fp16 : aller-retour exact de tous les codes finis", "[half]") {
  for (std::uint32_t code = 0; code < 0x10000U; ++code) {
    const auto bits = static_cast<std::uint16_t>(code);
    if ((bits & 0x7C00U) == 0x7C00U) continue;  // infinis et NaN
    REQUIRE(ref::float_to_half(ref::half_to_float(bits)) == bits);
  }
}

TEST_CASE("fp16 : une image arrondie plan par plan", "[half]") {
  RgbImage image(2, 2, 1.0F / 3.0F);
  image.planes[1].at(1, 1) = 70000.0F;
  const RgbImage rounded = ref::round_to_half(image);
  CHECK(rounded.planes[0].at(0, 0) == 0.333251953125F);
  CHECK(std::isinf(rounded.planes[1].at(1, 1)));
}
