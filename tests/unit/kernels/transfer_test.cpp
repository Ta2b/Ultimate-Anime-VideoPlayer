#include "kernels/ref/transfer.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/spec.h"

using namespace ajp::kernels;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Les formules de spec/transfer.h en double, à partir des fractions exactes des normes.
namespace exact {
constexpr double kM1 = 2610.0 / 16384.0;
constexpr double kM2 = 2523.0 / 4096.0 * 128.0;
constexpr double kC1 = 3424.0 / 4096.0;
constexpr double kC2 = 2413.0 / 4096.0 * 32.0;
constexpr double kC3 = 2392.0 / 4096.0 * 32.0;
const double kHlgA = 0.17883277;
const double kHlgB = 1.0 - 4.0 * kHlgA;
const double kHlgC = 0.5 - kHlgA * std::log(4.0 * kHlgA);

double pq_eotf(double e) {
  const double p = std::pow(e, 1.0 / kM2);
  return std::pow(std::max(p - kC1, 0.0) / (kC2 - kC3 * p), 1.0 / kM1) * 10000.0;
}
double pq_inverse(double nits) {
  const double y = std::pow(nits / 10000.0, kM1);
  return std::pow((kC1 + kC2 * y) / (1.0 + kC3 * y), kM2);
}
double hlg_oetf(double e) {
  return e <= 1.0 / 12.0 ? std::sqrt(3.0 * e) : kHlgA * std::log(12.0 * e - kHlgB) + kHlgC;
}
double srgb_eotf(double v) {
  return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
}
double srgb_inverse(double l) {
  return l <= 0.0031308 ? l * 12.92 : 1.055 * std::pow(l, 1.0 / 2.4) - 0.055;
}
}  // namespace exact

// |f − d| ≤ rel |d| + abs × crête : la tolérance de spec/tolerance.h.
void check_transfer(float got, double want, double peak) {
  REQUIRE_THAT(got, WithinAbs(want, tolerance::kTransferRelative * std::abs(want) +
                                        tolerance::kTransferAbsolute * peak));
}

}  // namespace

TEST_CASE("transfert : constantes PQ exactes", "[transfer]") {
  static_assert(pq::kM1 == 0.1593017578125F);
  static_assert(pq::kM2 == 78.84375F);
  static_assert(pq::kC1 == 0.8359375F);
  static_assert(pq::kC2 == 18.8515625F);
  static_assert(pq::kC3 == 18.6875F);
  // Rangées en float : à un demi-pas de float près (3e-8 vers 0,5).
  CHECK_THAT(hlg::kB, WithinAbs(exact::kHlgB, 3e-8));
  CHECK_THAT(hlg::kC, WithinAbs(exact::kHlgC, 3e-8));
}

TEST_CASE("transfert : PQ, valeurs connues de BT.2100 et BT.2408", "[transfer]") {
  CHECK_THAT(ref::pq_inverse_eotf(100.0F), WithinAbs(0.508078, 1e-5));
  CHECK_THAT(ref::pq_inverse_eotf(203.0F), WithinAbs(0.580689, 1e-5));  // blanc de référence HDR, 58 %
  CHECK_THAT(ref::pq_inverse_eotf(1000.0F), WithinAbs(0.751827, 1e-5));
  CHECK_THAT(ref::pq_inverse_eotf(10000.0F), WithinAbs(1.0, 1e-6));
  CHECK_THAT(ref::pq_inverse_eotf(0.0F), WithinAbs(7.3e-7, 1e-7));  // c1^m2, pas tout à fait 0
  CHECK(ref::pq_eotf(0.0F) == 0.0F);
  CHECK_THAT(ref::pq_eotf(1.0F), WithinRel(10000.0, 1e-6));
  CHECK_THAT(ref::pq_eotf(0.5080784F), WithinRel(100.0, 1e-4));
  // Gris 18 % d'un blanc SDR à 203 cd/m², soit 36,54 cd/m².
  CHECK_THAT(ref::pq_inverse_eotf(0.18F * 203.0F), WithinAbs(exact::pq_inverse(0.18 * 203.0), 2e-6));
  // Entrées bornées : NaN et négatifs donnent le bas du domaine.
  CHECK(ref::pq_eotf(-0.5F) == 0.0F);
  CHECK(ref::pq_eotf(std::numeric_limits<float>::quiet_NaN()) == 0.0F);
  CHECK(ref::pq_inverse_eotf(20000.0F) == ref::pq_inverse_eotf(10000.0F));
}

TEST_CASE("transfert : PQ en float dans la tolérance du double", "[transfer]") {
  for (int i = 0; i <= 1000; ++i) {
    const double e = i / 1000.0;
    check_transfer(ref::pq_eotf(static_cast<float>(e)), exact::pq_eotf(static_cast<float>(e)), pq::kPeakNits);
    const double nits = std::pow(10.0, -3.0 + 7.0 * i / 1000.0);
    check_transfer(ref::pq_inverse_eotf(static_cast<float>(nits)),
                   exact::pq_inverse(static_cast<float>(nits)), 1.0);
  }
}

TEST_CASE("transfert : HLG, OETF et OOTF", "[transfer]") {
  CHECK(ref::hlg_oetf(0.0F) == 0.0F);
  CHECK_THAT(ref::hlg_oetf(1.0F / 12.0F), WithinAbs(0.5, 1e-6));
  CHECK_THAT(ref::hlg_oetf(1.0F), WithinAbs(1.0, 1e-5));
  CHECK_THAT(ref::hlg_oetf(0.18F), WithinAbs(exact::hlg_oetf(0.18), 1e-6));
  for (int i = 0; i <= 100; ++i) {
    const float e = static_cast<float>(i) / 100.0F;
    check_transfer(ref::hlg_oetf(e), exact::hlg_oetf(e), 1.0);
    CHECK_THAT(ref::hlg_inverse_oetf(ref::hlg_oetf(e)), WithinAbs(e, 2e-6));
  }
  // Signal 75 % sur un écran de 1000 cd/m² : le blanc de référence de BT.2408, 203 cd/m².
  const auto display = ref::hlg_display(1000.0F);
  REQUIRE(display);
  CHECK(display->gamma == 1.2F);
  const float scene = ref::hlg_inverse_oetf(0.75F);
  const Rgb white = ref::hlg_ootf({scene, scene, scene}, *display);
  CHECK_THAT(white.g, WithinAbs(203.0, 1.0));
  const Rgb peak = ref::hlg_ootf({1.0F, 1.0F, 1.0F}, *display);
  CHECK_THAT(peak.r, WithinRel(1000.0, 1e-5));
  CHECK(ref::hlg_ootf({0, 0, 0}, *display).b == 0.0F);
  const auto dim = ref::hlg_display(400.0F);
  REQUIRE(dim);
  CHECK_THAT(dim->gamma, WithinAbs(1.2 + 0.42 * std::log10(0.4), 1e-6));
  CHECK(ref::hlg_display(300.0F).error().code == ErrorCode::ParameterOutOfRange);
  CHECK(ref::hlg_display(std::numeric_limits<float>::quiet_NaN()).error().code ==
        ErrorCode::ParameterOutOfRange);
}

TEST_CASE("transfert : sRGB, valeurs connues et continuité au coude", "[transfer]") {
  CHECK_THAT(ref::srgb_eotf(0.5F), WithinAbs(0.214041, 1e-6));
  CHECK_THAT(ref::srgb_inverse_eotf(0.18F), WithinAbs(0.461356, 1e-6));  // gris 18 %
  CHECK(ref::srgb_eotf(1.0F) == 1.0F);
  CHECK(ref::srgb_eotf(0.0F) == 0.0F);
  CHECK_THAT(ref::srgb_eotf(srgb::kEncodedKnee), WithinAbs(srgb::kLinearKnee, 1e-7));
  for (int i = 0; i <= 255; ++i) {
    const float v = static_cast<float>(i) / 255.0F;
    check_transfer(ref::srgb_eotf(v), exact::srgb_eotf(v), 1.0);
    check_transfer(ref::srgb_inverse_eotf(v), exact::srgb_inverse(v), 1.0);
    CHECK_THAT(ref::srgb_inverse_eotf(ref::srgb_eotf(v)), WithinAbs(v, 2e-6));
  }
}

TEST_CASE("transfert : BT.1886, noir et blanc de l'écran", "[transfer]") {
  const auto ideal = ref::bt1886_display(1.0F, 0.0F);
  REQUIRE(ideal);
  CHECK_THAT(ref::bt1886_eotf(0.5F, *ideal), WithinAbs(std::pow(0.5, 2.4), 1e-6));
  CHECK_THAT(ref::bt1886_inverse_eotf(0.18F, *ideal), WithinAbs(std::pow(0.18, 1.0 / 2.4), 1e-6));
  const auto lcd = ref::bt1886_display(100.0F, 0.1F);
  REQUIRE(lcd);
  CHECK_THAT(ref::bt1886_eotf(0.0F, *lcd), WithinRel(0.1, 1e-4));
  CHECK_THAT(ref::bt1886_eotf(1.0F, *lcd), WithinRel(100.0, 1e-5));
  for (int i = 0; i <= 20; ++i) {
    const float v = static_cast<float>(i) / 20.0F;
    CHECK_THAT(ref::bt1886_inverse_eotf(ref::bt1886_eotf(v, *lcd), *lcd), WithinAbs(v, 1e-5));
  }
  CHECK(ref::bt1886_display(1.0F, 1.0F).error().code == ErrorCode::ParameterOutOfRange);
  CHECK(ref::bt1886_display(1.0F, -0.1F).error().code == ErrorCode::ParameterOutOfRange);
}

TEST_CASE("transfert : primaires BT.709 vers BT.2020, BT.2087", "[transfer]") {
  constexpr double r00 = kBt709ToBt2020.at(0, 0);  // calculée à la compilation
  static_assert(r00 > 0.627 && r00 < 0.628);
  const double bt2087[9] = {0.6274, 0.3293, 0.0433, 0.0691, 0.9195, 0.0114, 0.0164, 0.0880, 0.8956};
  for (int r = 0; r < 3; ++r) {
    double sum = 0.0;
    for (int c = 0; c < 3; ++c) {
      CHECK_THAT(kBt709ToBt2020.at(r, c), WithinAbs(bt2087[r * 3 + c], 5e-5));
      sum += kBt709ToBt2020.at(r, c);
    }
    CHECK_THAT(sum, WithinAbs(1.0, 1e-12));  // même blanc D65
  }
  const Mat3 identity = multiply(kBt2020ToBt709, kBt709ToBt2020);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) CHECK_THAT(identity.at(r, c), WithinAbs(r == c ? 1.0 : 0.0, 1e-12));
  }
  // Rangée en float, la matrice reste dans sa tolérance, et le blanc passe intact.
  const Rgb red = ref::convert_primaries({1.0F, 0.0F, 0.0F}, kBt709ToBt2020);
  CHECK_THAT(red.r, WithinAbs(kBt709ToBt2020.at(0, 0), tolerance::kMatrixFloat));
  CHECK_THAT(red.g, WithinAbs(kBt709ToBt2020.at(1, 0), tolerance::kMatrixFloat));
  const Rgb white = ref::convert_primaries({1.0F, 1.0F, 1.0F}, kBt709ToBt2020);
  CHECK_THAT(white.b, WithinAbs(1.0, 1e-6));
}
