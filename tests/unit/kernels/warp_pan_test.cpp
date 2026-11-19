// Mouvement d'ensemble du warp (kernels/spec/warp_pan.h, kernels/ref/warp_pan.cpp) : estimation de la
// translation, remplacement du flot, fondu sous un demi-pixel, garder les poses intact.
#include "warp_pan_test.h"

#include <algorithm>
#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/ref/warp.h"
#include "kernels/spec.h"

using namespace kirari::kernels;
using namespace kirari::test;
using Catch::Matchers::WithinAbs;

namespace {

float max_diff(const RgbImage& a, const RgbImage& b) {
  float worst = 0.0F;
  for (std::size_t c = 0; c < 3; ++c) {
    for (int y = 0; y < a.height(); ++y) {
      for (int x = 0; x < a.width(); ++x)
        worst = std::max(worst, std::fabs(a.planes[c].at(x, y) - b.planes[c].at(x, y)));
    }
  }
  return worst;
}

}  // namespace

TEST_CASE("mouvement d'ensemble : la translation d'une texture lisse au centième de pixel", "[warp][pan]") {
  const float dx = GENERATE(0.3F, -0.6F, 1.7F);
  const float dy = GENERATE(0.0F, 0.45F);
  const PanEstimate e = ref::pan_estimate(pan_image(96, 64, 0.0F, 0.0F), pan_image(96, 64, dx, dy));
  CHECK(e.sure);
  CHECK_THAT(e.dx, WithinAbs(dx, 0.01));
  CHECK_THAT(e.dy, WithinAbs(dy, 0.01));
}

TEST_CASE("mouvement d'ensemble : une image sans texture n'est pas sûre", "[warp][pan]") {
  CHECK_FALSE(ref::pan_estimate(RgbImage(64, 48, 0.4F), RgbImage(64, 48, 0.4F)).sure);
  CHECK_FALSE(ref::pan_estimate(RgbImage(8, 8, 0.4F), RgbImage(8, 8, 0.4F)).sure);
}

TEST_CASE("mouvement d'ensemble : un flot trop petit prend la translation de la paire", "[warp][pan]") {
  // RIFE ne voit presque rien (flot nul) ; avec pan_lock, la sortie est celle du flot exact (±D/2).
  const RgbImage a = pan_image(64, 48, 0.0F, 0.0F);
  const RgbImage b = pan_image(64, 48, kPanDx, kPanDy);
  ref::Motion blind = pan_motion(64, 48, 0.0F, 0.0F, 0.0F, 0.0F);
  ref::Motion exact = pan_motion(64, 48, -kPanDx / 2, -kPanDy / 2, kPanDx / 2, kPanDy / 2);
  blind.pan_lock = true;
  exact.pan_lock = true;
  for (const float t : {0.25F, 0.5F, 0.75F}) {
    CAPTURE(t);
    const auto got = ref::warp_blend_at(a, b, blind, t);
    const auto want = ref::warp_blend_at(a, b, exact, t);
    REQUIRE(got);
    REQUIRE(want);
    CHECK(max_diff(*got, *want) < 2e-3F);
  }
}

TEST_CASE("mouvement d'ensemble : sous un demi-pixel, un fondu linéaire", "[warp][pan]") {
  // Images uniformes (estimation non sûre) : seul le fondu agit, poids 1 − t, masque ignoré.
  RgbImage a(16, 8, 1.0F);
  RgbImage b(16, 8, 0.0F);
  ref::Motion motion = pan_motion(16, 8, 0.0F, 0.0F, 0.0F, 0.0F);
  motion.mask = PlaneF(16, 8, 0.9F);
  motion.pan_lock = true;
  const float t = 0.25F;
  const auto out = ref::warp_blend_at(a, b, motion, t);
  REQUIRE(out);
  CHECK_THAT(out->planes[0].at(8, 4), WithinAbs(1.0 - t, 1e-5));
  // Sans pan_lock, la loi ajustée colle l'instant à a.
  motion.pan_lock = false;
  const auto law = ref::warp_blend_at(a, b, motion, t);
  REQUIRE(law);
  CHECK(law->planes[0].at(8, 4) > 0.99F);
}

TEST_CASE("mouvement d'ensemble : pas de saut entre 0,499 et 0,5", "[warp][pan]") {
  const RgbImage a = pan_image(48, 32, 0.0F, 0.0F);
  const RgbImage b = pan_image(48, 32, kPanDx, kPanDy);
  ref::Motion motion = pan_motion(48, 32, 0.0F, 0.0F, 0.0F, 0.0F);
  motion.mask = PlaneF(48, 32, 0.8F);
  motion.pan_lock = true;
  const auto mid = ref::warp_blend_at(a, b, motion, 0.5F);
  const auto near = ref::warp_blend_at(a, b, motion, 0.499F);
  REQUIRE(mid);
  REQUIRE(near);
  CHECK(max_diff(*mid, *near) < 0.01F);
}

TEST_CASE("mouvement d'ensemble : garder les poses n'est pas touché", "[warp][pan]") {
  const RgbImage a = pan_image(48, 32, 0.0F, 0.0F);
  const RgbImage b = pan_image(48, 32, kPanDx, kPanDy);
  ref::Motion off = pan_motion(48, 32, 0.0F, 0.0F, 0.0F, 0.0F);
  ref::Motion on = off;
  on.pan_lock = true;
  const auto x = ref::warp_blend_poses(a, b, off, nullptr, 0.25F);
  const auto y = ref::warp_blend_poses(a, b, on, nullptr, 0.25F);
  REQUIRE(x);
  REQUIRE(y);
  CHECK(*x == *y);
}

TEST_CASE("mouvement d'ensemble : un flot de l'instant de la sortie (at_instant)", "[warp][pan]") {
  // DistilDRBA et RIFE direct : le flot est celui de t. Flot nul et pan_lock : la sortie de la translation
  // à t, (−t D, (1 − t) D), et non au milieu.
  const RgbImage a = pan_image(64, 48, 0.0F, 0.0F);
  const RgbImage b = pan_image(64, 48, kPanDx, kPanDy);
  for (const float t : {0.25F, 0.75F}) {
    CAPTURE(t);
    ref::Motion blind = pan_motion(64, 48, 0.0F, 0.0F, 0.0F, 0.0F);
    ref::Motion exact =
        pan_motion(64, 48, -t * kPanDx, -t * kPanDy, (1.0F - t) * kPanDx, (1.0F - t) * kPanDy);
    for (ref::Motion* m : {&blind, &exact}) {
      m->pan_lock = true;
      m->at_instant = true;
    }
    const auto got = ref::warp_blend_at(a, b, blind, t);
    const auto want = ref::warp_blend_at(a, b, exact, t);
    REQUIRE(got);
    REQUIRE(want);
    CHECK(max_diff(*got, *want) < 2e-3F);
  }
}

TEST_CASE("at_instant : flot et masque lus tels quels à tout instant", "[warp][pan]") {
  const RgbImage a = pan_image(48, 32, 0.0F, 0.0F);
  const RgbImage b = pan_image(48, 32, kPanDx, kPanDy);
  ref::Motion motion = pan_motion(48, 32, -0.3F, 0.1F, 0.6F, -0.2F);
  motion.mask = PlaneF(48, 32, 0.7F);
  const auto mid = ref::warp_blend(a, b, motion);
  motion.at_instant = true;
  const auto at = ref::warp_blend_at(a, b, motion, 0.3F);
  REQUIRE(mid);
  REQUIRE(at);
  CHECK(*mid == *at);
  // Le fondu sous un demi-pixel va vers le poids 1 − t de l'instant donné.
  RgbImage white(16, 8, 1.0F);
  RgbImage black(16, 8, 0.0F);
  ref::Motion still = pan_motion(16, 8, 0.0F, 0.0F, 0.0F, 0.0F);
  still.mask = PlaneF(16, 8, 0.9F);
  still.pan_lock = true;
  still.at_instant = true;
  const auto faded = ref::warp_blend_at(white, black, still, 0.3F);
  REQUIRE(faded);
  CHECK_THAT(faded->planes[0].at(8, 4), WithinAbs(0.7, 1e-5));
}
