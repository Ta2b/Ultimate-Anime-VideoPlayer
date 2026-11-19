#include "kernels/ref/motion_camera.h"

#include <array>
#include <cmath>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/ref/warp.h"
#include "kernels/spec/motion_camera.h"

using namespace kirari::kernels;
using Catch::Matchers::WithinAbs;

namespace {

RgbImage ramp(int w, int h, float offset) {
  RgbImage image(w, h);
  for (PlaneF& p : image.planes) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) p.at(x, y) = static_cast<float>(x) + offset;
    }
  }
  return image;
}

}  // namespace

// Un seul test pour le module : translation, fenêtre et échelle, fondu, zoom au premier ordre, refus.
TEST_CASE("mouvement de la caméra : un panoramique pur se fait par le warp", "[motion_camera]") {
  // Translation de 4 pixels vers la droite, plans 32 × 4, fenêtre entière.
  const std::array<float, 6> pan{4.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F};
  const Window whole{.x = 0, .y = 0, .width = 32, .height = 4};
  auto motion = ref::motion_camera(pan, 1.0F, 32, 4, whole);
  REQUIRE(motion);
  CHECK(motion->flow[0].at(5, 2) == -2.0F);
  CHECK(motion->flow[1].at(5, 2) == 0.0F);
  CHECK(motion->flow[2].at(5, 2) == 2.0F);
  CHECK(motion->mask.at(5, 2) == 0.5F);
  // b est a décalé de 4 : à l'instant t, le warp rend a décalé de 4 t, loin des bords.
  const RgbImage a = ramp(32, 4, 0.0F);
  const RgbImage b = ramp(32, 4, -4.0F);
  for (const float t : {0.25F, 0.5F, 0.75F}) {
    auto out = ref::warp_blend_at(a, b, *motion, t);
    REQUIRE(out);
    CHECK_THAT(out->planes[0].at(16, 1), WithinAbs(16.0 - 4.0 * t, 1e-4));
  }

  // Fenêtre 8 × 2 en (2, 1) de plans 12 × 4, travail deux fois plus grand (échelle 1/2) : le déplacement en
  // pixels de la fenêtre est moitié ; dehors, rien.
  const Window area{.x = 2, .y = 1, .width = 8, .height = 2};
  auto small = ref::motion_camera(pan, 0.5F, 12, 4, area);
  REQUIRE(small);
  CHECK(small->flow[2].at(4, 2) == 1.0F);
  CHECK(small->flow[2].at(0, 0) == 0.0F);
  CHECK(small->mask.at(11, 3) == 0.5F);

  // La caméra identité : le fondu enchaîné.
  auto still = ref::motion_camera(kCameraStill, 1.0F, 32, 4, whole);
  REQUIRE(still);
  for (const PlaneF& p : still->flow) CHECK(p.at(7, 3) == 0.0F);

  // Zoom de 1 % centré à l'origine : le déplacement croît avec la position.
  const std::array<float, 6> zoom{0.0F, 1.01F, 0.0F, 0.0F, 0.0F, 1.01F};
  auto zoomed = ref::motion_camera(zoom, 1.0F, 32, 4, whole);
  REQUIRE(zoomed);
  CHECK_THAT(zoomed->flow[2].at(20, 0), WithinAbs(0.1, 1e-5));
  CHECK_THAT(zoomed->flow[3].at(20, 2), WithinAbs(0.01, 1e-5));

  // Refus : fenêtre hors des plans, échelle nulle, caméra non finie.
  CHECK_FALSE(ref::motion_camera(pan, 1.0F, 8, 4, whole));
  CHECK_FALSE(ref::motion_camera(pan, 0.0F, 32, 4, whole));
  std::array<float, 6> broken = pan;
  broken[0] = std::numeric_limits<float>::quiet_NaN();
  CHECK_FALSE(ref::motion_camera(broken, 1.0F, 32, 4, whole));
}
