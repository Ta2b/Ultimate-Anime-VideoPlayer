#include "kernels/ref/optical_flow.h"

#include <cstdint>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/ref/warp.h"

using namespace ajp::kernels;
using Catch::Matchers::WithinAbs;

namespace {

// Flot uniforme en pixels de la grille.
ref::FlowField uniform(int w, int h, float x, float y) {
  return {.x = PlaneF(w, h, x), .y = PlaneF(w, h, y)};
}

// Un objet de 8 points de large, en [8, 16) dans a, qui avance de 4 sur un fond fixe : en [12, 20) dans b.
ref::FlowField object_forward() {
  ref::FlowField f = uniform(32, 4, 0.0F, 0.0F);
  for (int y = 0; y < 4; ++y)
    for (int x = 8; x < 16; ++x) f.x.at(x, y) = 4.0F;
  return f;
}
ref::FlowField object_backward() {
  ref::FlowField f = uniform(32, 4, 0.0F, 0.0F);
  for (int y = 0; y < 4; ++y)
    for (int x = 12; x < 20; ++x) f.x.at(x, y) = -4.0F;
  return f;
}

}  // namespace

TEST_CASE("flux optique : S10.5 en pixels de la grille", "[optical_flow]") {
  CHECK(flow_to_grid(32, 1) == 1.0F);
  CHECK(flow_to_grid(-64, 2) == -1.0F);
  CHECK(flow_to_grid(16, 4) == 0.125F);
  CHECK(flow_cells(1280, 2) == 640);
  CHECK(flow_cells(1281, 4) == 321);
  Plane<std::int16_t> x(2, 1, 96);
  Plane<std::int16_t> y(2, 1, -32);
  const auto f = ref::decode_flow(x, y, 4);
  REQUIRE(f);
  CHECK(f->x.at(1, 0) == 0.75F);
  CHECK(f->y.at(0, 0) == -0.25F);
  CHECK(ref::decode_flow(x, y, 3).error().code == ErrorCode::ParameterOutOfRange);
  CHECK(ref::decode_flow(x, Plane<std::int16_t>(3, 1), 4).error().code == ErrorCode::SizeMismatch);
}

TEST_CASE("flux optique : une translation donne la moitié vers chaque image, masque 1/2", "[optical_flow]") {
  const auto motion = ref::mid_motion(uniform(16, 8, 3.0F, -1.0F), uniform(16, 8, -3.0F, 1.0F));
  REQUIRE(motion);
  CHECK(motion->area.width == 16);
  CHECK(motion->area.height == 8);
  CHECK_THAT(motion->flow[0].at(5, 4), WithinAbs(-1.5, 1e-6));
  CHECK_THAT(motion->flow[1].at(5, 4), WithinAbs(0.5, 1e-6));
  CHECK_THAT(motion->flow[2].at(5, 4), WithinAbs(1.5, 1e-6));
  CHECK_THAT(motion->flow[3].at(5, 4), WithinAbs(-0.5, 1e-6));
  CHECK_THAT(motion->mask.at(5, 4), WithinAbs(0.5, 1e-6));
}

TEST_CASE("flux optique : le warp place l'image du milieu à mi-chemin", "[optical_flow]") {
  // b est a décalée de 6 pixels vers la droite ; grille 2 : 3 pixels de la grille.
  constexpr int kW = 64;
  constexpr int kH = 16;
  RgbImage a(kW, kH);
  RgbImage b(kW, kH);
  for (std::size_t c = 0; c < 3; ++c) {
    for (int y = 0; y < kH; ++y) {
      for (int x = 0; x < kW; ++x) {
        a.planes[c].at(x, y) = static_cast<float>(x);
        b.planes[c].at(x, y) = static_cast<float>(x - 6);
      }
    }
  }
  const auto motion =
      ref::mid_motion(uniform(kW / 2, kH / 2, 3.0F, 0.0F), uniform(kW / 2, kH / 2, -3.0F, 0.0F));
  REQUIRE(motion);
  const auto middle = ref::warp_blend(a, b, *motion);
  REQUIRE(middle);
  for (int x = 8; x < kW - 8; ++x) CHECK_THAT(middle->planes[1].at(x, kH / 2), WithinAbs(x - 3.0, 1e-4));
}

TEST_CASE("flux optique : un objet qui avance garde son mouvement, le fond découvert vient de b",
          "[optical_flow]") {
  const auto motion = ref::mid_motion(object_forward(), object_backward());
  REQUIRE(motion);
  const auto mid_x = [&](int x) { return motion->flow[2].at(x, 2) * 2.0F; };
  // Au milieu, l'objet est en [10, 18) : il avance de 4, le fond ne bouge pas.
  for (int x = 11; x < 17; ++x) {
    CHECK_THAT(mid_x(x), WithinAbs(4.0, 0.05));
    CHECK_THAT(motion->mask.at(x, 2), WithinAbs(0.5, 0.05));
  }
  CHECK_THAT(mid_x(4), WithinAbs(0.0, 1e-6));
  CHECK_THAT(mid_x(26), WithinAbs(0.0, 1e-6));
  // Fond découvert derrière l'objet (en [8, 10) au milieu) : caché dans a, il vient de b.
  CHECK(mid_x(9) < 0.5F);
  CHECK(motion->mask.at(9, 2) < 0.1F);
  // Fond que l'objet recouvre (en [18, 20) au milieu) : caché dans b, il vient de a.
  CHECK(mid_x(19) < 0.5F);
  CHECK(motion->mask.at(19, 2) > 0.9F);
}

TEST_CASE("flux optique : des flots hors contrat sont refusés", "[optical_flow]") {
  CHECK(ref::mid_motion(uniform(4, 4, 0, 0), uniform(4, 2, 0, 0)).error().code == ErrorCode::SizeMismatch);
  ref::FlowField hollow = uniform(4, 4, 0, 0);
  hollow.y.data.pop_back();
  CHECK(ref::mid_motion(hollow, uniform(4, 4, 0, 0)).error().code == ErrorCode::SizeMismatch);
  CHECK(ref::mid_motion(ref::FlowField{}, uniform(4, 4, 0, 0)).error().code ==
        ErrorCode::DimensionOutOfRange);
}
