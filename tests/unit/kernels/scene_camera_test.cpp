// L'affinage de la caméra de la carte (kernels/spec/scene_camera.h) : sur une image lisse déplacée par une
// affine connue, la caméra ajustée sur des vecteurs biaisés (comme NVOF sur un panoramique lent) est
// ramenée à la vraie par Lucas-Kanade, et l'écart mesuré sur l'image le dit ; un aplat ne s'affine pas ; trop
// peu de décor, pas de demande.
// Un seul cas, comme le veut la consigne des tests groupés.
#include "kernels/ref/scene_camera.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/ref/scene_flow.h"

using namespace kirari::kernels;
using Catch::Matchers::WithinAbs;

namespace {

constexpr int kWidth = 256;  // 16 × 9 tuiles
constexpr int kHeight = 144;
constexpr int kDivisor = 2;

// Une image lisse, texturée dans les deux sens, au point continu (x, y).
double scene(double x, double y) {
  return 128.0 + 40.0 * std::sin(0.21 * x + 0.05 * y) + 30.0 * std::sin(0.13 * y - 0.07 * x) +
         20.0 * std::sin(0.31 * x + 0.27 * y);
}

std::uint8_t code(double v) {
  return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 255.0)));
}

// Le déplacement du modèle p (coordonnées réduites) au point continu (x, y).
std::array<double, 2> shift(const std::array<double, 6>& p, double x, double y) {
  const SceneCameraFrame f = scene_camera_frame(kWidth, kHeight);
  const double rx = (x - f.cx) * f.inv_scale;
  const double ry = (y - f.cy) * f.inv_scale;
  return {p[0] + p[1] * rx + p[2] * ry, p[3] + p[4] * rx + p[5] * ry};
}

struct Pair {
  Plane<std::uint8_t> a{kWidth, kHeight};
  Plane<std::uint8_t> b{kWidth, kHeight};
};

// a, et b : le point x de a en x + d(x) dans b, d du modèle p (l'affine inversée exactement).
Pair moved(const std::array<double, 6>& p, bool flat = false) {
  const SceneCameraFrame f = scene_camera_frame(kWidth, kHeight);
  const double s = f.inv_scale;
  const double m00 = 1.0 + p[1] * s;
  const double m01 = p[2] * s;
  const double m10 = p[4] * s;
  const double m11 = 1.0 + p[5] * s;
  const double ox = p[0] - (p[1] * f.cx + p[2] * f.cy) * s;
  const double oy = p[3] - (p[4] * f.cx + p[5] * f.cy) * s;
  const double det = m00 * m11 - m01 * m10;
  Pair out;
  for (int j = 0; j < kHeight; ++j) {
    for (int i = 0; i < kWidth; ++i) {
      const double x = i + 0.5;
      const double y = j + 0.5;
      const double ux = x - ox;
      const double uy = y - oy;
      const double sx = (m11 * ux - m01 * uy) / det;
      const double sy = (-m10 * ux + m00 * uy) / det;
      out.a.at(i, j) = flat ? std::uint8_t{90} : code(scene(x, y));
      out.b.at(i, j) = flat ? std::uint8_t{90} : code(scene(sx, sy));
    }
  }
  return out;
}

// Les tuiles de la paire, avec les vecteurs du modèle seen (celui que NVOF verrait) au centre de chaque bloc.
SceneFlowTiles tiles_of(const Pair& pair, const std::array<double, 6>& seen) {
  const int cw = scene_flow_cells(kWidth);
  const int ch = scene_flow_cells(kHeight);
  Plane<ref::FlowVector> flow(cw, ch);
  Plane<std::uint8_t> cost(cw, ch, 10);
  for (int j = 0; j < ch; ++j) {
    for (int i = 0; i < cw; ++i) {
      const auto d = shift(seen, i * kSceneFlowGrid + 2.0, j * kSceneFlowGrid + 2.0);
      flow.at(i, j) = {.x = static_cast<std::int16_t>(std::lround(d[0] * 32.0)),
                       .y = static_cast<std::int16_t>(std::lround(d[1] * 32.0))};
    }
  }
  auto tiles = ref::scene_tile_sums(pair.a, pair.b, flow, cost, kDivisor);
  REQUIRE(tiles);
  return *tiles;
}

}  // namespace

TEST_CASE("affinage de la caméra de la carte", "[scene_flow][scene_camera]") {
  // Une translation, un zoom de 1/128 et un cisaillement ; NVOF voit en plus un cisaillement inventé.
  constexpr std::array<double, 6> kTruth{2.3, 1.0, 0.5, -1.1, -0.4, 1.0};
  constexpr std::array<double, 6> kNvofBias{0.3, 0.0, 1.2, -0.2, 0.0, 0.0};

  SECTION("le cisaillement inventé par NVOF est retiré, l'écart mesuré sur l'image le dit") {
    std::array<double, 6> seen = kTruth;
    for (std::size_t k = 0; k < 6; ++k) seen[k] += kNvofBias[k];
    const Pair pair = moved(kTruth);
    const SceneFlowTiles tiles = tiles_of(pair, seen);
    ref::SceneCamera camera = ref::scene_camera_of(tiles);
    REQUIRE(camera.fit);
    CHECK_THAT(camera.model[2], WithinAbs(seen[2], 0.1));  // l'ajustement suit les vecteurs, biais compris
    const std::optional<SceneCameraRequest> request = ref::scene_camera_request(camera);
    REQUIRE(request);
    const auto refined = ref::scene_camera_refine(pair.a, pair.b, *request);
    REQUIRE(refined);
    CHECK(refined->steps == 2 * scene::kRefineSteps);
    for (std::size_t k = 0; k < 6; ++k) CHECK_THAT(refined->model[k], WithinAbs(kTruth[k], 0.05));
    ref::scene_camera_apply(camera, *refined);
    CHECK(camera.refined);
    CHECK(camera.check_tiles >= scene::kCheckMinTiles);
    CHECK(camera.check < 0.05);

    ref::SceneHistory history;
    const SceneMotion motion = ref::scene_motion_of(tiles, history, camera);
    CHECK(motion.camera_fit);
    CHECK(motion.camera_refined);
    CHECK(motion.residual_tiles == camera.check_tiles);
    CHECK_THAT(motion.residual, WithinAbs(camera.check * kDivisor, 1e-9));
  }

  SECTION("sans affinage, l'écart est celui des vecteurs") {
    std::array<double, 6> seen = kTruth;
    for (std::size_t k = 0; k < 6; ++k) seen[k] += kNvofBias[k];
    const Pair pair = moved(kTruth);
    const SceneFlowTiles tiles = tiles_of(pair, seen);
    const ref::SceneCamera camera = ref::scene_camera_of(tiles);
    ref::SceneHistory history;
    const SceneMotion plain = ref::scene_motion_of(tiles, history);
    CHECK_FALSE(plain.camera_refined);
    CHECK(plain.residual_tiles == 0);
    CHECK_THAT(plain.residual, WithinAbs(camera.residual * kDivisor, 1e-9));
  }

  SECTION("un aplat ne s'affine pas : rien à suivre") {
    const Pair pair = moved(kTruth, true);
    SceneCameraRequest request{.model = kTruth, .decor = {}};
    request.decor.assign(static_cast<std::size_t>(scene_flow_tiles(kWidth) * scene_flow_tiles(kHeight)), 1U);
    const auto refined = ref::scene_camera_refine(pair.a, pair.b, request);
    REQUIRE(refined);
    CHECK(refined->steps == 0);
    CHECK(refined->model == kTruth);
    ref::SceneCamera camera{.fit = true, .model = kTruth};
    ref::scene_camera_apply(camera, *refined);
    CHECK_FALSE(camera.refined);
    CHECK(camera.check < 0.0);
  }

  SECTION("trop peu de décor : pas de demande ; des tuiles d'une autre grille : refus") {
    ref::SceneCamera camera{.fit = true, .model = kTruth};
    camera.decor.assign(16 * 9, 0U);
    camera.decor[40] = 1U;
    camera.decor_pixels = kSceneTileFlowPixels * kSceneTileFlowPixels;
    CHECK_FALSE(ref::scene_camera_request(camera));
    camera.fit = false;
    camera.decor_pixels = 100000;
    CHECK_FALSE(ref::scene_camera_request(camera));
    const Pair pair = moved(kTruth);
    const SceneCameraRequest wrong{.model = kTruth, .decor = std::vector<std::uint8_t>(10, 1U)};
    CHECK_FALSE(ref::scene_camera_refine(pair.a, pair.b, wrong));
  }
}
