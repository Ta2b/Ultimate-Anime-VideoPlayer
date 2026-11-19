// L'affinage de la caméra de la carte (kernels/spec/scene_camera.h) : sur une image lisse déplacée par une
// affine connue, la caméra ajustée sur des vecteurs biaisés (comme NVOF sur un panoramique lent) est
// ramenée à la vraie par Lucas-Kanade, et l'écart mesuré sur l'image le dit ; un dessin qui bouge hors de la
// caméra sans que NVOF le voie (une bouche, des yeux) est dessin par l'écart après la caméra, même là où NVOF
// apparie mal ; une lueur qui s'allume sur place est un effet, une tache nette qui apparaît ne l'est pas ; un
// aplat ne s'affine pas ; trop peu de décor, pas de demande. Un seul cas, comme le veut la consigne des tests
// groupés.
#include "kernels/ref/scene_camera.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
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

// Le dessin : 2 × 2 tuiles, décalées de kPatchShift pixels de plus que la caméra dans b.
constexpr int kPatchX0 = 96;
constexpr int kPatchY0 = 48;
constexpr int kPatchSide = 2 * kSceneTileFlowPixels;
constexpr double kPatchShift = 4.0;

bool in_patch(int i, int j) {
  return i >= kPatchX0 && i < kPatchX0 + kPatchSide && j >= kPatchY0 && j < kPatchY0 + kPatchSide;
}

// Ce qui change dans le carré du dessin : il bouge (Moved), une lueur douce s'y allume (Glow : cloche en
// cosinus de kGlowCodes sur le rayon du carré), une tache nette y apparaît (Spot : kGlowCodes de plus sur un
// disque de kSpotRadius pixels, bord franc).
enum class Patch : std::uint8_t { None, Moved, Glow, Spot };
constexpr double kGlowCodes = 50.0;
constexpr double kSpotRadius = 10.0;

double lit(Patch patch, int i, int j) {
  const double r =
      std::hypot(i + 0.5 - (kPatchX0 + 0.5 * kPatchSide), j + 0.5 - (kPatchY0 + 0.5 * kPatchSide));
  const double radius = 0.5 * kPatchSide;
  if (patch == Patch::Glow && r < radius)
    return kGlowCodes * 0.5 * (1.0 + std::cos(std::numbers::pi * r / radius));
  if (patch == Patch::Spot && r < kSpotRadius) return kGlowCodes;
  return 0.0;
}

// a, et b : le point x de a en x + d(x) dans b, d du modèle p (l'affine inversée exactement) ; dans le carré
// du dessin, ce que dit patch.
Pair moved(const std::array<double, 6>& p, bool flat = false, Patch patch = Patch::None) {
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
      const double extra = patch == Patch::Moved && in_patch(i, j) ? kPatchShift : 0.0;
      out.b.at(i, j) = flat ? std::uint8_t{90} : code(scene(sx - extra, sy) + lit(patch, i, j));
    }
  }
  return out;
}

// Les tuiles de la paire, avec les vecteurs du modèle seen (celui que NVOF verrait) au centre de chaque bloc,
// de coût 10, patch_cost dans le carré du dessin.
SceneFlowTiles tiles_of(const Pair& pair, const std::array<double, 6>& seen, std::uint8_t patch_cost = 10) {
  const int cw = scene_flow_cells(kWidth);
  const int ch = scene_flow_cells(kHeight);
  Plane<ref::FlowVector> flow(cw, ch);
  Plane<std::uint8_t> cost(cw, ch, 10);
  for (int j = 0; j < ch; ++j) {
    for (int i = 0; i < cw; ++i) {
      if (in_patch(i * kSceneFlowGrid, j * kSceneFlowGrid)) cost.at(i, j) = patch_cost;
      const auto d = shift(seen, i * kSceneFlowGrid + 2.0, j * kSceneFlowGrid + 2.0);
      flow.at(i, j) = {.x = static_cast<std::int16_t>(std::lround(d[0] * 32.0)),
                       .y = static_cast<std::int16_t>(std::lround(d[1] * 32.0))};
    }
  }
  auto tiles = ref::scene_tile_sums(pair.a, pair.b, flow, cost, kDivisor);
  REQUIRE(tiles);
  return *tiles;
}

// La vraie caméra (coordonnées réduites).
constexpr std::array<double, 6> kTruth{2.3, 1.0, 0.5, -1.1, -0.4, 1.0};

// La carte d'une paire où NVOF ne voit que la caméra, après l'affinage et sa vérification.
struct Mapped {
  SceneFlowTiles tiles;
  ref::SceneCamera camera;
  SceneMotion motion;
  const SceneTileCheck& check(int col, int row) const {
    return camera.checks[static_cast<std::size_t>(row * tiles.cols + col)];
  }
};
Mapped mapped(const Pair& pair, std::uint8_t patch_cost = 10) {
  Mapped out;
  out.tiles = tiles_of(pair, kTruth, patch_cost);
  out.camera = ref::scene_camera_of(out.tiles);
  REQUIRE(out.camera.fit);
  const auto request = ref::scene_camera_request(out.camera);
  REQUIRE(request);
  const auto refined = ref::scene_camera_refine(pair.a, pair.b, *request);
  REQUIRE(refined);
  ref::scene_camera_apply(out.camera, *refined);
  REQUIRE(out.camera.checks.size() == out.tiles.tiles.size());
  ref::SceneHistory history;
  out.motion = ref::scene_motion_of(out.tiles, history, out.camera);
  return out;
}

// L'accord des pixels qui bougent : |éclaircis − assombris| / qui bougent.
float agree(const SceneTileCheck& t) {
  return t.moving > 0.0F ? std::abs(2.0F * t.brighter - t.moving) / t.moving : 0.0F;
}

}  // namespace

TEST_CASE("affinage de la caméra de la carte", "[scene_flow][scene_camera]") {
  // Une translation, un zoom de 1/128 et un cisaillement ; NVOF voit en plus un cisaillement inventé.
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

  SECTION("un dessin que NVOF ne voit pas bouge après la caméra : dessin ; le décor reste décor") {
    const Pair pair = moved(kTruth, false, Patch::Moved);
    const SceneFlowTiles tiles = tiles_of(pair, kTruth);  // NVOF n'y voit que la caméra
    ref::SceneCamera camera = ref::scene_camera_of(tiles);
    REQUIRE(camera.fit);
    ref::SceneHistory before;
    const SceneMotion vectors = ref::scene_motion_of(tiles, before, camera);
    const auto request = ref::scene_camera_request(camera);
    REQUIRE(request);
    const auto refined = ref::scene_camera_refine(pair.a, pair.b, *request);
    REQUIRE(refined);
    ref::scene_camera_apply(camera, *refined);
    REQUIRE(camera.checks.size() == tiles.tiles.size());
    ref::SceneHistory history;
    const SceneMotion motion = ref::scene_motion_of(tiles, history, camera);
    const int col0 = kPatchX0 / kSceneTileFlowPixels;
    const int row0 = kPatchY0 / kSceneTileFlowPixels;
    for (int row = row0; row < row0 + 2; ++row) {
      for (int col = col0; col < col0 + 2; ++col) {
        CHECK(vectors.at(col, row).kind == SceneClass::Background);  // les vecteurs seuls : décor
        const SceneTileCheck& t = camera.checks[static_cast<std::size_t>(row * tiles.cols + col)];
        CHECK(t.moving >= scene::kMovingShare);
        CHECK(motion.at(col, row).kind == SceneClass::Drawing);
        CHECK(motion.at(col, row).confidence >= 0.5F);
        CHECK(motion.at(col, row).changed);
      }
    }
    // Loin du dessin (à deux tuiles au moins), tout est décor, sûr, et ses pixels ne bougent pas.
    for (int row = 1; row + 1 < tiles.rows; ++row) {
      for (int col = 1; col + 1 < tiles.cols; ++col) {
        if (col >= col0 - 2 && col < col0 + 4 && row >= row0 - 2 && row < row0 + 4) continue;
        const SceneTileCheck& t = camera.checks[static_cast<std::size_t>(row * tiles.cols + col)];
        CHECK(t.moving < 0.02F);
        CHECK(t.error < 1.0F);
        CHECK(motion.at(col, row).kind == SceneClass::Background);
        CHECK(motion.at(col, row).confidence >= 0.9F);
      }
    }
  }

  SECTION("une lueur qui s'allume sur place : effet ; une tache nette qui apparaît : dessin") {
    const int col0 = kPatchX0 / kSceneTileFlowPixels;
    const int row0 = kPatchY0 / kSceneTileFlowPixels;
    const Mapped glow = mapped(moved(kTruth, false, Patch::Glow));
    const Mapped spot = mapped(moved(kTruth, false, Patch::Spot));
    for (int row = row0; row < row0 + 2; ++row) {
      for (int col = col0; col < col0 + 2; ++col) {
        CAPTURE(col, row);
        const SceneTileCheck& g = glow.check(col, row);
        CHECK(g.moving >= scene::kMovingShare);
        CHECK(g.brighter == g.moving);
        CHECK(g.sharp == 0.0F);
        CHECK(g.settled >= scene::kLightKept * g.error);  // aucun recalage ne l'explique
        CHECK(glow.motion.at(col, row).kind == SceneClass::Effect);
        CHECK(glow.motion.at(col, row).confidence >= 0.99F);
        CHECK(glow.motion.at(col, row).changed);
        const SceneTileCheck& s = spot.check(col, row);
        CHECK(s.moving >= scene::kMovingShare);
        CHECK(agree(s) >= scene::kLightAgree);  // d'un seul sens aussi, mais d'un bord net
        CHECK(s.sharp > scene::kLightSharp * s.moving);
        CHECK(spot.motion.at(col, row).kind == SceneClass::Drawing);
      }
    }
    // Ailleurs, rien ne change : décor.
    CHECK(glow.motion.at(1, 1).kind == SceneClass::Background);
    CHECK(spot.motion.at(1, 1).kind == SceneClass::Background);
  }

  SECTION("un trait qui bouge où NVOF apparie mal : dessin, jamais effet ; une lueur reste un effet") {
    const int col0 = kPatchX0 / kSceneTileFlowPixels;
    const int row0 = kPatchY0 / kSceneTileFlowPixels;
    // Coût de 16 : NVOF y apparie (sous kSceneCostHigh), mais cher (au-dessus de kEffectCost) ; de 250 : sans
    // correspondance.
    const Pair pair = moved(kTruth, false, Patch::Moved);
    const Mapped costly = mapped(pair, 16);
    const Mapped lost = mapped(pair, 250);
    const Mapped glow = mapped(moved(kTruth, false, Patch::Glow), 16);
    int sharp = 0;  // le trait se voit sur une tuile au moins : ses voisines en héritent (near_sharp)
    for (int row = row0; row < row0 + 2; ++row) {
      for (int col = col0; col < col0 + 2; ++col) {
        CAPTURE(col, row);
        const SceneTileCheck& t = costly.check(col, row);
        CHECK(t.moving >= scene::kEffectShare);
        sharp += t.sharp > scene::kLightSharp * t.moving ? 1 : 0;
        const SceneTileSums& sums =
            costly.tiles.tiles[static_cast<std::size_t>(row * costly.tiles.cols + col)];
        CHECK(sums.cost >= scene::kEffectCost * sums.cells);
        CHECK(costly.motion.at(col, row).kind == SceneClass::Drawing);
        CHECK(lost.motion.at(col, row).kind == SceneClass::Drawing);
        CHECK(glow.motion.at(col, row).kind == SceneClass::Effect);
      }
    }
    CHECK(sharp >= 1);
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
