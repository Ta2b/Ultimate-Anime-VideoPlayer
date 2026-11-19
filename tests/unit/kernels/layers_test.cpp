#include "kernels/ref/layers.h"

#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/spec/layers.h"

using namespace kirari::kernels;
using Catch::Matchers::WithinAbs;

namespace {

// Un décor texturé : sa valeur ne dépend que de x (un panoramique horizontal le décale exactement).
float decor(int x) {
  return 0.5F + 0.4F * std::sin(0.3F * static_cast<float>(x));
}

RgbImage panned(int w, int h, int offset) {
  RgbImage img(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (auto& p : img.planes) p.at(x, y) = decor(x - offset);
    }
  }
  return img;
}

LayerGrid full_grid(int cols, int rows, float tile) {
  LayerGrid g{.cols = cols, .rows = rows, .tile_w = tile, .tile_h = tile};
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) g.set(c, r);
  }
  return g;
}

}  // namespace

// Un seul test pour tout le contrat : déplacements exacts, composition d'un panoramique, dessin et garde qui
// gardent l'image de RIFE, plaque à une source.
TEST_CASE("calques : déplacements de caméra et composition de référence", "[layers]") {
  // Caméra : 8 pixels de travail vers la droite, travail 64 × 32 agrandi ×2 : 16 pixels de sortie.
  const Affine camera{8.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const Affine out_camera = to_output(camera, 64, 32, 128, 64);
  CHECK_THAT(out_camera[0], WithinAbs(16.0, 1e-9));
  CHECK_THAT(out_camera[1], WithinAbs(1.0, 1e-12));
  const auto shifts = layer_shifts(out_camera, 0.5);
  REQUIRE(shifts);
  CHECK_THAT(shifts->a[0], WithinAbs(-8.0, 1e-9));
  CHECK_THAT(shifts->b[0], WithinAbs(8.0, 1e-9));
  // Zoom et rotation : inverse puis composition rendent l'identité ; une partie linéaire écrasée ne s'inverse
  // pas.
  const Affine zoom{3.0, 1.02, 0.01, -2.0, -0.01, 1.02};
  const auto back = inverse(zoom);
  REQUIRE(back);
  const Affine id = compose(zoom, *back);
  for (std::size_t i = 0; i < 6; ++i) CHECK_THAT(id[i], WithinAbs(kIdentityAffine[i], 1e-12));
  CHECK_FALSE(inverse(Affine{0.0, 0.1, 0.0, 0.0, 0.0, 0.1}));
  const Affine zoom_out = to_output(zoom, 64, 32, 128, 64);
  CHECK_THAT(determinant(zoom_out), WithinAbs(determinant(zoom), 1e-9));

  // Panoramique : b est a décalée de 16 pixels ; à t = ½ le décor est a décalée de 8.
  const int w = 128;
  const int h = 64;
  const RgbImage a = panned(w, h, 0);
  const RgbImage b = panned(w, h, 16);
  LayerParams params{.grid = full_grid(8, 4, 16.0F), .count = 2};
  params.sources[0] = {.shift = to_float(shifts->a), .weight = 0.5F};
  params.sources[1] = {.shift = to_float(shifts->b), .weight = 0.5F};
  RgbImage out(w, h, 0.25F);  // l'image de RIFE, fausse partout
  REQUIRE(ref::compose_layers(out, a, b, params));
  for (const int x : {20, 64, 100}) {
    CHECK_THAT(out.planes[1].at(x, 30), WithinAbs(decor(x - 8), 1e-5));
  }
  // Au bord gauche, la source a sort de l'image : b seule rend le décor qui entre.
  CHECK_THAT(out.planes[0].at(3, 30), WithinAbs(decor(3 - 8), 1e-5));

  // Une tuile dessin (colonne 3, ligne 1) : son cœur et la moitié de ses voisines gardent RIFE.
  LayerParams drawing = params;
  drawing.grid = LayerGrid{.cols = 8, .rows = 4, .tile_w = 16.0F, .tile_h = 16.0F};
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 8; ++c) {
      if (c != 3 || r != 1) drawing.grid.set(c, r);
    }
  }
  CHECK(ref::layer_mask(drawing.grid, 3 * 16 + 7.5F, 16 + 7.5F) == 0.0F);
  CHECK(ref::layer_mask(drawing.grid, 2 * 16 + 7.5F, 16 + 7.5F) == 1.0F);
  RgbImage kept(w, h, 0.25F);
  REQUIRE(ref::compose_layers(kept, a, b, drawing));
  CHECK(kept.planes[0].at(3 * 16 + 8, 16 + 8) == 0.25F);
  CHECK_THAT(kept.planes[0].at(100, 50), WithinAbs(decor(100 - 8), 1e-5));

  // Garde : là où b ne suit pas la caméra (un éclair blanc), l'image de RIFE reste.
  RgbImage flash = b;
  for (int y = 40; y < 50; ++y) {
    for (int x = 70; x < 90; ++x) {
      for (auto& p : flash.planes) p.at(x, y) = 1.0F;
    }
  }
  RgbImage guarded(w, h, 0.25F);
  REQUIRE(ref::compose_layers(guarded, a, flash, params));
  CHECK(guarded.planes[2].at(72, 45) == 0.25F);

  // Plaque : une source, déplacement nul ; la garde compare à l'image en place.
  LayerParams plate{
      .grid = full_grid(8, 4, 16.0F), .count = 1, .guard_low = kPlateGuardLow, .guard_high = kPlateGuardHigh};
  RgbImage near = a;
  for (auto& p : near.planes) {
    for (float& v : p.data) v += 2.0F / 255.0F;
  }
  REQUIRE(ref::compose_layers(near, a, a, plate));
  CHECK_THAT(near.planes[0].at(64, 32), WithinAbs(decor(64), 1e-6));
  RgbImage far(w, h, 0.0F);  // trop loin de la plaque : rien ne change
  REQUIRE(ref::compose_layers(far, a, a, plate));
  CHECK(far.planes[0].at(64, 32) == 0.0F);

  // Refus : grille vide, garde vide.
  LayerParams empty = params;
  empty.grid.cols = 0;
  CHECK_FALSE(ref::compose_layers(out, a, b, empty));
  LayerParams flat = params;
  flat.guard_high = flat.guard_low;
  CHECK_FALSE(ref::compose_layers(out, a, b, flat));
}
