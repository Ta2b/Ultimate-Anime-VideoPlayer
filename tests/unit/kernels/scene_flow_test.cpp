// La carte de la scène (kernels/spec/scene_flow.h) sur des flux synthétiques : caméra seule, dessin qui
// bouge sur un décor qui défile, aplat, coupe, fondu, image tenue, et le lissage des classes d'une paire à
// l'autre (animation en deux comprise, effet tenu sur une image fixe).
// Un seul cas, comme le veut la consigne des tests groupés.
#include "kernels/ref/scene_flow.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace kirari::kernels;
using Catch::Matchers::WithinAbs;

namespace {

constexpr int kWidth = 256;  // image du flux : 64 × 36 vecteurs, 16 × 9 tuiles
constexpr int kHeight = 144;
constexpr int kDivisor = 2;  // travail 512 × 288

std::uint8_t texture(int x, int y) {
  return static_cast<std::uint8_t>(20 + (x * 37 + y * 91 + (x * y) % 23 * 7) % 200);
}

Plane<std::uint8_t> textured(int shift_x = 0, int shift_y = 0) {
  Plane<std::uint8_t> p(kWidth, kHeight);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) p.at(x, y) = texture(x - shift_x, y - shift_y);
  }
  return p;
}

struct Pair {
  Plane<std::uint8_t> a = textured();
  Plane<std::uint8_t> b = textured();
  Plane<ref::FlowVector> flow{scene_flow_cells(kWidth), scene_flow_cells(kHeight)};
  Plane<std::uint8_t> cost{scene_flow_cells(kWidth), scene_flow_cells(kHeight), 20};

  // Le flux en pixels du flux, sur les vecteurs [x0, x1) × [y0, y1).
  void move(float dx, float dy, int x0 = 0, int y0 = 0, int x1 = 1 << 20, int y1 = 1 << 20) {
    for (int j = y0; j < std::min(y1, flow.height); ++j) {
      for (int i = x0; i < std::min(x1, flow.width); ++i) {
        flow.at(i, j) = {.x = static_cast<std::int16_t>(dx * 32.0F),
                         .y = static_cast<std::int16_t>(dy * 32.0F)};
      }
    }
  }

  // Vecteurs [x0, x1) × [y0, y1) sans correspondance, et leurs pixels changés dans b (un effet).
  void unmatch(int x0, int y0, int x1, int y1) {
    for (int j = y0; j < y1; ++j) {
      for (int i = x0; i < x1; ++i) cost.at(i, j) = 250;
    }
    for (int y = y0 * kSceneFlowGrid; y < y1 * kSceneFlowGrid; ++y) {
      for (int x = x0 * kSceneFlowGrid; x < x1 * kSceneFlowGrid; ++x)
        b.at(x, y) = static_cast<std::uint8_t>(255 - b.at(x, y));
    }
  }

  SceneMotion map(ref::SceneHistory& history) const {
    const auto tiles = ref::scene_tile_sums(a, b, flow, cost, kDivisor);
    REQUIRE(tiles);
    return ref::scene_motion_of(*tiles, history);
  }
};

int count(const SceneMotion& m, SceneClass kind) {
  int n = 0;
  for (const SceneTile& t : m.tiles) n += t.kind == kind ? 1 : 0;
  return n;
}

}  // namespace

TEST_CASE("carte de la scène sur des flux synthétiques", "[scene_flow]") {
  ref::SceneHistory history;

  SECTION("la luma du flux et la grille") {
    RgbImage rgb(kWidth * kDivisor, kHeight * kDivisor, 0.5F);
    const auto luma = ref::scene_luma(rgb, kDivisor);
    REQUIRE(luma);
    CHECK(luma->width == kWidth);
    CHECK(luma->height == kHeight);
    CHECK(luma->at(10, 10) == 128);
    CHECK_FALSE(ref::scene_luma(RgbImage(60, 60, 0.0F), 2));
    CHECK(scene_flow_divisor(1080) == 2);
    CHECK(scene_flow_divisor(720) == 1);
    CHECK(scene_flow_divisor(2160) == 3);
    const auto tiles = ref::scene_tile_sums(textured(), textured(), Pair{}.flow, Pair{}.cost, kDivisor);
    REQUIRE(tiles);
    CHECK(tiles->cols == 16);
    CHECK(tiles->rows == 9);
    CHECK(tiles->tiles[0].cells == 16.0F);
    CHECK(tiles->tiles[0].pixels == 256.0F);
  }

  SECTION("image tenue : rien ne bouge") {
    const Pair pair;
    const SceneMotion m = pair.map(history);
    CHECK(m.camera_fit);
    CHECK(m.background_held);
    CHECK(m.drawing_held);
    CHECK_FALSE(m.cut_likely);
    CHECK_FALSE(m.fade);
    CHECK(m.tile_px == 32);
    for (const SceneTile& t : m.tiles) CHECK_FALSE(t.changed);
  }

  SECTION("panoramique : la caméra seule, en pixels de travail") {
    Pair pair;
    pair.b = textured(3, -1);
    pair.move(3.0F, -1.0F);
    const SceneMotion m = pair.map(history);
    REQUIRE(m.camera_fit);
    CHECK_THAT(m.camera[0], WithinAbs(6.0, 1e-3));
    CHECK_THAT(m.camera[3], WithinAbs(-2.0, 1e-3));
    CHECK_THAT(m.camera[1], WithinAbs(1.0, 1e-6));
    CHECK_THAT(m.camera[5], WithinAbs(1.0, 1e-6));
    CHECK(m.residual < 0.01);
    CHECK_FALSE(m.background_held);
    CHECK(m.drawing_held);  // le dessin suit la caméra
    CHECK(count(m, SceneClass::Background) == static_cast<int>(m.tiles.size()));
    CHECK_THAT(m.at(5, 5).dx, WithinAbs(6.0, 1e-3));
    CHECK(m.at(5, 5).changed);
  }

  SECTION("un dessin qui bouge sur le décor qui défile") {
    Pair pair;
    pair.b = textured(2, 0);
    pair.move(2.0F, 0.0F);
    pair.move(10.0F, 5.0F, 16, 8, 32, 20);  // tuiles 4 à 7, 2 à 4
    const SceneMotion m = pair.map(history);
    REQUIRE(m.camera_fit);
    CHECK_THAT(m.camera[0], WithinAbs(4.0, 1e-3));
    CHECK(m.at(5, 3).kind == SceneClass::Drawing);
    CHECK(m.at(0, 0).kind == SceneClass::Background);
    CHECK(m.at(12, 7).kind == SceneClass::Background);
    CHECK(count(m, SceneClass::Drawing) == 12);
    CHECK_FALSE(m.drawing_held);
  }

  SECTION("un effet sur un dessin tenu : ce n'est pas tenu") {
    Pair pair;
    pair.unmatch(16, 8, 32, 20);  // tuiles 4 à 7, 2 à 4
    const SceneMotion m = pair.map(history);
    REQUIRE(m.camera_fit);
    CHECK(m.background_held);
    CHECK(m.at(5, 3).kind == SceneClass::Effect);
    CHECK_FALSE(m.cut_likely);
    CHECK_FALSE(m.drawing_held);
  }

  SECTION("sans caméra ajustée (aplat), ce qui bouge n'est pas tenu") {
    Pair pair;
    for (std::uint8_t& v : pair.a.data) v = 128;
    pair.b = pair.a;
    const SceneMotion still = pair.map(history);
    CHECK_FALSE(still.camera_fit);
    CHECK(still.drawing_held);  // rien n'a changé
    pair.move(3.0F, 0.0F, 16, 8, 32, 20);
    const SceneMotion moved = pair.map(history);
    CHECK_FALSE(moved.camera_fit);
    CHECK(moved.at(5, 3).changed);
    CHECK_FALSE(moved.drawing_held);
  }

  SECTION("coupe : rien ne correspond") {
    Pair pair;
    for (std::uint8_t& v : pair.b.data) v = static_cast<std::uint8_t>(255 - v);
    for (std::uint8_t& c : pair.cost.data) c = 250;
    history.cols = 16;  // une coupe vide l'historique
    const SceneMotion m = pair.map(history);
    CHECK(m.cut_likely);
    CHECK_FALSE(m.fade);
    CHECK_FALSE(m.camera_fit);
    CHECK_FALSE(m.drawing_held);
    CHECK(history.cols == 0);
    CHECK(count(m, SceneClass::Effect) == static_cast<int>(m.tiles.size()));
  }

  SECTION("fondu : la luminosité glisse partout, ce n'est pas une coupe") {
    Pair pair;
    for (std::uint8_t& v : pair.b.data) v = static_cast<std::uint8_t>(v + 20);
    for (std::uint8_t& c : pair.cost.data) c = 200;  // le coût monte partout
    const SceneMotion m = pair.map(history);
    CHECK(m.fade);
    CHECK_FALSE(m.cut_likely);
  }

  SECTION("une tuile ne change de classe que vue deux paires de suite") {
    Pair still;
    still.b = textured(2, 0);
    still.move(2.0F, 0.0F);
    Pair drawn = still;
    drawn.move(10.0F, 5.0F, 16, 8, 32, 20);
    CHECK(still.map(history).at(5, 3).kind == SceneClass::Background);
    const SceneMotion once = drawn.map(history);
    CHECK(once.at(5, 3).kind == SceneClass::Background);
    CHECK(once.at(5, 3).confidence < 0.5F);  // contredite : plus sûre pour les clients
    CHECK(once.at(5, 3).changed);            // brut, non lissé
    CHECK(drawn.map(history).at(5, 3).kind == SceneClass::Drawing);
  }

  SECTION("animation en deux : l'image doublée ne compte pas, le dessin reste dessin") {
    const Pair held;
    Pair moved;
    moved.move(10.0F, 5.0F, 16, 8, 32, 20);
    CHECK(held.map(history).at(5, 3).kind == SceneClass::Background);
    CHECK(moved.map(history).at(5, 3).kind == SceneClass::Background);
    CHECK(held.map(history).at(5, 3).kind == SceneClass::Background);
    CHECK(moved.map(history).at(5, 3).kind == SceneClass::Drawing);
    const SceneMotion again = held.map(history);
    CHECK(again.at(5, 3).kind == SceneClass::Drawing);
    CHECK_FALSE(again.at(5, 3).changed);
    CHECK(again.drawing_held);  // brut : rien ne bouge à cette paire
  }

  SECTION("un effet ne tient que deux images fixes, puis la tuile reprend sa classe brute") {
    Pair lit;
    lit.unmatch(16, 8, 32, 20);
    const Pair held;
    CHECK(lit.map(history).at(5, 3).kind == SceneClass::Effect);
    for (int k = 0; k < scene::kEffectIdlePairs; ++k) {
      CAPTURE(k);
      const SceneMotion still = held.map(history);
      CHECK_FALSE(still.at(5, 3).changed);
      CHECK(still.at(5, 3).kind == SceneClass::Effect);  // une animation en trois : l'effet tient
    }
    CHECK(held.map(history).at(5, 3).kind == SceneClass::Background);
    CHECK(lit.map(history).at(5, 3).kind == SceneClass::Effect);  // vu deux paires de suite
  }

  SECTION("aplat : décor s'il suit la caméra et ne change pas, sinon incertain") {
    Pair pair;
    pair.b = textured(2, 0);
    pair.move(2.0F, 0.0F);
    for (int y = 32; y < 80; ++y) {
      for (int x = 64; x < 128; ++x) pair.a.at(x, y) = pair.b.at(x, y) = 128;  // tuiles 4 à 7, 2 à 4
    }
    ref::SceneHistory fresh;
    const SceneMotion follows = pair.map(fresh);
    REQUIRE(follows.camera_fit);
    CHECK(follows.at(5, 3).kind == SceneClass::Background);
    CHECK(follows.at(5, 3).confidence >= 0.5F);
    pair.move(-6.0F, 0.0F, 16, 8, 32, 20);
    fresh.clear();
    CHECK(pair.map(fresh).at(5, 3).kind == SceneClass::Uncertain);  // un vecteur d'aplat ne prouve rien
    pair.move(2.0F, 0.0F);
    for (int y = 32; y < 80; ++y) {
      for (int x = 64; x < 128; ++x) pair.b.at(x, y) = 160;
    }
    fresh.clear();
    CHECK(pair.map(fresh).at(5, 3).kind == SceneClass::Uncertain);  // l'aplat change
  }
}
