#include "kernels/ref/motion_doubt.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kernels/spec/motion_doubt.h"

using namespace kirari::kernels;
using Catch::Matchers::WithinAbs;

namespace {

// Plans 12 × 8, fenêtre 8 × 4 en (2, 2) ; déplacements 4 et masque 0,9 dedans, 99 dehors.
ref::Motion moving() {
  const Window area{.x = 2, .y = 2, .width = 8, .height = 4};
  ref::Motion m{
      .flow = {PlaneF(12, 8, 99.0F), PlaneF(12, 8, 99.0F), PlaneF(12, 8, 99.0F), PlaneF(12, 8, 99.0F)},
      .mask = PlaneF(12, 8, 99.0F),
      .area = area};
  for (int y = area.y; y < area.y + area.height; ++y) {
    for (int x = area.x; x < area.x + area.width; ++x) {
      for (PlaneF& p : m.flow) p.at(x, y) = 4.0F;
      m.mask.at(x, y) = 0.9F;
    }
  }
  return m;
}

}  // namespace

// Un seul test pour le module : grille nulle, pleine, moitié gauche, bords, NaN et refus.
TEST_CASE("fondu local du mouvement : la grille de doute tire RIFE vers le fondu", "[motion_doubt]") {
  const ref::Motion m = moving();
  const std::array<std::uint8_t, 4> none{0, 0, 0, 0};
  auto same = ref::motion_doubt(m, {.cols = 2, .rows = 2, .weights = none}, 0.5F);
  REQUIRE(same);
  CHECK(same->flow == m.flow);  // aucun doute : le mouvement de RIFE au bit près
  CHECK(same->mask == m.mask);

  ref::Motion broken = m;
  broken.flow[0].at(3, 3) = std::numeric_limits<float>::quiet_NaN();
  const std::array<std::uint8_t, 1> full{kDoubtFull};
  auto faded = ref::motion_doubt(broken, {.cols = 1, .rows = 1, .weights = full}, 0.25F);
  REQUIRE(faded);
  CHECK(faded->flow[0].at(3, 3) == 0.0F);  // doute entier : l'entrée n'est pas lue, NaN compris
  CHECK(faded->mask.at(5, 4) == 0.25F);
  CHECK(faded->flow[2].at(0, 0) == 99.0F);  // hors de la fenêtre : recopié
  CHECK(faded->mask.at(11, 7) == 99.0F);

  // Moitié gauche en doute (2 × 1 cases) : bord gauche fondu, bord droit intact, rampe entre les centres.
  const std::array<std::uint8_t, 2> left{kDoubtFull, 0};
  auto half = ref::motion_doubt(m, {.cols = 2, .rows = 1, .weights = left}, 0.5F);
  REQUIRE(half);
  CHECK(half->flow[1].at(2, 3) == 0.0F);
  CHECK(half->mask.at(2, 3) == 0.5F);
  CHECK(half->flow[1].at(9, 3) == 4.0F);
  CHECK(half->mask.at(9, 3) == 0.9F);
  // x = 6 (relatif 4) : u = (4,5 × 2 / 8) − 0,5 = 0,625, w = 0,375.
  const float w = doubt_at(left.data(), 2, 1, doubt_coord(4, 2, 8), 0.0F);
  CHECK_THAT(w, WithinAbs(0.375, 1e-6));
  CHECK_THAT(half->flow[3].at(6, 3), WithinAbs(4.0 * (1.0 - 0.375), 1e-5));
  CHECK_THAT(half->mask.at(6, 3), WithinAbs(0.9 + (0.5 - 0.9) * 0.375, 1e-5));

  // Refus : grille vide, trop grande, de la mauvaise taille, fade hors de [0, 1].
  CHECK_FALSE(ref::motion_doubt(m, {.cols = 0, .rows = 1, .weights = {}}, 0.5F));
  const std::array<std::uint8_t, 2> two{0, 0};
  CHECK_FALSE(ref::motion_doubt(m, {.cols = kDoubtMaxCells + 1, .rows = 1, .weights = two}, 0.5F));
  CHECK_FALSE(ref::motion_doubt(m, {.cols = 2, .rows = 2, .weights = two}, 0.5F));
  CHECK_FALSE(ref::motion_doubt(m, {.cols = 2, .rows = 1, .weights = two}, 1.5F));
}
