// Rectangles salis de l'interface gardée (C24-01) : un titre en haut et une barre en bas restent deux bandes.
#include <catch2/catch_test_macros.hpp>

#include "app/play/placebo_imgui.h"

namespace ajp::app::play {
namespace {

long long area(const std::vector<pl_rect2d>& rects) {
  long long sum = 0;
  for (const pl_rect2d& r : rects) sum += static_cast<long long>(r.x1 - r.x0) * (r.y1 - r.y0);
  return sum;
}

}  // namespace

TEST_CASE("C24-01 : le titre et la barre restent deux bandes, sans relire toute la hauteur", "[imgui]") {
  std::vector<pl_rect2d> rects;
  add_dirty_rect(rects, {0, 0, 3840, 96});
  add_dirty_rect(rects, {0, 2000, 3840, 2160});
  REQUIRE(rects.size() == 2);
  // L'union d'avant couvrait 3840 × 2160 ; les deux bandes font moins de 20 % de la hauteur.
  CHECK(area(rects) * 100 < 3840LL * 2160 * 20);
}

TEST_CASE("C24-01 : des rectangles qui se touchent ou se recouvrent se fondent", "[imgui]") {
  std::vector<pl_rect2d> rects;
  add_dirty_rect(rects, {0, 0, 100, 50});
  add_dirty_rect(rects, {0, 50, 100, 80});  // touche le premier
  add_dirty_rect(rects, {10, 10, 20, 20});  // contenu
  add_dirty_rect(rects, {0, 300, 100, 320});
  add_dirty_rect(rects, {0, 70, 100, 310});  // relie les deux bandes
  REQUIRE(rects.size() == 1);
  CHECK(rects[0].y0 == 0);
  CHECK(rects[0].y1 == 320);
}

TEST_CASE("C24-01 : un rectangle vide est ignoré, le nombre de bandes est borné", "[imgui]") {
  std::vector<pl_rect2d> rects;
  add_dirty_rect(rects, {5, 5, 5, 9});
  CHECK(rects.empty());
  for (int i = 0; i < 20; ++i) add_dirty_rect(rects, {0, i * 100, 10, i * 100 + 10});
  CHECK(rects.size() <= kMaxDirtyRects);
  // Chaque petit rectangle reste couvert par une bande.
  for (int i = 0; i < 20; ++i) {
    bool covered = false;
    for (const pl_rect2d& r : rects) covered = covered || (r.y0 <= i * 100 && r.y1 >= i * 100 + 10);
    CHECK(covered);
  }
}

}  // namespace ajp::app::play
