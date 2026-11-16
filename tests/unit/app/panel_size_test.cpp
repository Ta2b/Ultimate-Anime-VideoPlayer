#include "app/play/panel_size.h"

#include <catch2/catch_test_macros.hpp>

using ajp::app::play::PanelArea;
using ajp::app::play::PanelSize;

// Le menu des réglages garde sa taille en part de la fenêtre : il suit un redimensionnement de la fenêtre,
// et une taille tirée aux bords devient la nouvelle part.
TEST_CASE("Menu des réglages : la taille suit la fenêtre en proportion", "[app][panel]") {
  PanelSize size;
  // Première ouverture : 820 de large sur 1600, toute la hauteur.
  PanelArea wanted = size.target({1600.0F, 900.0F}, 820.0F);
  CHECK(size.area_changed());
  CHECK(wanted == PanelArea{820.0F, 900.0F});
  size.observe(wanted);

  // La fenêtre double : le menu double avec elle, et la taille s'impose.
  wanted = size.target({3200.0F, 1800.0F}, 820.0F);
  CHECK(size.area_changed());
  CHECK(wanted == PanelArea{1640.0F, 1800.0F});
  size.observe(wanted);

  // Les bords tirés à la moitié de la place : la part devient la moitié.
  CHECK(size.target({3200.0F, 1800.0F}, 820.0F) == PanelArea{1640.0F, 1800.0F});
  CHECK_FALSE(size.area_changed());
  size.observe({1600.0F, 900.0F});

  // La fenêtre revient à sa taille de départ : le menu en garde la moitié.
  wanted = size.target({1600.0F, 900.0F}, 820.0F);
  CHECK(wanted == PanelArea{800.0F, 450.0F});
}

TEST_CASE("Menu des réglages : une taille rabotée par le minimum ne change pas la part", "[app][panel]") {
  PanelSize size;
  size.observe(size.target({1600.0F, 900.0F}, 800.0F));  // la moitié de la largeur
  // Une petite fenêtre : le panneau ne descend pas sous son minimum (600 ici), plus que la moitié.
  size.target({800.0F, 450.0F}, 800.0F);
  size.observe({600.0F, 450.0F});
  CHECK(size.target({800.0F, 450.0F}, 800.0F) == PanelArea{400.0F, 450.0F});
  size.observe({600.0F, 450.0F});
  // La fenêtre regrandit : la moitié revient.
  CHECK(size.target({1600.0F, 900.0F}, 800.0F) == PanelArea{800.0F, 900.0F});
}
