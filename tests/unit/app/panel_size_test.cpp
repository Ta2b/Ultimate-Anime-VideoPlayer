#include "app/play/panel_size.h"

#include <catch2/catch_test_macros.hpp>

using kirari::app::play::PanelArea;
using kirari::app::play::PanelRect;
using kirari::app::play::PanelSize;

// Le menu des réglages garde sa place en part de la fenêtre : il suit un redimensionnement de la fenêtre,
// et une taille tirée aux bords ou un déplacement devient la nouvelle part.
TEST_CASE("Menu des réglages : la taille suit la fenêtre en proportion", "[app][panel]") {
  PanelSize size;
  const PanelRect first{.x = 780.0F, .y = 0.0F, .width = 820.0F, .height = 900.0F};
  // Première ouverture : 820 de large sur 1600, collé à droite, toute la hauteur.
  PanelRect wanted = size.target({1600.0F, 900.0F}, first);
  CHECK(size.area_changed());
  CHECK(wanted == first);
  size.observe(wanted);

  // La fenêtre double : le menu double avec elle, et sa place s'impose.
  wanted = size.target({3200.0F, 1800.0F}, first);
  CHECK(size.area_changed());
  CHECK(wanted == PanelRect{1560.0F, 0.0F, 1640.0F, 1800.0F});
  size.observe(wanted);

  // Les bords tirés à la moitié de la place, en haut à gauche : la part devient la moitié.
  CHECK(size.target({3200.0F, 1800.0F}, first) == PanelRect{1560.0F, 0.0F, 1640.0F, 1800.0F});
  CHECK_FALSE(size.area_changed());
  size.observe({0.0F, 0.0F, 1600.0F, 900.0F});

  // La fenêtre revient à sa taille de départ : le menu en garde la moitié, au même endroit.
  wanted = size.target({1600.0F, 900.0F}, first);
  CHECK(wanted == PanelRect{0.0F, 0.0F, 800.0F, 450.0F});
}

TEST_CASE("Menu des réglages : déplacé, il garde sa place en proportion", "[app][panel]") {
  PanelSize size;
  size.observe(size.target({1600.0F, 900.0F}, {800.0F, 0.0F, 800.0F, 450.0F}));
  // Glissé par son titre au milieu de la place.
  size.target({1600.0F, 900.0F}, {});
  size.observe({400.0F, 225.0F, 800.0F, 450.0F});
  CHECK(size.target({1600.0F, 900.0F}, {}) == PanelRect{400.0F, 225.0F, 800.0F, 450.0F});
  // La fenêtre double : le menu reste au milieu.
  CHECK(size.target({3200.0F, 1800.0F}, {}) == PanelRect{800.0F, 450.0F, 1600.0F, 900.0F});
}

TEST_CASE("Menu des réglages : un rectangle raboté par le minimum ne change pas la part", "[app][panel]") {
  PanelSize size;
  size.observe(size.target({1600.0F, 900.0F}, {800.0F, 0.0F, 800.0F, 900.0F}));  // la moitié droite
  // Une petite fenêtre : le panneau ne descend pas sous son minimum (600 ici), plus que la moitié.
  size.target({800.0F, 450.0F}, {});
  size.observe({200.0F, 0.0F, 600.0F, 450.0F});
  CHECK(size.target({800.0F, 450.0F}, {}) == PanelRect{400.0F, 0.0F, 400.0F, 450.0F});
  size.observe({200.0F, 0.0F, 600.0F, 450.0F});
  // La fenêtre regrandit : la moitié droite revient.
  CHECK(size.target({1600.0F, 900.0F}, {}) == PanelRect{800.0F, 0.0F, 800.0F, 900.0F});
}

// Sans vidéo derrière (accueil, construction d'un moteur), le menu a une place fixe à droite qui laisse la
// carte de l'écran visible à sa gauche.
TEST_CASE("Menu des réglages : sans vidéo, une place fixe qui laisse la carte de l'accueil", "[app][panel]") {
  using kirari::app::play::fixed_panel;
  // Une grande fenêtre : la largeur la plus grande, collée à droite, toute la hauteur.
  CHECK(fixed_panel({2524.0F, 1404.0F}, 1375.0F, 650.0F, 727.5F) ==
        PanelRect{1149.0F, 0.0F, 1375.0F, 1404.0F});
  // La fenêtre de 1280 × 720 à 125 % : le menu prenait toute la largeur et cachait « Ouvrir une vidéo » ;
  // il laisse maintenant la place de la carte à gauche, à sa largeur la plus petite s'il le faut.
  const PanelRect small = fixed_panel({1244.0F, 684.0F}, 1244.0F, 650.0F, 727.5F);
  CHECK(small.width == 650.0F);
  CHECK(small.x == 594.0F);
  // Une fenêtre plus étroite que le minimum : le menu la remplit, sans en sortir.
  CHECK(fixed_panel({500.0F, 400.0F}, 500.0F, 650.0F, 727.5F) == PanelRect{0.0F, 0.0F, 500.0F, 400.0F});
}
