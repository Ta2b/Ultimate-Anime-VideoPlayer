#include "app/play/window_size.h"

#include <catch2/catch_test_macros.hpp>

using ajp::app::play::window_size_for;
using ajp::app::play::WindowSize;

TEST_CASE("Fenêtre : sans échelle du bureau (macOS, Wayland, 100 %), la taille demandée", "[app][window]") {
  CHECK(window_size_for(1280, 720, 1.0F, 1920, 1040) == WindowSize{1280, 720});
  CHECK(window_size_for(1280, 720, 0.0F, 0, 0) == WindowSize{1280, 720});
  // Même sur un petit écran : rien ne change hors de Windows et X11 mis à l'échelle.
  CHECK(window_size_for(1280, 720, 1.0F, 1024, 700) == WindowSize{1280, 720});
}

TEST_CASE("Fenêtre : à 150 % sous Windows, en points comme ailleurs", "[app][window]") {
  CHECK(window_size_for(1280, 720, 1.5F, 2880, 1752) == WindowSize{1920, 1080});
  CHECK(window_size_for(1280, 720, 1.25F, 0, 0) == WindowSize{1600, 900});
}

TEST_CASE("Fenêtre : bornée aux 9/10 de l'écran, proportions gardées", "[app][window]") {
  // Un portable 1920 × 1200 à 200 % : 2560 × 1440 ne tient pas, ramené à 1728 × 972.
  const WindowSize size = window_size_for(1280, 720, 2.0F, 1920, 1152);
  CHECK(size == WindowSize{1728, 972});
  CHECK(size.width * 720 == size.height * 1280);
  // Jamais plus petite que la taille demandée.
  CHECK(window_size_for(1280, 720, 1.5F, 1366, 728) == WindowSize{1280, 720});
}
