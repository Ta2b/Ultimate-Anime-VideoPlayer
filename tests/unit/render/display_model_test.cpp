// Le modèle de l'écran (render/display_model.h) : échelle de KWin, pic en unités du contenu, courbe du
// limiteur d'une dalle OLED.
#include "render/display_model.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using ajp::render::DisplayModel;
using Catch::Matchers::WithinRel;

TEST_CASE("Modèle d'écran : le pic de KWin en unités du contenu", "[render][display]") {
  // Luminosité SDR à 300 : le blanc de 203 du contenu s'affiche à 300 ; un pic de 1000 affiché se vise à 677.
  const DisplayModel kwin{.panel_peak_nits = 1000.0, .reference_nits = 300.0};
  CHECK_THAT(kwin.scale(), WithinRel(300.0 / 203.0, 1e-9));
  CHECK_THAT(kwin.content_peak_nits(), WithinRel(1000.0 * 203.0 / 300.0, 1e-9));
  // Référence à 203 (ou Windows) : rien ne change.
  CHECK_THAT(DisplayModel{.panel_peak_nits = 800.0}.content_peak_nits(), WithinRel(800.0, 1e-9));
}

TEST_CASE("Modèle d'écran : limiteur d'une dalle OLED, pic tenu par un LCD", "[render][display]") {
  const DisplayModel oled{.panel_peak_nits = 1016.0, .oled = true};
  CHECK_THAT(oled.max_nits(0.02), WithinRel(1016.0, 1e-9));
  CHECK_THAT(oled.max_nits(0.10), WithinRel(460.0, 1e-9));
  CHECK_THAT(oled.max_nits(1.0), WithinRel(261.0, 1e-9));
  CHECK(oled.max_nits(0.01) == oled.max_nits(0.02));
  // Décroissante entre deux relevés.
  CHECK(oled.max_nits(0.05) < 1016.0);
  CHECK(oled.max_nits(0.05) > 460.0);
  CHECK(oled.max_nits(0.3) < oled.max_nits(0.25));
  REQUIRE(oled.frame_average_content_nits());
  CHECK_THAT(*oled.frame_average_content_nits(), WithinRel(261.0, 1e-9));
  const DisplayModel lcd{.panel_peak_nits = 1000.0};
  CHECK(lcd.max_nits(1.0) == 1000.0);
  CHECK_FALSE(lcd.frame_average_content_nits());
}
