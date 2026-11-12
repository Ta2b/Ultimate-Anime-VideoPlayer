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

TEST_CASE("Option --hdr-metadata : le maximum annoncé est le pic du contenu", "[render][display]") {
  const ajp::render::KwinHint kde = ajp::render::kwin_hint(1000.0, 300.0);
  CHECK(kde.max_content_nits == 1000.0);
  CHECK_THAT(kde.panel_peak_nits, Catch::Matchers::WithinAbs(1477.8, 0.1));
  const ajp::render::KwinHint identity = ajp::render::kwin_hint(1000.0, 203.0);
  CHECK_THAT(identity.panel_peak_nits, Catch::Matchers::WithinAbs(1000.0, 1e-9));
  CHECK(identity.max_content_nits == 1000.0);
}

TEST_CASE("contrat : k = 1,305, C_pk = 766, MaxFALL = 200 en contenu", "[render][display]") {
  // KDE à 265 nits de référence sur un QD-OLED de 1000 : le plein écran du limiteur (261) vaut 200 du
  // contenu.
  ajp::render::DisplayContract c{.reference_nits = 265.0, .panel_peak_nits = 1000.0};
  c.oled = true;
  c.abl = ajp::render::default_abl(c.panel_peak_nits, c.oled);
  CHECK_THAT(c.k(), WithinRel(265.0 / 203.0, 1e-9));
  CHECK_THAT(c.content_peak_nits(), WithinRel(766.0, 1e-3));
  CHECK_THAT(c.abl.back(), WithinRel(261.0, 0.01));
  CHECK_THAT(c.content_frame_average_nits(), WithinRel(200.0, 0.01));
  CHECK(c.abl.front() == 1000.0);
  // Une dalle LCD tient son pic sur toute surface.
  for (const double nits : ajp::render::default_abl(600.0, false)) CHECK(nits == 600.0);
}

TEST_CASE("métadonnées du contrat v2 : C_pk annoncé, MaxFALL et noir en unités du contenu",
          "[render][display]") {
  ajp::render::DisplayContract c{.reference_nits = 265.0, .panel_peak_nits = 1000.0};
  c.oled = true;
  c.min_nits = 0.0005 * 265.0 / 203.0;
  c.abl = ajp::render::default_abl(c.panel_peak_nits, c.oled);
  const ajp::render::ContractMetadata meta = ajp::render::contract_metadata(c);
  CHECK_THAT(meta.max_luma, WithinRel(766.0, 1e-3));
  CHECK(meta.max_cll == meta.max_luma);
  CHECK_THAT(meta.max_fall, WithinRel(200.0, 0.01));
  CHECK_THAT(meta.min_luma, WithinRel(0.0005, 1e-9));
  CHECK(meta.primaries == ajp::render::kDisplayP3);  // sans primaires de l'EDID
  c.primaries = ajp::render::Chromaticities{0.68, 0.30, 0.25, 0.71, 0.15, 0.06, 0.3135, 0.3291};
  CHECK(ajp::render::contract_metadata(c).primaries == *c.primaries);
}

TEST_CASE("limiteur : la courbe va du pic au plein champ que l'écran dit (C11-04)", "[render][display]") {
  // WOLED de 800 nits, 150 plein champ : de 800 à 2 % à 150 à 100 %, toujours décroissante.
  const auto woled = ajp::render::abl_to_full_frame(800.0, 150.0);
  CHECK_THAT(woled.front(), WithinRel(800.0, 1e-9));
  CHECK_THAT(woled.back(), WithinRel(150.0, 1e-9));
  for (std::size_t i = 1; i < woled.size(); ++i) CHECK(woled[i] < woled[i - 1]);
  // La courbe QD-OLED de référence se retrouve avec ses propres bouts.
  const auto qd = ajp::render::abl_to_full_frame(1016.0, 261.0);
  CHECK_THAT(qd[1], WithinRel(460.0, 1e-6));
  CHECK_THAT(qd[3], WithinRel(310.0, 1e-6));
  // Un plein champ au pic (LCD) ou inconnu : le pic partout.
  for (const double nits : ajp::render::abl_to_full_frame(600.0, 600.0)) CHECK(nits == 600.0);
  for (const double nits : ajp::render::abl_to_full_frame(600.0, 0.0)) CHECK(nits == 600.0);
}
