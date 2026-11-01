// Grade adaptatif : relecture de play.toml et glissement des réglages de la scène sombre à la scène claire.
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "app/play/chain.h"
#include "app/play/settings.h"

TEST_CASE("le grade adaptatif se relit à l'identique et survit au changement de préréglage",
          "[app][settings]") {
  ajp::app::play::PlaySettings s;
  s.grade_values.adaptive = true;
  s.grade_values.adaptive_lo = 15.0;
  s.grade_values.adaptive_hi = 54.0;
  s.grade_values.adaptive_gray = 32.0;
  s.grade_values.adaptive_shadows = 0.7;
  s.grade_values.adaptive_vibrance = 0.1;
  const auto back = ajp::app::play::parse_settings(ajp::app::play::format_settings(s));
  REQUIRE(back.has_value());
  CHECK(back->grade_values.adaptive);
  CHECK(back->grade_values.adaptive_lo == 15.0);
  CHECK(back->grade_values.adaptive_hi == 54.0);
  CHECK(back->grade_values.adaptive_gray == 32.0);
  CHECK(back->grade_values.adaptive_shadows == 0.7);
  CHECK(back->grade_values.adaptive_vibrance == 0.1);
  const auto switched = ajp::app::play::with_preset(ajp::render::GradePreset::Naturel, s.grade_values);
  CHECK(switched.adaptive);
  CHECK(switched.adaptive_hi == 54.0);
  CHECK(switched.adaptive_gray == 32.0);
  CHECK(ajp::app::play::PlaySettings{}.grade_values.adaptive);  // adaptatif par défaut
}

TEST_CASE("grade adaptatif : noirs, couleurs et gris moyen glissent de la scène sombre à la scène claire",
          "[app][settings]") {
  ajp::app::play::PlaySettings s;
  s.true_hdr_gray = 70;
  s.grade_values.shadows = 0.2;
  s.grade_values.vibrance_rtx = 0.8;
  s.grade_values.adaptive = true;
  s.grade_values.adaptive_shadows = 0.8;
  s.grade_values.adaptive_vibrance = 0.0;
  s.grade_values.adaptive_vibrance_cut = 0.0;
  s.grade_values.adaptive_gray = 30.0;
  const auto at = [&](double t, double tv = 0.0) { return ajp::app::play::with_adaptive(s, t, tv); };
  CHECK(at(0.0).grade_values.shadows == 0.2);  // scène sombre : les curseurs
  CHECK(at(0.0).true_hdr_gray == 70);
  CHECK(at(1.0).grade_values.shadows == 0.8);  // scène claire : les valeurs adaptatives
  CHECK(at(1.0).true_hdr_gray == 30);
  CHECK(at(1.0).grade_values.vibrance_rtx == 0.8);  // la vibrance suit la couleur, pas la luminosité
  CHECK(at(1.0, 1.0).grade_values.vibrance_rtx == 0.0);
  CHECK(at(0.5).grade_values.shadows == 0.5);
  CHECK(at(0.5).true_hdr_gray == 50);
  CHECK(at(0.5, 0.5).grade_values.vibrance_rtx == 0.4);
  CHECK(at(0.0, 1.0).grade_values.vibrance_rtx ==
        0.0);                                       // ciel bleu saturé, scène pas claire : couleurs coupées
  CHECK(at(0.0, 1.0).grade_values.shadows == 0.2);  // les noirs restent ceux de la scène sombre
  s.grade_values.adaptive_vibrance_cut = 0.5;       // scènes claires : la moitié de la vibrance en moins
  CHECK(at(1.0).grade_values.vibrance_rtx == 0.4);
  CHECK(at(0.5).grade_values.vibrance_rtx == Catch::Approx(0.6));
  CHECK(at(0.0).grade_values.vibrance_rtx == 0.8);
  s.grade_values.adaptive = false;
  CHECK(at(1.0).true_hdr_gray == 70);  // coupé : rien ne bouge
}

TEST_CASE("grade adaptatif : une série claire par nature reçoit plus de contraste et de couleur",
          "[app][settings]") {
  ajp::app::play::PlaySettings s;
  s.true_hdr_contrast = 100;
  s.grade_values.adaptive = true;
  const auto ordinary = ajp::app::play::with_adaptive(s, 0.0, 0.0, 0.0);
  CHECK(ordinary.true_hdr_contrast == 100);
  CHECK(ordinary.grade_values.punch_base == s.grade_values.punch_base);
  const auto bright = ajp::app::play::with_adaptive(s, 0.0, 0.0, 1.0);
  CHECK(bright.true_hdr_contrast == 115);
  CHECK(bright.grade_values.punch_base > s.grade_values.punch_base);
  CHECK(bright.grade_values.chroma_base > s.grade_values.chroma_base);
  s.true_hdr_contrast = 190;
  CHECK(ajp::app::play::with_adaptive(s, 0.0, 0.0, 1.0).true_hdr_contrast == 200);  // borné à 200
}
