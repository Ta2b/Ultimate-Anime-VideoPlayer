// Réglages du menu des paramètres : le look (LUT), le grade adaptatif et les réponses de tête livrées.
#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "app/play/bundled_hrir.h"
#include "app/play/live_settings.h"
#include "app/play/settings.h"

TEST_CASE("le look (LUT) et son intensité se relisent à l'identique, un nom de chemin est refusé",
          "[app][settings]") {
  ajp::app::play::PlaySettings s;
  s.look = "maison:shinkai";
  s.look_mix = 0.65;
  s.look_layers = {{"HDR_base.cube", 0.4}, {"maison:pastel", 0.25}, {"x.cube", 1.0}};
  const auto back = ajp::app::play::parse_settings(ajp::app::play::format_settings(s));
  REQUIRE(back.has_value());
  CHECK(back->look == "maison:shinkai");
  CHECK(back->look_mix == 0.65);
  CHECK(back->look_layers == s.look_layers);
  CHECK_FALSE(ajp::app::play::parse_settings("[grade]\nlooks = ['a/b.cube']\n").has_value());
  CHECK_FALSE(
      ajp::app::play::parse_settings("[grade]\nlooks = ['a.cube']\nlooks_mix = [-0.1]\n").has_value());
  CHECK_FALSE(ajp::app::play::parse_settings(
                  "[grade]\nlooks = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']\n")
                  .has_value());
  const auto no_mix = ajp::app::play::parse_settings("[grade]\nlooks = ['a.cube']\n");
  REQUIRE(no_mix.has_value());
  CHECK(no_mix->look_layers.at(0).mix == 1.0);
  CHECK_FALSE(ajp::app::play::parse_settings("[grade]\nlook = '../x.cube'\n").has_value());
  CHECK_FALSE(ajp::app::play::parse_settings("[grade]\nlook_mix = 1.5\n").has_value());
  const ajp::app::play::PlaySettings defaults;  // la pile de LUT de Tatoub par défaut : base + 7 LUT
  CHECK(defaults.look == "HDR grading base ver.9.0 33 point.cube");
  CHECK(defaults.look_layers.size() == 7);
}

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

TEST_CASE("réponses de tête livrées : chacune se charge par bundled:<clé>", "[settings][audio][hrir]") {
  using namespace ajp::app::play;
  REQUIRE(bundled_hrirs().size() == 5);
  for (const BundledHrir& entry : bundled_hrirs()) {
    const auto loaded = load_listening_hrir("bundled:" + std::string(entry.key));
    INFO(entry.key << ": " << (loaded ? "" : loaded.error().detail));
    REQUIRE(loaded.has_value());
    REQUIRE(*loaded != nullptr);
    CHECK((*loaded)->length() > 0);
  }
  CHECK(find_bundled_hrir("bundled:inconnue") == nullptr);
  CHECK(find_bundled_hrir("/home/x/y.wav") == nullptr);
  CHECK_FALSE(load_listening_hrir("").value() != nullptr);
}
