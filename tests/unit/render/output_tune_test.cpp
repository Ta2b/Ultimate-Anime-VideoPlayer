// Les réglages de la sortie HDR sans GPU (render/output_tune.h) : le blanc des surfaces à la moitié du pic,
// le plafond des éclats, le pied adouci de la pile livrée. Décisions 16 et 17 du 09/11/2026.
#include "render/output_tune.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "grade_chain_support.h"

namespace ajp::render {

using Catch::Approx;

TEST_CASE("sortie HDR : le sommet de la courbe pose le niveau demandé, entre Reinhard et l'identité",
          "[render][tune]") {
  for (const double reference : {203.0, 300.0}) {
    // Un blanc de pile à 6 300 nits de contenu, visé à 500 nits affichés.
    const double peak = curve_peak_for(6300.0, 500.0, reference);
    CHECK(compress_neutral(6300.0, reference, peak) == Approx(500.0).epsilon(1e-9));
    // La courbe est croissante et ne dépasse pas son sommet.
    double before = 0.0;
    for (double nits = 1.0; nits <= 10000.0; nits *= 1.5) {
      const double out = compress_neutral(nits, reference, peak);
      CHECK(out > before);
      CHECK(out <= peak * 1.0001);
      before = out;
    }
  }
  // Cible sous la courbe de Reinhard seule : le sommet reste le sien, le niveau sort un peu plus haut.
  const double low = curve_peak_for(6300.0, 120.0, 203.0);
  CHECK(compress_neutral(6300.0, 203.0, low) == Approx(203.0 * (6300.0 / 203.0) / (1.0 + 6300.0 / 203.0)));
  // Niveau déjà sous la cible (aucune pile, A41-04) : l'identité, rien n'est compressé.
  const double none = curve_peak_for(233.0, 500.0, 203.0);
  CHECK(compress_neutral(233.0, 203.0, none) == Approx(233.0));
  CHECK(compress_neutral(40.0, 203.0, none) == Approx(40.0));
}

TEST_CASE("sortie HDR : un émetteur monte sans passer le plafond ni descendre sous sa surface",
          "[render][tune]") {
  // Sous le genou (65 % du plafond) : le gain tel quel.
  CHECK(emitter_level(100.0, 1.5, 1000.0) == Approx(150.0));
  // Au-dessus : une épaule, croissante avec le gain, sous le plafond.
  double before = 500.0;
  for (double gain = 1.0; gain < 20.0; gain *= 1.3) {
    const double out = emitter_level(500.0, gain, 1000.0);
    CHECK(out >= before);
    CHECK(out < 1000.0);
    before = out;
  }
  CHECK(emitter_level(500.0, 16.0, 1000.0) > 990.0);
  // Une surface déjà au-dessus du plafond n'est pas baissée par un gain d'émetteur.
  CHECK(emitter_level(700.0, 2.0, 625.0) == Approx(700.0));
  // Curseur Éclats sous 100 % : le niveau baisse tel quel.
  CHECK(emitter_level(500.0, 0.8, 1000.0) == Approx(400.0));
}

TEST_CASE("sortie HDR : blanc des surfaces à la moitié du pic, limiteur en scène très claire",
          "[render][tune]") {
  // La moitié du pic, jamais sous 300 nits ni au-dessus de 70 % du pic (écrans peu lumineux).
  CHECK(surface_white_target(350.0) == Approx(245.0));
  CHECK(surface_white_target(400.0) == Approx(280.0));
  CHECK(surface_white_target(500.0) == Approx(300.0));
  CHECK(surface_white_target(600.0) == Approx(300.0));
  CHECK(surface_white_target(1000.0) == Approx(500.0));
  CHECK(surface_white_target(2000.0) == Approx(1000.0));
  const CubeLut stack = gpu_test::default_stack();
  for (const double peak : {400.0, 600.0, 1000.0, 1500.0, 2000.0}) {
    GradeParams p = gpu_test::player_params(peak);
    p.fs_apl = 20.0;
    const OutputTune dark = output_tune(p, &stack, 203.0, peak);
    INFO("pic " << peak << " : blanc " << dark.white << ", sommet de la courbe " << dark.curve_peak);
    CHECK(dark.white == Approx(surface_white_target(peak)).epsilon(0.01));
    CHECK(dark.top == Approx(peak));
    p.fs_apl = 180.0;  // scène très claire : le plafond descend, sans rejoindre les surfaces
    const OutputTune bright = output_tune(p, &stack, 203.0, peak);
    CHECK(bright.white == Approx(dark.white));
    CHECK(bright.top == Approx(std::max(0.5 * peak, kEmitterMinRoom * dark.white)).epsilon(0.01));
  }
  // KDE à 300 nits de luminosité SDR : mêmes niveaux affichés.
  const GradeParams p = gpu_test::player_params(1000.0 * 203.0 / 300.0);
  CHECK(output_tune(p, &stack, 300.0, 1000.0).white == Approx(500.0).epsilon(0.01));
  // Sans pile (fichier .cube illisible, A41-04) : l'image n'est pas compressée, le blanc reste au relèvement.
  const OutputTune bare = output_tune(gpu_test::player_params(1000.0), nullptr, 203.0, 1000.0);
  CHECK(bare.white == Approx(203.0 * kAnimeLift));
}

TEST_CASE("pied des noirs : vrai noir sous RVB 4, nuits gardées, courbe croissante (pile livrée)",
          "[render][tune]") {
  const CubeLut stack = gpu_test::default_stack();
  const double white = 203.0 * kAnimeLift;
  const LookToe toe = look_toe(white);
  const auto level = [&](double code) {
    return look_neutral(stack, toe, std::pow(code / 255.0, 2.4) * white);
  };
  // Le noir pur reste un vrai noir.
  CHECK(level(0.0) == 0.0);
  CHECK(level(4.0) == Approx(0.0).margin(1e-6));
  // Les nuits : RVB 16 (0,30 nit) sortait à 0,02 nit et RVB 32 (1,6 nit) à 0,65 nit.
  CHECK(level(16.0) > 0.10);
  CHECK(level(16.0) < 0.304);
  CHECK(level(32.0) > 1.0);
  CHECK(level(32.0) < 1.603);
  // Au-dessus de RVB 48, la pile seule : rien ne change.
  const double join = std::pow(48.0 / 255.0, 2.4) * white;
  CHECK(toe_pile_share(toe, join) == 1.0);
  // Croissante de RVB 4 à RVB 64, sans marche : la pente en log reste entre 0,5 et 4.
  double before = level(4.5);
  for (double code = 5.0; code <= 64.0; code += 0.5) {
    const double now = level(code);
    INFO("code " << code << " : " << now << " nit");
    CHECK(now > before);
    if (code > 8.0) {
      const double slope = std::log(now / before) / (2.4 * std::log(code / (code - 0.5)));
      CHECK(slope > 0.5);
      CHECK(slope < 4.0);
    }
    before = now;
  }
}

}  // namespace ajp::render
