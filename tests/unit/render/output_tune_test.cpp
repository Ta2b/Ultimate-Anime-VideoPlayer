// Les réglages de la sortie HDR sans GPU (render/output_tune.h) : la courbe vers le pic de l'écran, le blanc
// des surfaces que la pile livrée porte, le pied adouci (décision 17 du 09/11/2026).
#include "render/output_tune.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "grade_chain_support.h"
#include "render/display_model.h"

namespace ajp::render {

using Catch::Approx;

TEST_CASE("sortie HDR : la courbe va vers le pic de l'écran, l'identité sans pile", "[render][tune]") {
  for (const double reference : {203.0, 300.0}) {
    double before = 0.0;
    for (double nits = 1.0; nits <= 10000.0; nits *= 1.5) {
      const double out = compress_neutral(nits, reference, 1000.0);
      CHECK(out > before);
      CHECK(out <= 1000.0 * 1.0001);
      before = out;
    }
    // Sans pile (A41-04) : rien n'est compressé, le signal est seulement rapporté au blanc du compositeur.
    for (const double nits : {40.0, 233.0, 4000.0})
      CHECK(compress_neutral(nits, reference, identity_curve_peak(reference)) ==
            Approx(nits * reference / kContentReferenceNits));
  }
}

// Le rendu d'avant la 0.11.0-rc.1 (décision de Tatoub du 11/11/2026) : le blanc d'une surface sort là où la
// pile le porte sous la courbe de KWin, vers 70 % du pic, et les curseurs de la pile le déplacent. La rc.1
// résolvait le sommet de la courbe pour le poser à la moitié du pic : les curseurs n'y changeaient plus rien.
TEST_CASE("sortie HDR : blanc des surfaces vers 70 % du pic, déplacé par les curseurs de la pile",
          "[render][tune]") {
  const gpu_test::Pile stack;
  for (const auto& [peak, white] : {std::pair{400.0, 320.0}, {1000.0, 690.0}, {2000.0, 1304.0}}) {
    const OutputTune tune = output_tune(gpu_test::player_params(peak), stack.luts, 203.0, peak);
    INFO("pic " << peak << " : blanc " << tune.white << ", sommet de la courbe " << tune.curve_peak);
    CHECK(tune.curve_peak == Approx(peak));
    CHECK(tune.white == Approx(white).epsilon(0.03));
  }
  const auto white_of = [](int contrast, int highlight) {
    const gpu_test::Pile pile(contrast, highlight);
    GradeParams p = gpu_test::player_params(1000.0);
    p.look_mixes = pile.mixes;
    return output_tune(p, pile.luts, 203.0, 1000.0).white;
  };
  const double base = white_of(0, 0);
  INFO("blanc : " << base << " ; contraste -100 " << white_of(-100, 0) << ", 100 " << white_of(100, 0)
                  << " ; éclats -100 " << white_of(0, -100) << ", 100 " << white_of(0, 100));
  CHECK(white_of(-100, 0) < 0.85 * base);
  CHECK(white_of(100, 0) > 1.1 * base);
  CHECK(white_of(0, -100) < 0.85 * base);
  CHECK(white_of(0, 100) > 1.1 * base);
  // Sans pile (fichier .cube illisible, A41-04) : l'image n'est pas compressée, le blanc reste au relèvement.
  const OutputTune bare = output_tune(gpu_test::player_params(1000.0), {}, 203.0, 1000.0);
  CHECK(bare.white == Approx(203.0 * kAnimeLift));
}

TEST_CASE("pied des noirs : vrai noir sous RVB 4, nuits gardées, courbe croissante (pile livrée)",
          "[render][tune]") {
  const gpu_test::Pile stack;
  const double white = 203.0 * kAnimeLift;
  const LookToe toe = look_toe(white);
  const auto level = [&](double code) {
    return look_neutral(stack.chain(), toe, std::pow(code / 255.0, 2.4) * white);
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
