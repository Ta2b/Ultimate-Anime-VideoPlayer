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

TEST_CASE("sortie HDR : le curseur Blanc des surfaces baisse le blanc, 100 % garde celui de la pile",
          "[render][tune]") {
  const gpu_test::Pile stack;
  for (const double peak : {400.0, 1000.0}) {
    GradeParams p = gpu_test::player_params(peak);
    const OutputTune full = output_tune(p, stack.luts, 203.0, peak);
    CHECK(full.curve_peak == Approx(peak));  // 100 % : la courbe de la rc.4, vers le pic
    double before = full.white;
    for (const double share : {0.9, 0.7, 0.5}) {
      p.surface_white = share;
      const OutputTune lower = output_tune(p, stack.luts, 203.0, peak);
      CAPTURE(peak, share, lower.white);
      CHECK(lower.white < before);
      // La courbe ne descend pas sous Reinhard seul : sur un écran peu lumineux, le plus bas reste au-dessus.
      if (peak >= 1000.0) CHECK(lower.white == Approx(share * full.white).epsilon(0.02));
      CHECK(lower.white >= 0.98 * share * full.white);
      CHECK(lower.top == Approx(peak));  // les éclats montent toujours au pic
      before = lower.white;
    }
  }
  // Sans pile, rien n'est compressé : le curseur n'y fait rien.
  GradeParams bare = gpu_test::player_params(1000.0);
  bare.surface_white = 0.5;
  CHECK(output_tune(bare, {}, 203.0, 1000.0).curve_peak == Approx(identity_curve_peak(203.0)));
}

// B10-03 : sous Reinhard seul, la courbe ne descendait plus ; tous les crans du curseur rendaient le même
// blanc (61 % à 60 et 50 % sur 400 nits, 82 % de 80 à 50 % sous un blanc SDR de KDE à 300 nits).
TEST_CASE("Blanc des surfaces : chaque cran baisse le blanc, aussi sous KDE (blanc SDR 300 nits)",
          "[render][tune]") {
  const gpu_test::Pile stack;
  for (const double reference : {203.0, 300.0}) {
    for (const double peak : {400.0, 600.0}) {
      GradeParams p = gpu_test::player_params(peak);
      const double full = output_tune(p, stack.luts, reference, peak).white;
      double before = full;
      for (const double share : {0.9, 0.8, 0.7, 0.6, 0.5}) {
        p.surface_white = share;
        const OutputTune tune = output_tune(p, stack.luts, reference, peak);
        CAPTURE(reference, peak, share, tune.white, before, tune.curve_gain);
        CHECK(tune.white < before - 1.0);  // un cran qui ne change rien est une promesse non tenue
        CHECK(tune.white == Approx(share * full).epsilon(0.02));
        CHECK(tune.curve_gain <= 1.0);
        CHECK(tune.full_peak == Approx(peak));  // les éclats partent toujours de la courbe à 100 %
        before = tune.white;
      }
    }
  }
}

// B41-01 : « Blanc des surfaces » sous 100 % baissait aussi les émetteurs (699 nits au lieu de 889 sur un
// écran de 1 000 nits à 70 %), contre la promesse de l'infobulle. Un émetteur franc garde son niveau.
TEST_CASE("Blanc des surfaces : un émetteur garde son niveau, sans marche à la porte des émetteurs",
          "[render][tune]") {
  const gpu_test::Pile stack;
  constexpr double kSpecGain = 1.444;  // un blanc à Éclats 100 %
  for (const double peak : {400.0, 1000.0}) {
    GradeParams p = gpu_test::player_params(peak);
    const OutputTune full = output_tune(p, stack.luts, 203.0, peak);
    const double surface_full = full.white;
    const double emitter_full = emitter_over_surface(surface_full, surface_full, kSpecGain, full.top);
    for (const double share : {0.9, 0.7, 0.5}) {
      p.surface_white = share;
      const OutputTune lower = output_tune(p, stack.luts, 203.0, peak);
      // La surface sous la courbe baissée, et sous la courbe à 100 % (full_peak, celle de full).
      CHECK(lower.full_peak == Approx(full.curve_peak));
      const double m = lower.white;
      const double m_full = surface_full;
      const double emitter = emitter_over_surface(m, m_full, kSpecGain, lower.top);
      CAPTURE(peak, share, m, m_full, emitter, emitter_full);
      CHECK(m < surface_full);
      CHECK(emitter >= 0.97 * emitter_full);
      // Sans montée, l'émetteur suit les surfaces ; une montée infime ne fait pas de marche.
      CHECK(emitter_over_surface(m, m_full, 1.0, lower.top) == Approx(m));
      CHECK(emitter_over_surface(m, m_full, 1.0001, lower.top) == Approx(m).epsilon(0.01));
    }
  }
}

TEST_CASE("éclats après la pile : le gain entier sous le genou, une épaule douce sous le pic de l'écran",
          "[render][tune]") {
  for (const double top : {400.0, 1000.0}) {
    // Sous le genou (65 % du pic), le gain passe entier ; un gain de 1 ou moins ne touche à rien.
    CHECK(emitter_level(100.0, 2.0, top) == Approx(200.0));
    CHECK(emitter_level(300.0, 0.9, top) == Approx(270.0));
    // Au-dessus, l'épaule monte toujours et reste sous le pic.
    double before = 0.0;
    for (double gain = 1.0; gain <= 64.0; gain *= 1.25) {
      const double out = emitter_level(0.6 * top, gain, top);
      CHECK(out >= before);
      CHECK(out <= top);
      before = out;
    }
    // Une surface déjà au pic ou au-delà reste où elle est.
    CHECK(emitter_level(top, 3.0, top) == Approx(top));
    CHECK(emitter_level(1.2 * top, 3.0, top) == Approx(1.2 * top));
  }
  // Le plafond suit l'écran : le même émetteur monte plus haut sur une dalle plus lumineuse.
  CHECK(emitter_level(300.0, 3.0, 1000.0) > emitter_level(300.0, 3.0, 400.0));
  // La sortie réglée pour un écran porte son pic comme plafond.
  const gpu_test::Pile stack;
  GradeParams params = gpu_test::player_params(800.0);
  CHECK(output_tune(params, stack.luts, 203.0, 800.0).top == Approx(800.0));
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
