// Courbe HDR par plan (étape 8.5 du plan, partie CPU) : la luminance d'un plan de contenu vers l'écran,
// d'après le contrat de l'écran et l'APL du plan. Pure, sans GPU ; la cuisson en LUT et le
// planificateur de fond (8.6) s'en serviront. Rien ne l'appelle encore dans la lecture.
//
// - Décalage E (diaphragmes, |E| ≤ 0,25) : −0,25 par diaphragme d'APL du plan au-dessus de la médiane
//   de la série, plus le décalage choisi par l'utilisateur ; les plans clairs descendent un peu, les
//   sombres montent un peu, sans jamais s'écarter de plus d'un demi-diaphragme entre eux.
// - Sous le genou K = 0,75 · 203 · 2^E (nits affichés) : un gain constant 2^E, le gris 203 du contenu
//   tombe à 203 · 2^E.
// - Au-dessus : Hermite monotone (Fritsch-Carlson) en PQ, du genou (pente du gain) jusqu'au maximum du
//   contenu, qui tombe sur T = min(C_pk, T_aire, T_ABL), jamais sous le genou.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "render/cube_lut.h"

namespace ajp::render {

inline constexpr std::size_t kCurveHistBins = 32;  // histogramme de luminance du plan, par code PQ

struct CurveInput {
  double shot_apl_nits = 0.0;    // APL du plan (contenu)
  double series_apl_nits = 0.0;  // médiane des APL de la série ; 0 : inconnue (pas de pente)
  double user_e = 0.0;           // décalage choisi (diaphragmes)
  double content_max_nits = 1000.0;
  double panel_peak_nits = 1000.0;  // C_pk, pic de l'écran en nits de contenu
  double area_limit_nits = 0.0;     // T_aire ; 0 : sans limite
  double abl_limit_nits = 0.0;      // T_ABL ; 0 : sans limite
  // Budget ABL : l'APL affichée du plan ne le dépasse pas (T baissé par bissection, jamais sous le
  // genou) ; 0 : sans budget. L'APL se prédit sur l'histogramme du plan (poids par code PQ).
  double apl_budget_nits = 0.0;
  std::array<float, kCurveHistBins> lum_hist{};
  double vibrance = 1.0;  // gain de chroma hors du secteur des peaux, au plus 1,15
  // Contraste du pied (8.10, ±kCurveMaxToe) : sous kCurveToeNits, la sortie est multipliée par
  // (nits / kCurveToeNits)^toe ; positif creuse les ombres, négatif les ouvre. 0 : sans effet.
  double toe = 0.0;
};

struct ShotCurve {
  double e = 0.0;         // décalage retenu (diaphragmes)
  double gain = 1.0;      // 2^e
  double knee_in = 0.0;   // genou, nits du contenu
  double knee_out = 0.0;  // genou, nits affichés
  double top_in = 0.0;    // maximum du contenu
  double top_out = 0.0;   // T
  double m0 = 0.0;        // pentes Hermite en PQ, au genou et en haut
  double m1 = 0.0;
  double vibrance = 1.0;
  double toe = 0.0;
};

inline constexpr double kCurveMaxE = 0.25;
inline constexpr double kCurveSlope = 0.25;
inline constexpr double kCurveMaxToe = 0.2;
inline constexpr double kCurveToeNits = 20.0;  // nits du contenu

double curve_pq(double nits);  // PQ (ST 2084) de nits, 0 à 1
double curve_nits(double pq);  // l'inverse
ShotCurve plan_curve(const CurveInput& in);
// La luminance affichée (nits) d'une luminance de contenu (nits). Sous 0,5 nit, un pied ramène le
// gain vers 1 : les noirs ne bougent pas d'un plan à l'autre.
double map_luminance(const ShotCurve& curve, double nits);
// L'APL affichée prédite d'un histogramme (poids par code PQ).
double predicted_apl(const ShotCurve& curve, const std::array<float, kCurveHistBins>& hist);

struct Rgb {
  double r = 0.0;
  double g = 0.0;
  double b = 0.0;
};
// Une couleur BT.2020 linéaire (nits) vers l'écran : I suit la courbe, Ct/Cp × (I_out/I_in)^0,4, et la
// vibrance hors du secteur des peaux.
Rgb map_color(const ShotCurve& curve, Rgb nits);
// La LUT 3D (size³ entrées RGB, rouge le plus rapide) en codes PQ, indexée par √code sur chaque axe.
std::vector<float> bake(const ShotCurve& curve, int size = 65);
// La même, le look SDR (contrat v2 : ver9-couleur.cube, domaine BT.709 gamma 2,4, blanc 203 nits) appliqué
// avant la courbe : la part BT.709 de la couleur passe dans le look (une couleur au-dessus du blanc y entre
// ramenée sous 1 puis remonte d'autant), la part hors BT.709 reste. look nul : bake(curve, size).
std::vector<float> bake(const ShotCurve& curve, const CubeLut* look_sdr, int size = 65);
inline constexpr double kCurveMaxVibrance = 1.15;
// ΔE_ITP (BT.2124) entre deux couleurs BT.2020 linéaires en nits ; 1 ≈ un écart à peine visible.
double delta_e_itp(Rgb a, Rgb b);

}  // namespace ajp::render
