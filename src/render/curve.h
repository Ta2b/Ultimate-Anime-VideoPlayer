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

namespace ajp::render {

struct CurveInput {
  double shot_apl_nits = 0.0;    // APL du plan (contenu)
  double series_apl_nits = 0.0;  // médiane des APL de la série ; 0 : inconnue (pas de pente)
  double user_e = 0.0;           // décalage choisi (diaphragmes)
  double content_max_nits = 1000.0;
  double panel_peak_nits = 1000.0;  // C_pk, pic de l'écran en nits de contenu
  double area_limit_nits = 0.0;     // T_aire ; 0 : sans limite
  double abl_limit_nits = 0.0;      // T_ABL ; 0 : sans limite
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
};

inline constexpr double kCurveMaxE = 0.25;
inline constexpr double kCurveSlope = 0.25;

double curve_pq(double nits);  // PQ (ST 2084) de nits, 0 à 1
double curve_nits(double pq);  // l'inverse
ShotCurve plan_curve(const CurveInput& in);
// La luminance affichée (nits) d'une luminance de contenu (nits).
double map_luminance(const ShotCurve& curve, double nits);

}  // namespace ajp::render
