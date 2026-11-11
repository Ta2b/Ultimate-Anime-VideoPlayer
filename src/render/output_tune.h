// Ce que la sortie HDR du grade rend, sans GPU : le blanc des surfaces et le pied des noirs. Les shaders font
// le même calcul (anime-hdr-look.hook, anime-hdr-black.hook) ; ce fichier en est le double en C++, pour
// régler leurs paramètres et pour les tests.
//
// La pile de LUT est réglée à l'œil sous la compression que KWin faisait sans métadonnées : la dernière passe
// applique la même courbe, de 10 000 nits vers le pic de l'écran. Le blanc d'une surface sort donc là où la
// pile le porte (vers 70 % du pic), et les curseurs de la pile le déplacent ; le réglage « Blanc des
// surfaces » le baisse (GradeParams::surface_white). La montée des éclats s'applique
// ensuite, après la pile, jusqu'au pic de l'écran : leur écart sur les surfaces ne dépend pas des curseurs
// (docs/hdr-anime.md), et un émetteur part du niveau qu'il aurait sous la courbe à 100 % (B41-01) : baisser
// les surfaces ne baisse pas les éclats.
// Décision 17 du chantier C06 : vrai noir pour le noir pur (sous RVB 4 environ), pied adouci au-dessus pour
// garder les nuits.
#pragma once

#include <array>
#include <span>

#include "render/cube_lut.h"
#include "render/grade_params.h"

namespace ajp::render {

// Nits d'un code PQ (ST 2084) et son inverse.
namespace pq {
double to_nits(double code);
double from_nits(double nits);
}  // namespace pq

// Le pied adouci de la pile d'HDR, sur le maximum RVB avant la pile (nits). Les seuils sont les codes
// 8 bits d'une source au gamma 2,4, son blanc à `white` nits (le blanc relevé : sdr_white × kAnimeLift).
struct LookToe {
  double black = 0.0;  // RVB 4 : en dessous, 0
  double full = 0.0;   // RVB 8 : le pied entier
  double mid = 0.0;    // RVB 32 : la pile commence à reprendre sa part
  double join = 0.0;   // RVB 48 : la pile seule
};
inline constexpr double kToeExponent = 1.3;  // pente du pied en log (celle de la pile : 2)
LookToe look_toe(double white);
// Le niveau du pied pour un maximum RVB `m`, et la part de la pile (0 : le pied seul, 1 : la pile seule).
double toe_level(const LookToe& toe, double m);
double toe_pile_share(const LookToe& toe, double m);
// Un gris de `nits` passé par la pile (codes PQ, chaque LUT lue en trilinéaire) et le pied : ses nits.
double look_neutral(const LookChain& chain, const LookToe& toe, double nits);

// La compression de la dernière passe sur un gris : `content_nits` (signal PQ) rendu en nits affichés, pour
// le blanc de référence `reference` du compositeur et le sommet `curve_peak` de la courbe (nits affichés).
double compress_neutral(double content_nits, double reference, double curve_peak);
// Le sommet de la courbe qui ne compresse rien (l'identité) pour le blanc de référence `reference`.
double identity_curve_peak(double reference);

// Le sommet de la courbe qui rend `content_nits` à `target` nits affichés. Borné : jamais sous la courbe de
// Reinhard seule (cible trop basse), jamais au-dessus de l'identité (rien n'est compressé).
double curve_peak_for(double content_nits, double target, double reference);
// La courbe qui rend `content_nits` à `target` nits affichés, sans butée (B10-03) : le sommet de
// curve_peak_for, puis, sous Reinhard seul, un gain (< 1) sur toute la sortie. Chaque cran du curseur « Blanc
// des surfaces » baisse ainsi le blanc, même sur un écran peu lumineux ou sous un blanc SDR de KDE élevé.
struct SurfaceCurve {
  double peak = 1000.0;  // sommet de la courbe, nits affichés
  double gain = 1.0;     // facteur sur la sortie de la courbe : 1, sauf sous Reinhard seul
};
SurfaceCurve surface_curve_for(double content_nits, double target, double reference);

// Le niveau d'un émetteur à l'écran : la surface `m` (nits affichés) montée de `gain`, sous le plafond `top`
// (le même calcul que ajk_emit, anime-hdr-black.hook).
double emitter_level(double m, double gain, double top);
// Le niveau d'un émetteur quand « Blanc des surfaces » baisse la courbe (B41-01, le même calcul que la
// dernière passe d'anime-hdr-black.hook) : `m` est sa surface sous la courbe baissée, `m_full` sous la courbe
// à 100 %. Un émetteur franc (au moins un demi-diaphragme de montée) part de `m_full` : il garde le niveau
// qu'il a à 100 %. Une montée plus faible mêle les deux, sans marche ; un gain de 1 ou moins suit les
// surfaces.
double emitter_over_surface(double m, double m_full, double gain, double top);

// Les réglages de la dernière passe pour un grade, les LUT de sa pile (leurs intensités sont dans
// params.look_mixes ; aucune : pas de pile) et l'écran.
// Avec la pile, la courbe va vers le pic de l'écran ; sans elle, la dernière passe ne compresse pas une image
// que la pile n'a pas montée (A41-04).
struct OutputTune {
  double curve_peak = 1000.0;  // sommet de la courbe de compression, nits affichés
  double curve_gain = 1.0;     // facteur sur la sortie de la courbe (SurfaceCurve::gain)
  double full_peak = 1000.0;  // sommet de la courbe à « Blanc des surfaces » 100 % : là où partent les éclats
  double white = 700.0;       // blanc d'une surface à l'écran, nits affichés (le MaxFALL annoncé)
  double top = 1000.0;        // plafond des éclats, nits affichés : le pic de l'écran
  friend bool operator==(const OutputTune&, const OutputTune&) = default;
};
OutputTune output_tune(const GradeParams& params, std::span<const CubeLut* const> luts, double reference,
                       double panel_peak);

}  // namespace ajp::render
