// Ce que la sortie HDR du grade vise, sans GPU : le niveau du blanc des surfaces, le plafond des éclats et
// le pied des noirs. Les shaders font le même calcul (anime-hdr-look.hook, anime-hdr-black.hook) ; ce
// fichier en est le double en C++, pour régler leurs paramètres et pour les tests.
//
// Décisions 16 et 17 du chantier C06 (docs/audits/PRIORITES.md, docs/hdr-anime.md) :
// - 16 : un rendu HDR éclatant, mais les surfaces blanches redescendent pour que les vrais éclats gardent
//   leur écart. Le blanc d'une surface sort à la moitié du pic de l'écran (il sortait vers 70 %), et la
//   montée des émetteurs s'applique après la pile de LUT, sous le limiteur de la scène.
// - 17 : vrai noir pour le noir pur (sous RVB 4 environ), pied adouci au-dessus pour garder les nuits.
#pragma once

#include <array>
#include <span>

#include "render/cube_lut.h"
#include "render/grade_params.h"

namespace ajp::render {

// Le blanc d'une surface à l'écran : la moitié du pic. Sur un écran peu lumineux, la moitié du pic
// aplatirait les surfaces claires (à 400 nits, RVB 192 à 255 tenaient dans 10 % de lumière) : le blanc ne
// descend pas sous 300 nits, sans dépasser 70 % du pic.
inline constexpr double kSurfaceWhiteShare = 0.5;
inline constexpr double kSurfaceWhiteFloorNits = 300.0;
inline constexpr double kSurfaceWhiteMaxShare = 0.7;
double surface_white_target(double panel_peak);
// Le plafond des éclats ne descend jamais sous ce multiple du blanc des surfaces (scène très claire).
inline constexpr double kEmitterMinRoom = 1.25;

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
// Le sommet de la courbe qui rend `content_nits` à `target` nits affichés. Borné : jamais sous la courbe de
// Reinhard seule (la cible est trop basse), jamais au-dessus de l'identité (le niveau est déjà sous la
// cible : rien n'est compressé).
double curve_peak_for(double content_nits, double target, double reference);

// Le niveau d'un émetteur à l'écran : la surface `m` (nits affichés) montée de `gain`, sous le plafond `top`.
double emitter_level(double m, double gain, double top);

// Les réglages de la dernière passe pour un grade, les LUT de sa pile (leurs intensités sont dans
// params.look_mixes ; aucune : pas de pile) et l'écran.
struct OutputTune {
  double curve_peak = 1000.0;  // sommet de la courbe de compression, nits affichés
  double white = 500.0;        // blanc d'une surface à l'écran, nits affichés
  double top = 1000.0;         // plafond des éclats, nits affichés
  friend bool operator==(const OutputTune&, const OutputTune&) = default;
};
OutputTune output_tune(const GradeParams& params, std::span<const CubeLut* const> luts, double reference,
                       double panel_peak);

}  // namespace ajp::render
