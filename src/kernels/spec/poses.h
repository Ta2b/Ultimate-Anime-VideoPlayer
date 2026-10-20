// Contrat de « garder les poses » : le DRBA (Distance Ratio Based Adjuster, routineLife1/DRBA,
// licence MIT, models/drm.py : calc_drm_rife en mode linéaire) adapté au warp de spec/warp.h et à
// une lecture en temps réel.
//
// Le DRBA règle l'instant de chaque pixel par le rapport des distances qu'il parcourt avant et
// après l'image centrale : un personnage animé en deux sur un panoramique en uns garde son rythme
// (il tient sa pose puis change d'un coup) pendant que le fond glisse. Le DRBA d'origine lit
// l'image suivante de la paire ; la lecture ne l'a pas encore. Ici, la distance d'après est
// prise égale à celle d'avant, la cadence de l'animation (en uns, en deux) se répétant d'une
// paire à l'autre : c'est exactement le DRBA pour un mouvement régulier et pour l'animation en
// deux, ses deux cas d'école.
//
// Écrit pour ce dépôt. Tests : tests/unit/kernels/warp_test.cpp.
#pragma once

namespace ajp::kernels {

// Pour le pixel de sortie (u, v) de la grille du mouvement (spec/warp.h), à l'instant t de la paire
// (a, b), avec le mouvement au milieu de la paire et celui de la paire précédente (même grille) :
//
// 1. tp = t, puis kPoseIterations fois :
//    le point vu en (u, v) à l'instant tp : (u2, v2) = (u, v) − ((1 − tp) f0 + tp f1)(u, v),
//    son déplacement D = f1(u2, v2) − f0(u2, v2), sa position dans a : pa = (u, v) − tp D ;
//    son déplacement dans la paire précédente : q = pa − f1'(pa), puis q = pa − f1'(q),
//    D' = f1'(q) − f0'(q) (f0', f1' : mouvement de la paire précédente) ; une paire précédente
//    tenue (images égales) donne D' = 0 ;
//    rapport r = (|D'| + ε) / (|D'| + |D| + 2ε), ε = kPoseEpsilon ;
//    tp = pose_time(t, r), borné à [kPoseTimeLimit, 1 − kPoseTimeLimit].
// 2. La sortie est warp_blend_at au temps tp pour ce pixel (spec/warp.h : mouvement depuis le
//    milieu, loi de mélange comprise).
//
// r = 1/2 (mouvement régulier) laisse tp = t. r = 0 (la paire précédente ne bougeait pas, ce pixel
// est animé en deux) tient a jusqu'au milieu de la paire, puis passe à b.
inline constexpr int kPoseIterations = 2;
inline constexpr float kPoseEpsilon = 0.01F;  // pixels de la grille
inline constexpr float kPoseTimeLimit = 1e-3F;

// Instant réglé : 2 t r avant le milieu de la paire (le DRBA vu de a), 1 − 2 (1 − t) r à partir du
// milieu (vu de b, distance d'après prise égale à celle d'avant).
constexpr float pose_time(float t, float r) {
  return t < 0.5F ? 2.0F * t * r : 1.0F - 2.0F * (1.0F - t) * r;
}

}  // namespace ajp::kernels
