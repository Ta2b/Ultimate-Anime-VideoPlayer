// Contrat du « mouvement d'ensemble » du warp de RIFE (réglage pan_lock, [interpolation] de play.toml) :
// une translation globale de la paire, estimée finement sur les images, remplace le flot de RIFE là où il est
// petit et proche d'elle ; sous un demi-pixel de déplacement, le mélange glisse vers un fondu linéaire.
//
// Pourquoi (docs/interpolation.md, « Mouvement d'ensemble ») : sur un panoramique lent, RIFE voit mal le
// sous-pixel (0,2 px pour 1 px réel, rien à 0,3 px) et son flot varie d'une planche à l'autre : les traits
// ondulent, et la loi de mélange colle les instants à la source la plus proche (saccade). Une translation
// mesurée au centième de pixel donne à tout ce qui glisse ensemble le même déplacement.
//
// Estimation (pan_estimate en ref, pan_global.comp, pan_global en CUDA (warp_kernels.cu)), une fois par appel
// du warp, sur la luma (kPanLumaR, kPanLumaG, kPanLumaB) des images a et b à la taille du warp (w × h) :
//   mx = max(4, w / 12), my = max(4, h / 12) ; kPanGrid × kPanGrid points entiers
//   px = mx + (i · (w − 2 mx)) / kPanGrid, py = my + (j · (h − 2 my)) / kPanGrid (divisions entières) ;
//   gx, gy : différences centrées de la luma de a au point ;
//   kPanIterations fois : r = luma de b lue en bilinéaire (bornée à l'image) en (px + dx, py + dy) − luma de
//   a ;
//     sommes Sxx, Sxy, Syy, Sxr, Syr ; det = Sxx Syy − Sxy² ;
//     det > kPanDetRatio (Sxx + Syy)² : (dx, dy) −= (Syy Sxr − Sxy Syr, Sxx Syr − Sxy Sxr) / det,
//     sinon l'estimation n'est pas sûre (image sans texture) ;
//   sûre : le dernier det était valable et |(dx, dy)| < kPanMaxShift.
// (dx, dy) est le déplacement du contenu de a vers b, en pixels de l'image.
//
// Emploi (from_middle et estimated du warp, hors garder les poses) : s l'instant du flot lu, 1/2 pour le flot
// du milieu de RIFE, t pour un flot estimé à l'instant de la sortie (Motion::at_instant : DistilDRBA, que le
// warp reçoit avec l'image du centre en a, et RIFE estimé directement à t). Le flot lu f (4 plans, pixels de
// la fenêtre du mouvement) est remplacé par g = (−s dx / sx, −s dy / sy, (1 − s) dx / sx, (1 − s) dy / sy)
// quand l'estimation est sûre, max |f| < kPanFlowLimit et max |f − g| < kPanAgreement ; puis, même sans
// estimation sûre, d le déplacement du point,
//   w = smoothstep(kSlowFadeLo, kSlowFadeHi, |d|) ;
//   logit(m) = w · (loi ajustée de spec/warp.h, ou masque estimé tel quel) + (1 − w) · ln((1 − t) / t)
// (fondu linéaire, poids 1 − t, sous un demi-pixel ; à t = 1/2, ln vaut 0 et le masque est ramené vers 1/2).
// Garder les poses (spec/poses.h) n'est pas touché ; celles que DistilDRBA garde de lui-même restent hors du
// fondu dès qu'elles bougent de plus de deux pixels, et leur flot ne suit pas la translation de la paire.
#pragma once

#include "kernels/spec/fp_off_begin.h"  // IWYU pragma: keep

namespace ajp::kernels {

inline constexpr int kPanGrid = 32;
inline constexpr int kPanIterations = 3;
inline constexpr float kPanLumaR = 0.2126F;
inline constexpr float kPanLumaG = 0.7152F;
inline constexpr float kPanLumaB = 0.0722F;
inline constexpr float kPanDetRatio = 1e-6F;
inline constexpr float kPanMaxShift = 8.0F;
inline constexpr float kPanFlowLimit = 2.0F;
inline constexpr float kPanAgreement = 1.0F;
inline constexpr float kSlowFadeLo = 0.5F;
inline constexpr float kSlowFadeHi = 2.0F;

// Estimation d'une paire : déplacement (px de l'image) et sûreté.
struct PanEstimate {
  float dx = 0.0F;
  float dy = 0.0F;
  bool sure = false;
};

// Le point i (0 … kPanGrid − 1) de la grille sur une longueur n.
constexpr int pan_point(int i, int n) {
  const int m = n / 12 > 4 ? n / 12 : 4;
  return m + (i * (n - 2 * m)) / kPanGrid;
}

// smoothstep(kSlowFadeLo, kSlowFadeHi, d) : poids de la loi ajustée face au fondu.
constexpr float slow_fade_weight(float d) {
  float s = (d - kSlowFadeLo) / (kSlowFadeHi - kSlowFadeLo);
  s = s < 0.0F ? 0.0F : (s > 1.0F ? 1.0F : s);
  return s * s * (3.0F - 2.0F * s);
}

}  // namespace ajp::kernels

#include "kernels/spec/fp_off_end.h"  // IWYU pragma: keep
