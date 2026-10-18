// Contrat du warp arrière par flot et du mélange de deux images à l'instant t : base de
// Fast et de l'ordonnancement S3 (mouvement à la source, warp et mélange à la sortie).
//
// Repris de animejanai-inference-rtx@118c272 :
//   src/kernels.cu:963-1013 (sample_plane, sample_window, k_warp_blend),
//   src/kernels.cu:1025-1041 (k_flow_max), 1051-1096 (k_warp_blend_mid),
//   src/aji_trt.cpp:1242-1292 (setup_warp : seuil d'une seule passe par paire),
//   src/aji_trt.cpp:2633-2661 (loi de mélange ajustée, TrueHDR avant le warp).
// Tests : tests/unit/kernels/warp_test.cpp.
#pragma once

namespace ajp::kernels {

// Rectangle de l'image dans un plan plus grand (la fenêtre des tenseurs remplis de RIFE).
struct Window {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

// Échantillon bilinéaire d'un plan w × h en (x, y), centres des pixels aux entiers :
//   x, y bornés à [0, w − 1] × [0, h − 1] (bords en clamp ; NaN donne 0, comme fmaxf) ;
//   x0 = tronque(x), x1 = min(x0 + 1, w − 1), fx = x − x0 (de même en y) ;
//   haut = p(x0, y0)(1 − fx) + p(x1, y0) fx ; bas = p(x0, y1)(1 − fx) + p(x1, y1) fx ;
//   valeur = haut (1 − fy) + bas fy.
// Dans une fenêtre, les mêmes coordonnées sont relatives à son coin.

// Le mouvement : 4 plans (f0x, f0y, f1x, f1y) de déplacements, en pixels de la fenêtre, de
// l'instant vers a et vers b ; et un masque, poids de a dans [0, 1]. Pour le pixel (x, y)
// d'une image w × h (souvent plus grande que la fenêtre fw × fh) :
//   sx = w / fw, sy = h / fh ; u = (x + 0,5) / sx − 0,5, v = (y + 0,5) / sy − 0,5 ;
//   a lu en (x + sx f0x(u, v), y + sy f0y(u, v)), b en (x + sx f1x(u, v), y + sy f1y(u, v)) ;
//   sortie = a m + b (1 − m), m = masque(u, v).

// Instant t quelconque depuis le mouvement estimé au milieu de la paire (t = 1/2 exactement :
// mouvement et masque tels quels). Sinon chaque point va en ligne droite :
//   (u2, v2) = (u, v) − ((1 − t) f0 + t f1)(u, v) : où le point vu en t était au milieu ;
//   d = f1(u2, v2) − f0(u2, v2) ; vers a : −t d ; vers b : (1 − t) d ;
//   mm = clamp(masque(u2, v2), ε, 1 − ε) ;
//   logit(m) = A logit(mm) + B ln((1 − t) / t) + C, loi de RIFE ajustée sur de vraies paires.
inline constexpr float kBlendLawA = 0.285F;
inline constexpr float kBlendLawB = 2.591F;
inline constexpr float kBlendLawC = 0.170F;
inline constexpr float kMaskEpsilon = 1e-4F;

// Sous ce mouvement maximal, mesuré à 720p et mis à l'échelle de la hauteur où RIFE travaille,
// une seule passe RIFE sert toute la paire (warp_blend_at aux autres instants).
inline constexpr float kSingleRunMotion720 = 15.0F;

constexpr float single_run_motion(int rife_height) {
  return kSingleRunMotion720 * static_cast<float>(rife_height) / 720.0F;
}

}  // namespace ajp::kernels
