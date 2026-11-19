// Contrat de l'affinage de la caméra de la carte de la scène (porte 2.1 du plan du flux NVIDIA, méthode du
// banc : tools/bench/qualite/nvof_mesures.py, refine_affine et photo_residual). L'affine que la carte ajuste
// sur les vecteurs de NVOF (ref::scene_camera_of) est biaisée sous le pixel sur les caméras lentes (sur un
// panoramique lent du banc, NVOF invente un cisaillement d'environ 2 pixels à 720p) ; Lucas-Kanade global sur
// les pixels du décor le retire. Sur les mesures par paire du banc, 391 paires sur 407 passent ainsi la porte
// (écart médian sous 0,5 pixel à 720p, au moins 40 % de décor).
//
// Repère. Pixels du flux (spec/scene_flow.h), le centre du pixel (i, j) en (i + 1/2, j + 1/2) ; coordonnées
// réduites X = (i + 1/2 − cx) / s, Y = (j + 1/2 − cy) / s (scene_camera_frame), en float. Modèle p (6
// doubles, lus en float par pixel) : le point (x, y) de a est en (x + dx, y + dy) dans b, avec
// dx = p0 + p1 X + p2 Y et dy = p3 + p4 X + p5 Y ; b se lit donc en (i + dx, j + dy), indices de pixels.
//
// Entrées : les lumas du flux de la paire, a et b (8 bits, W × H), le modèle de départ (celui de NVOF) et les
// tuiles du décor (cols × rows, celles que l'ajustement a gardées).
//
// Affinage. Les lumas floutées par une boîte (2r + 1)² : moyenne, bords répétés, en float, somme sur la ligne
// puis sur la colonne, chacune dans l'ordre des décalages de −r à r, puis × 1/(2r + 1)². r vaut
// kRefineCoarseRadius pendant kRefineSteps pas (la portée), puis kRefineFineRadius pendant kRefineSteps pas
// (la précision). Un pas de Gauss-Newton sur les pixels (i, j) des tuiles du décor dont i et j sont multiples
// de kRefineStride (scene_warp_term, scene_refine_terms) :
//   le pixel compte si 1 ≤ i + dx < W − 2 et 1 ≤ j + dy < H − 2 ;
//   w = B(i + dx, j + dy) (bilinéaire), gx et gy : différences centrées de B à ±1 pixel du point ;
//   e = A(i, j) − w ; J = (gx, gx X, gx Y, gy, gy X, gy Y) ;
//   poids de Huber : 1 si |e| ≤ c, c / |e| sinon, c = kRefineHuber × max(m, kRefineHuberFloor), m la médiane
//   des |e| du pas lue sur l'histogramme de scene_refine_bin (scene_refine_median) ;
//   H = Σ poids J Jᵀ, g = Σ poids J e, en double ; p += H⁻¹ g (scene_refine_solve).
// H mal conditionné (rapport de ses valeurs propres au-delà de kRefineMaxCondition, ou valeur propre nulle) :
// le pas n'est pas fait, ni aucun autre ; p reste celui du dernier pas.
//
// Vérification, celle que mesure la porte 2.1, sur chaque tuile de l'image (décor ou non). Lumas floutées en
// (2 kCheckRadius + 1)², kCheckSteps pas de Lucas-Kanade sur le décalage (sx, sy) qui reste après le modèle
// affiné, sur tous les pixels de la tuile dans l'image (scene_warp_term, au point i + dx + sx) : sommes
// Σ gx², Σ gx gy, Σ gy², Σ gx e, Σ gy e et le nombre de pixels comptés, puis scene_check_step. Rend
// (sx, sy, l1, l2, pixels) du dernier pas. Au premier pas, sur le modèle seul, aussi Σ |e| et le nombre de
// pixels qui bougent (scene_pixel_moving), rendus par pixel compté (scene_check_seen) : l'écart après la
// caméra, ce que la carte voit d'un dessin qui bouge hors d'elle (ref::scene_motion_of).
// ref::scene_camera_apply tire l'écart de la caméra des seules tuiles du décor.
//
// Référence : ref::scene_camera_refine (kernels/ref/scene_camera.cpp) ; port : gpu/cuda/backend
// (scene_camera_kernels.cu). Les sommes du port sont en float par fil puis en double : le résultat suit la
// référence à l'arrondi près, pas au bit.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kernels/spec/fp_off_begin.h"  // IWYU pragma: keep

#if defined(__CUDACC__)
#define KIRARI_SCENE_HD __host__ __device__
#else
#define KIRARI_SCENE_HD
#endif

namespace kirari::kernels {

namespace scene {
inline constexpr int kRefineCoarseRadius = 3;  // flou 7 × 7 : la portée des premiers pas
inline constexpr int kRefineFineRadius = 1;    // flou 3 × 3 : la précision des derniers
inline constexpr int kRefineSteps = 3;         // pas de Gauss-Newton par flou
// Un pixel sur deux en x et en y : 6 paramètres, des dizaines de milliers de pixels de décor en 960 × 540, et
// le coût sur les SM divisé par quatre (la porte du temps par image est à 0,3 ms).
inline constexpr int kRefineStride = 2;
inline constexpr int kRefineMinPixels = 1000;  // pixels du flux des tuiles du décor, sinon pas d'affinage
inline constexpr double kRefineMaxCondition = 1e8;     // rapport des valeurs propres de H au plus
inline constexpr float kRefineHuber = 2.5F * 1.4826F;  // × la médiane des |e|
inline constexpr float kRefineHuberFloor = 0.25F;      // médiane prise au moins à ceci, codes
inline constexpr int kRefineBins = 256;                // histogramme des |e| : 1/8 de code par case
inline constexpr float kRefineBinsPerCode = 8.0F;
inline constexpr int kRefineSums = 27;  // H (21, triangle haut ligne par ligne), puis g (6)
inline constexpr int kCheckRadius = 1;  // flou 3 × 3, celui des derniers pas
inline constexpr int kCheckSteps = 2;
// Structure par pixel (niveaux² de gradient) d'une direction résolue : au-dessus du grain, environ 2 après le
// flou 3 × 3.
inline constexpr float kCheckFloor = 8.0F;
inline constexpr float kCheckClip = 2.0F;    // pas de vérification par axe au plus, pixels du flux
inline constexpr float kCheckCover = 0.25F;  // part des pixels d'une tuile comptés, au moins
inline constexpr int kCheckMinTiles = 3;     // tuiles mesurées au moins pour un écart
static_assert(kCheckRadius == kRefineFineRadius, "la vérification reprend les lumas du dernier flou");
// Un pixel bouge après la caméra si |e| dépasse ce qu'un décalage de kMovingSlack pixels à 720p explique :
// |e| > kMovingCodes + slack × |∇b| (scene_pixel_moving). Le grain, flouté en 3 × 3, reste sous 2 à 3 codes ;
// un trait mal recalé de moins d'un demi-pixel (la porte 2.1) reste sous le terme du gradient. Réglés sans
// mesure (19 novembre), à revoir sur la ligne « image » du banc 2.3 (carte_lecteur.py).
inline constexpr float kMovingCodes = 6.0F;
inline constexpr float kMovingSlack = 0.5F;  // pixels à 720p
}  // namespace scene

// Ce que le lecteur demande pour une paire, et ce qu'il relit.
struct SceneCameraRequest {
  std::array<double, 6> model{};    // départ : l'affine de NVOF, repère ci-dessus
  std::vector<std::uint8_t> decor;  // cols × rows, 1 : tuile du décor
};

struct SceneTileCheck {
  float sx = 0.0F;  // décalage qui reste après le modèle, pixels du flux
  float sy = 0.0F;
  float l1 = 0.0F;  // valeurs propres du tenseur de structure du dernier pas (sommes sur la tuile)
  float l2 = 0.0F;
  float pixels = 0.0F;  // pixels comptés au dernier pas
  // Au premier pas, sur le modèle seul : écart moyen |e| (codes, lumas floutées) et part des pixels qui
  // bougent (scene_pixel_moving), sur les pixels comptés.
  float error = 0.0F;
  float moving = 0.0F;
};
static_assert(sizeof(SceneTileCheck) == 7 * sizeof(float));

struct SceneCameraRefined {
  std::array<double, 6> model{};      // affiné ; celui de la demande si aucun pas n'a été fait
  int steps = 0;                      // pas de Gauss-Newton faits, 2 × kRefineSteps au plus
  std::vector<SceneTileCheck> tiles;  // cols × rows
};

struct SceneCameraFrame {
  float cx = 0.0F;
  float cy = 0.0F;
  float inv_scale = 1.0F;  // 1 / s
};

// Le repère d'une image du flux de width × height : centre, et s = max(1, max(W, H) / 2). Celui de
// l'ajustement sur les tuiles (ref::scene_camera_of), où les tuiles sont à leur centre.
KIRARI_SCENE_HD inline SceneCameraFrame scene_camera_frame(int width, int height) {
  const float half = 0.5F * static_cast<float>(width > height ? width : height);
  return {0.5F * static_cast<float>(width), 0.5F * static_cast<float>(height),
          1.0F / (half > 1.0F ? half : 1.0F)};
}

// Coordonnées réduites et déplacement du modèle p au pixel (i, j).
struct SceneModelAt {
  float x = 0.0F;
  float y = 0.0F;
  float dx = 0.0F;
  float dy = 0.0F;
};
KIRARI_SCENE_HD inline SceneModelAt scene_model_at(const float* p, const SceneCameraFrame& f, int i, int j) {
  SceneModelAt m;
  m.x = (static_cast<float>(i) + 0.5F - f.cx) * f.inv_scale;
  m.y = (static_cast<float>(j) + 0.5F - f.cy) * f.inv_scale;
  m.dx = p[0] + p[1] * m.x + p[2] * m.y;
  m.dy = p[3] + p[4] * m.x + p[5] * m.y;
  return m;
}

// Lecture bilinéaire de img (width de large) en (x, y), indices de pixels ; x0 + 1 et y0 + 1 dans l'image.
KIRARI_SCENE_HD inline float scene_bilinear(const float* img, int width, float x, float y) {
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const float* r0 = img + static_cast<std::size_t>(y0) * static_cast<std::size_t>(width) + x0;
  const float* r1 = r0 + width;
  const float top = r0[0] + fx * (r0[1] - r0[0]);
  const float bottom = r1[0] + fx * (r1[1] - r1[0]);
  return top + fy * (bottom - top);
}

// Le terme d'un pixel : gradient de b au point déplacé et écart à a ; valid faux hors de l'image.
struct SceneWarpTerm {
  float gx = 0.0F;
  float gy = 0.0F;
  float e = 0.0F;
  bool valid = false;
};
KIRARI_SCENE_HD inline SceneWarpTerm scene_warp_term(const float* a, const float* b, int width, int height,
                                                     int i, int j, float dx, float dy) {
  SceneWarpTerm t;
  const float xs = static_cast<float>(i) + dx;
  const float ys = static_cast<float>(j) + dy;
  // Écrit pour qu'un NaN ne compte pas.
  if (!(xs >= 1.0F && xs < static_cast<float>(width - 2) && ys >= 1.0F &&
        ys < static_cast<float>(height - 2)))
    return t;
  const float w = scene_bilinear(b, width, xs, ys);
  t.gx = 0.5F * (scene_bilinear(b, width, xs + 1.0F, ys) - scene_bilinear(b, width, xs - 1.0F, ys));
  t.gy = 0.5F * (scene_bilinear(b, width, xs, ys + 1.0F) - scene_bilinear(b, width, xs, ys - 1.0F));
  t.e = a[static_cast<std::size_t>(j) * static_cast<std::size_t>(width) + static_cast<std::size_t>(i)] - w;
  t.valid = true;
  return t;
}

// La case de |e| dans l'histogramme du pas, et la médiane qu'on y lit (le milieu de la case où le compte
// cumulé atteint la moitié, arrondie au-dessus).
KIRARI_SCENE_HD inline int scene_refine_bin(float abs_e) {
  const float bin = abs_e * scene::kRefineBinsPerCode;
  return bin < static_cast<float>(scene::kRefineBins - 1) ? static_cast<int>(bin) : scene::kRefineBins - 1;
}
KIRARI_SCENE_HD inline float scene_refine_median(const unsigned* histogram, unsigned count) {
  const unsigned half = (count + 1U) / 2U;
  unsigned run = 0;
  for (int k = 0; k < scene::kRefineBins; ++k) {
    run += histogram[k];
    if (count > 0U && run >= half) return (static_cast<float>(k) + 0.5F) / scene::kRefineBinsPerCode;
  }
  return 0.0F;
}
KIRARI_SCENE_HD inline float scene_huber_weight(float e, float median) {
  const float c =
      scene::kRefineHuber * (median > scene::kRefineHuberFloor ? median : scene::kRefineHuberFloor);
  const float magnitude = e < 0.0F ? -e : e;
  return magnitude <= c ? 1.0F : c / magnitude;
}

// Les 27 contributions d'un pixel : H (triangle haut) puis g, déjà pondérées.
KIRARI_SCENE_HD inline void scene_refine_terms(const SceneWarpTerm& t, const SceneModelAt& m, float weight,
                                               float* out) {
  const float jac[6] = {t.gx, t.gx * m.x, t.gx * m.y, t.gy, t.gy * m.x, t.gy * m.y};
  int k = 0;
  for (int r = 0; r < 6; ++r) {
    const float wr = weight * jac[r];
    for (int c = r; c < 6; ++c) out[k++] = wr * jac[c];
  }
  for (int r = 0; r < 6; ++r) out[21 + r] = weight * jac[r] * t.e;
}

// Une rotation de Jacobi de la matrice symétrique e, qui annule e[a][b].
KIRARI_SCENE_HD inline void scene_jacobi_rotate(double (&e)[6][6], int a, int b) {
  if (e[a][b] == 0.0) return;
  const double theta = (e[b][b] - e[a][a]) / (2.0 * e[a][b]);
  const double t = (theta >= 0.0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
  const double co = 1.0 / sqrt(t * t + 1.0);
  const double si = t * co;
  for (int r = 0; r < 6; ++r) {
    const double ra = e[r][a];
    const double rb = e[r][b];
    e[r][a] = co * ra - si * rb;
    e[r][b] = si * ra + co * rb;
  }
  for (int c = 0; c < 6; ++c) {
    const double ac = e[a][c];
    const double bc = e[b][c];
    e[a][c] = co * ac - si * bc;
    e[b][c] = si * ac + co * bc;
  }
}

// H assez bien conditionné : valeurs propres (Jacobi) toutes positives, la plus grande au plus
// kRefineMaxCondition fois la plus petite.
KIRARI_SCENE_HD inline bool scene_refine_conditioned(const double (&h)[6][6]) {
  double e[6][6];
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) e[r][c] = h[r][c];
  }
  for (int sweep = 0; sweep < 50; ++sweep) {
    double off = 0.0;
    double diag = 0.0;
    for (int r = 0; r < 6; ++r) {
      diag += e[r][r] * e[r][r];
      for (int c = r + 1; c < 6; ++c) off += e[r][c] * e[r][c];
    }
    if (!(off > 1e-30 * diag)) break;
    for (int a = 0; a < 5; ++a) {
      for (int b = a + 1; b < 6; ++b) scene_jacobi_rotate(e, a, b);
    }
  }
  double low = e[0][0];
  double high = e[0][0];
  for (int r = 1; r < 6; ++r) {
    low = e[r][r] < low ? e[r][r] : low;
    high = e[r][r] > high ? e[r][r] : high;
  }
  return low > 0.0 && high <= scene::kRefineMaxCondition * low;
}

// H delta = g par Cholesky (H = L Lᵀ) ; faux si H n'est pas définie positive ou si delta est démesuré.
KIRARI_SCENE_HD inline bool scene_refine_cholesky(const double (&h)[6][6], const double* g, double* delta) {
  double l[6][6] = {};
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c <= r; ++c) {
      double s = h[r][c];
      for (int q = 0; q < c; ++q) s -= l[r][q] * l[c][q];
      if (r != c) {
        l[r][c] = s / l[c][c];
      } else if (s > 0.0) {
        l[r][r] = sqrt(s);
      } else {
        return false;
      }
    }
  }
  double y[6];
  for (int r = 0; r < 6; ++r) {
    double s = g[r];
    for (int q = 0; q < r; ++q) s -= l[r][q] * y[q];
    y[r] = s / l[r][r];
  }
  bool bounded = true;
  for (int r = 5; r >= 0; --r) {
    double s = y[r];
    for (int q = r + 1; q < 6; ++q) s -= l[q][r] * delta[q];
    delta[r] = s / l[r][r];
    bounded = bounded && fabs(delta[r]) < 1e6;  // NaN compris
  }
  return bounded;
}

// Le pas de Gauss-Newton : faux, p inchangé, si H est mal conditionné (scene_refine_conditioned) ; sinon
// p += H⁻¹ g (scene_refine_cholesky).
KIRARI_SCENE_HD inline bool scene_refine_solve(const double* sums, double* p) {
  double h[6][6];
  int k = 0;
  for (int r = 0; r < 6; ++r) {
    for (int c = r; c < 6; ++c) {
      h[r][c] = sums[k];
      h[c][r] = sums[k];
      ++k;
    }
  }
  double delta[6];
  if (!scene_refine_conditioned(h) || !scene_refine_cholesky(h, sums + 21, delta)) return false;
  for (int r = 0; r < 6; ++r) p[r] += delta[r];
  return true;
}

// Un pas de la vérification d'une tuile : tenseur de structure, chaque direction assez texturée résolue,
// l'autre laissée (sur un bord, l'écart le long du bord est indéterminé).
struct SceneCheckSums {
  float xx = 0.0F;
  float xy = 0.0F;
  float yy = 0.0F;
  float bx = 0.0F;
  float by = 0.0F;
  float pixels = 0.0F;
  float error = 0.0F;   // Σ |e|
  float moving = 0.0F;  // pixels qui bougent
};
KIRARI_SCENE_HD inline void scene_check_step(const SceneCheckSums& s, SceneTileCheck& out) {
  const float half = 0.5F * (s.xx + s.yy);
  const float d = 0.5F * (s.xx - s.yy);
  const float disc = sqrtf(d * d + s.xy * s.xy);
  out.l1 = half + disc;
  out.l2 = half - disc;
  out.pixels = s.pixels;
  const float theta = 0.5F * atan2f(2.0F * s.xy, s.xx - s.yy);
  const float c = cosf(theta);
  const float si = sinf(theta);
  const float floor = scene::kCheckFloor * (s.pixels > 1.0F ? s.pixels : 1.0F);
  const float p1 = out.l1 >= floor ? (c * s.bx + si * s.by) / (out.l1 > 1e-6F ? out.l1 : 1e-6F) : 0.0F;
  const float p2 = out.l2 >= floor ? (c * s.by - si * s.bx) / (out.l2 > 1e-6F ? out.l2 : 1e-6F) : 0.0F;
  const float ux = p1 * c - p2 * si;
  const float uy = p1 * si + p2 * c;
  out.sx += ux < -scene::kCheckClip ? -scene::kCheckClip : (ux > scene::kCheckClip ? scene::kCheckClip : ux);
  out.sy += uy < -scene::kCheckClip ? -scene::kCheckClip : (uy > scene::kCheckClip ? scene::kCheckClip : uy);
}

// La tolérance de scene_pixel_moving pour un flux de height lignes : kMovingSlack pixels à 720p.
KIRARI_SCENE_HD inline float scene_moving_slack(int height) {
  return scene::kMovingSlack * static_cast<float>(height) / 720.0F;
}
// Le pixel bouge après la caméra : son écart n'est pas celui d'un décalage d'au plus slack pixels.
KIRARI_SCENE_HD inline bool scene_pixel_moving(const SceneWarpTerm& t, float slack) {
  const float magnitude = t.e < 0.0F ? -t.e : t.e;
  return magnitude > scene::kMovingCodes + slack * sqrtf(t.gx * t.gx + t.gy * t.gy);
}
// Le premier pas de la vérification : l'écart après la caméra, par pixel compté.
KIRARI_SCENE_HD inline void scene_check_seen(const SceneCheckSums& s, SceneTileCheck& out) {
  const float count = s.pixels > 1.0F ? s.pixels : 1.0F;
  out.error = s.error / count;
  out.moving = s.moving / count;
}

}  // namespace kirari::kernels

#include "kernels/spec/fp_off_end.h"  // IWYU pragma: keep
