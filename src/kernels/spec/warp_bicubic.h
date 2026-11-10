// La lecture bicubique des images du warp (audits A49-06, A22-03, compromis C7) : quand les images sont plus
// grandes que la fenêtre du mouvement (warp après SPAN, ou RIFE estimé plus bas que la sortie), le warp les
// lit par Catmull-Rom au lieu du bilinéaire, qui adoucit le trait que SPAN vient de dessiner. Le mouvement
// et le masque restent lus en bilinéaire (spec/warp.h).
//
// Échantillon d'un plan w × h en (x, y), centres des pixels aux entiers :
//   x, y bornés à [0, w − 1] × [0, h − 1] comme le bilinéaire (NaN donne 0) ;
//   x0 = tronque(x), fx = x − x0 ; prises x0 − 1 … x0 + 2, indices bornés au plan (de même en y) ;
//   poids de Catmull-Rom (cubique de Keys, a = −1/2) en fx puis en fy : quatre sommes de ligne, puis la somme
//   des lignes, en float, depuis 0 ;
//   la valeur est bornée au minimum et au maximum des quatre voisins du centre (x0, x0 + 1) × (y0, y0 + 1) :
//   pas de rebond clair ou sombre le long d'un trait. Sur un entier, la valeur est le pixel.
//
// Partagé entre l'hôte et l'appareil : le test sur processeur compare cette formule à la référence en double
// de kernels/ref/warp.cpp, le test GPU compare le noyau à cette même référence.
#pragma once

#include "kernels/spec/chroma_up.h"     // AJP_SPEC_HOST_DEVICE, AJP_SPEC_DEVICE
#include "kernels/spec/fp_off_begin.h"  // IWYU pragma: keep

namespace ajp::kernels {

// Comment le warp lit ses deux images.
enum class WarpImages : unsigned char { Bilinear, Bicubic };

// Le filtre des images w × h pour un mouvement estimé dans une fenêtre fw × fh : bicubique dès que l'image
// est plus grande, sinon bilinéaire (à taille égale, SPAN passe après et nettoie).
constexpr WarpImages warp_images(int w, int h, int fw, int fh) {
  return w > fw || h > fh ? WarpImages::Bicubic : WarpImages::Bilinear;
}

// Les quatre poids de Catmull-Rom pour la fraction f de [0, 1[ : prises −1, 0, 1, 2. Leur somme vaut 1.
AJP_SPEC_HOST_DEVICE constexpr void warp_cubic_weights(float f, float* w) {
  const float f2 = f * f;
  const float f3 = f2 * f;
  w[0] = -0.5F * f3 + f2 - 0.5F * f;
  w[1] = 1.5F * f3 - 2.5F * f2 + 1.0F;
  w[2] = -1.5F * f3 + 2.0F * f2 + 0.5F * f;
  w[3] = 0.5F * f3 - 0.5F * f2;
}

AJP_SPEC_HOST_DEVICE constexpr int warp_clamp_index(int i, int n) {
  if (i < 0) return 0;
  return i >= n ? n - 1 : i;
}

// L'échantillon en (x, y) d'un plan w × h ; fetch(i, j) rend le pixel (i, j), toujours dans le plan.
template <typename Fetch>
AJP_SPEC_DEVICE constexpr float warp_bicubic_at(float x, float y, int w, int h, const Fetch& fetch) {
  const float xmax = static_cast<float>(w - 1);
  const float ymax = static_cast<float>(h - 1);
  x = !(x > 0.0F) ? 0.0F : (x > xmax ? xmax : x);  // NaN donne 0
  y = !(y > 0.0F) ? 0.0F : (y > ymax ? ymax : y);
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  float wx[4] = {};
  float wy[4] = {};
  warp_cubic_weights(x - static_cast<float>(x0), wx);
  warp_cubic_weights(y - static_cast<float>(y0), wy);
  float value = 0.0F;
  float low = 0.0F;
  float high = 0.0F;
  for (int j = 0; j < 4; ++j) {
    const int yy = warp_clamp_index(y0 - 1 + j, h);
    float row = 0.0F;
    for (int i = 0; i < 4; ++i) {
      const float p = fetch(warp_clamp_index(x0 - 1 + i, w), yy);
      row += wx[i] * p;
      if ((i == 1 || i == 2) && (j == 1 || j == 2)) {
        const bool first = i == 1 && j == 1;
        low = first || p < low ? p : low;
        high = first || p > high ? p : high;
      }
    }
    value += wy[j] * row;
  }
  return value < low ? low : (value > high ? high : value);
}

}  // namespace ajp::kernels

#include "kernels/spec/fp_off_end.h"  // IWYU pragma: keep
