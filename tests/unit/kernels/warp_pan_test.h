// Scène de test du mouvement d'ensemble (kernels/spec/warp_pan.h), partagée par la référence et les backends
// : une texture lisse, b la même décalée de (kPanDx, kPanDy) px, et un flot de RIFE uniforme égal au
// mouvement d'ensemble au milieu (le remplacement n'y frôle aucun seuil).
#pragma once

#include <cmath>

#include "kernels/image.h"
#include "kernels/ref/warp.h"

namespace kirari::test {

inline constexpr float kPanDx = 0.6F;
inline constexpr float kPanDy = 0.2F;

inline float pan_texture(float x, float y, int c) {
  const float fc = static_cast<float>(c);
  return 0.5F + 0.3F * std::sin(0.31F * x + 0.17F * y + fc) +
         0.15F * std::cos(0.13F * x - 0.29F * y + 2.0F * fc);
}

// Image w × h du contenu décalé de (dx, dy) : chaque point lit la texture en (x − dx, y − dy).
inline kernels::RgbImage pan_image(int w, int h, float dx, float dy) {
  kernels::RgbImage image(w, h);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        image.planes[static_cast<std::size_t>(c)].at(x, y) =
            pan_texture(static_cast<float>(x) - dx, static_cast<float>(y) - dy, c);
      }
    }
  }
  return image;
}

// Flot uniforme (f0x, f0y, f1x, f1y) et masque 1/2 sur des plans w × h, fenêtre entière.
inline kernels::ref::Motion pan_motion(int w, int h, float f0x, float f0y, float f1x, float f1y) {
  kernels::ref::Motion m{.flow = {kernels::PlaneF(w, h, f0x), kernels::PlaneF(w, h, f0y),
                                  kernels::PlaneF(w, h, f1x), kernels::PlaneF(w, h, f1y)},
                         .mask = kernels::PlaneF(w, h, 0.5F),
                         .area = {0, 0, w, h}};
  return m;
}

}  // namespace kirari::test
