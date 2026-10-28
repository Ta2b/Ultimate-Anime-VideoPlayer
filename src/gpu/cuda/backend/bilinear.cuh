// Lecture bilinéaire des tenseurs FP16 planaires des noyaux pixel (bilinear_at de ref/warp.cpp) :
// coordonnées relatives au coin d'une fenêtre et bornées à elle. Deux lectures du même calcul :
// Exact, quatre chargements et poids en float (le chemin de référence, et le repli quand le tenseur
// n'a pas d'objet texture), et Filtered, une prise de l'objet texture au filtre bilinéaire
// matériel (étude 3) : la fraction y est sur 8 bits, l'erreur au plus le gradient local / 512.
// Bornée à la fenêtre avant la prise, la texture ne lit jamais hors d'elle : au bord, la fraction
// est nulle et le voisin hors fenêtre pèse 0.
#pragma once

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ajp::gpu::cuda {

// La fenêtre fw × fh au coin (ox, oy) des plans plane_w × plane_h d'un tenseur
// 1 × C × plane_h × plane_w ; texture : le même tenseur vu en plane_w × (C · plane_h), ou 0.
struct Planes {
  const __half* data = nullptr;
  cudaTextureObject_t texture = 0;
  int plane_w = 0;
  int plane_h = 0;
  int ox = 0;
  int oy = 0;
  int fw = 0;
  int fh = 0;
};

struct Exact {
  __device__ static float sample(const Planes& t, int c, float x, float y) {
    x = fminf(fmaxf(x, 0.0F), static_cast<float>(t.fw - 1));
    y = fminf(fmaxf(y, 0.0F), static_cast<float>(t.fh - 1));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = min(x0 + 1, t.fw - 1);
    const int y1 = min(y0 + 1, t.fh - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const __half* p =
        t.data +
        (static_cast<std::size_t>(c) * static_cast<std::size_t>(t.plane_h) + static_cast<std::size_t>(t.oy)) *
            static_cast<std::size_t>(t.plane_w) +
        static_cast<std::size_t>(t.ox);
    const auto at = [&](int xx, int yy) {
      return __half2float(p[static_cast<std::size_t>(yy) * static_cast<std::size_t>(t.plane_w) +
                            static_cast<std::size_t>(xx)]);
    };
    const float top = at(x0, y0) * (1.0F - fx) + at(x1, y0) * fx;
    const float bottom = at(x0, y1) * (1.0F - fx) + at(x1, y1) * fx;
    return top * (1.0F - fy) + bottom * fy;
  }
};

struct Filtered {
  // Le centre du texel i est en i + 1/2.
  __device__ static float sample(const Planes& t, int c, float x, float y) {
    x = fminf(fmaxf(x, 0.0F), static_cast<float>(t.fw - 1));
    y = fminf(fmaxf(y, 0.0F), static_cast<float>(t.fh - 1));
    return tex2D<float>(t.texture, static_cast<float>(t.ox) + x + 0.5F,
                        static_cast<float>(c * t.plane_h + t.oy) + y + 0.5F);
  }
};

}  // namespace ajp::gpu::cuda
