// Lecture bilinéaire des tenseurs FP16 planaires des noyaux pixel (bilinear_at de ref/warp.cpp) :
// coordonnées relatives au coin d'une fenêtre et bornées à elle. Deux lectures du même calcul :
// Exact, quatre chargements et poids en float (le chemin par défaut, et le repli quand le tenseur
// n'a pas d'objet texture), et Filtered, une prise de l'objet texture au filtre bilinéaire
// matériel (étude 3) : la fraction y est sur 8 bits, l'erreur au plus le gradient local / 512. Filtered ne
// sert qu'à l'essai --filtered-warp (BackendInit::filtered_warp), tant que son gain n'est pas mesuré.
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

  // Les canaux first à first + N − 1 au même point : la position, les voisins et les poids calculés une
  // fois, puis le même calcul que sample pour chaque plan (le résultat est celui de N appels).
  template <int N>
  __device__ static void sample_n(const Planes& t, int first, float x, float y, float (&out)[N]) {
    x = fminf(fmaxf(x, 0.0F), static_cast<float>(t.fw - 1));
    y = fminf(fmaxf(y, 0.0F), static_cast<float>(t.fh - 1));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = min(x0 + 1, t.fw - 1);
    const int y1 = min(y0 + 1, t.fh - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const std::size_t w = static_cast<std::size_t>(t.plane_w);
    const std::size_t plane = static_cast<std::size_t>(t.plane_h) * w;
    const std::size_t corner = static_cast<std::size_t>(t.oy) * w + static_cast<std::size_t>(t.ox);
    const std::size_t r0 = corner + static_cast<std::size_t>(y0) * w;
    const std::size_t r1 = corner + static_cast<std::size_t>(y1) * w;
#pragma unroll
    for (int k = 0; k < N; ++k) {
      const __half* p = t.data + static_cast<std::size_t>(first + k) * plane;
      const float top = __half2float(p[r0 + static_cast<std::size_t>(x0)]) * (1.0F - fx) +
                        __half2float(p[r0 + static_cast<std::size_t>(x1)]) * fx;
      const float bottom = __half2float(p[r1 + static_cast<std::size_t>(x0)]) * (1.0F - fx) +
                           __half2float(p[r1 + static_cast<std::size_t>(x1)]) * fx;
      out[k] = top * (1.0F - fy) + bottom * fy;
    }
  }
};

// Catmull-Rom (kernels/spec/warp.h : RifeWarp), quatre × quatre chargements en float : position bornée à la
// fenêtre, points répétés au bord, sortie bornée à [0, 1]. Pour les images seulement (le warp final) ;
// toujours lue exactement, même quand le tenseur a un objet texture.
struct CatmullRom {
  __device__ static void weights(float t, float (&w)[4]) {
    const float t2 = t * t;
    const float t3 = t2 * t;
    w[0] = -0.5F * t3 + t2 - 0.5F * t;
    w[1] = 1.5F * t3 - 2.5F * t2 + 1.0F;
    w[2] = -1.5F * t3 + 2.0F * t2 + 0.5F * t;
    w[3] = 0.5F * t3 - 0.5F * t2;
  }
  __device__ static float sample(const Planes& t, int c, float x, float y) {
    x = fminf(fmaxf(x, 0.0F), static_cast<float>(t.fw - 1));
    y = fminf(fmaxf(y, 0.0F), static_cast<float>(t.fh - 1));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    float wx[4];
    float wy[4];
    weights(x - static_cast<float>(x0), wx);
    weights(y - static_cast<float>(y0), wy);
    const __half* p =
        t.data +
        (static_cast<std::size_t>(c) * static_cast<std::size_t>(t.plane_h) + static_cast<std::size_t>(t.oy)) *
            static_cast<std::size_t>(t.plane_w) +
        static_cast<std::size_t>(t.ox);
    float sum = 0.0F;
    float lo = 3.0e38F;  // la sortie reste entre le plus petit et le plus grand des quatre points centraux
    float hi = -3.0e38F;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const auto yy = static_cast<std::size_t>(min(max(y0 - 1 + j, 0), t.fh - 1));
      float row = 0.0F;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const auto xx = static_cast<std::size_t>(min(max(x0 - 1 + i, 0), t.fw - 1));
        const float v = __half2float(p[yy * static_cast<std::size_t>(t.plane_w) + xx]);
        row += v * wx[i];
        if ((i == 1 || i == 2) && (j == 1 || j == 2)) {
          lo = fminf(lo, v);
          hi = fmaxf(hi, v);
        }
      }
      sum += row * wy[j];
    }
    return fminf(fmaxf(sum, lo), hi);
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

  template <int N>
  __device__ static void sample_n(const Planes& t, int first, float x, float y, float (&out)[N]) {
#pragma unroll
    for (int k = 0; k < N; ++k) out[k] = sample(t, first + k, x, y);
  }
};

}  // namespace ajp::gpu::cuda
