// Prototype : noyaux FP16 maison pour tout ce que RIFE v4.25 fait en pleine résolution.
//
// Les convolutions basse résolution restent dans six moteurs TensorRT (encode + blocs 0 à 4) ; entre eux,
// un seul noyau par bloc remplace : DepthToSpace + Resize bilinéaire ×k + mise à l'échelle du flot +
// accumulation + grille + GridSample + Cast + Split/Reshape/Concat + Resize d'entrée du bloc suivant.
//
// Le Resize d'entrée du bloc suivant (échelle 1/d, half_pixel, linéaire) est une moyenne exacte de 2 × 2
// pixels pleine résolution (d >= 2) ou l'identité (d = 1) : le noyau n'évalue donc le flot et le warp que
// sur les pixels que le bloc suivant lira (4/64 des pixels pour d = 8, 4/16 pour d = 4), puis moyenne.
// Rien de pleine résolution n'est écrit hors du tenseur d'entrée du dernier bloc.
//
// Disposition des 28 canaux d'entrée des blocs 1 à 4 (celle du graphe ONNX) :
//   0-2 image A warpée, 3-5 image B warpée, 6-9 encode A warpé, 10-13 encode B warpé, 14 temps,
//   15 masque, 16-23 attributs, 24-27 flot accumulé × (1/d).
// Bloc 0 : 15 canaux (images A, B, encode A, encode B, temps), tous moyennés sans warp.
//
// Les arrondis FP16 du graphe (sortie du Resize, Mul, Add, Cast, Sigmoid...) sont reproduits aux mêmes
// endroits pour que l'écart au graphe ne vienne que de l'ordre de sommation.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rf {

constexpr int kNbBlocs = 5;

// Une source de flot : sortie 52 canaux d'un bloc (avant DepthToSpace), Z = DepthToSpace(Y) de zh × zw,
// agrandie ×s vers la pleine résolution.
struct SourceFlot {
  const __half* y;
  int yh;  // hauteur de Y (52 canaux) = zh / 2
  int yw;
  int s;  // facteur d'agrandissement du Resize de sortie (8, 8, 4, 2)
};

struct ParamsColle {
  int h, w;  // pleine résolution
  int d;     // réduction du Resize d'entrée du bloc suivant (8, 4, 2, 1)
  int oh, ow;
  int nflot;  // nombre de sources de flot (1 à 4) ; la dernière donne aussi masque et attributs
  SourceFlot f[4];
  float echelle_flot;  // 1/d
  const __half* in;    // entrée du réseau : 11 plans h × w
  const __half* enc;   // sortie du moteur encode : 8 plans h × w (A puis B)
  __half* x;           // entrée du bloc suivant
};

struct ParamsEntree {
  int h, w, d, oh, ow;
  const __half* in;
  const __half* enc;
  __half* x;
};

struct ParamsFin {
  int h, w;
  const __half* in;
  const __half* enc;
  const __half* x4;  // entrée du bloc 4 (canaux 24-27 = flot accumulé des blocs 0 à 3)
  const __half* y4;  // sortie 52 canaux du bloc 4 (384 × 640)
  int y4h, y4w;
  __half* out;  // 3 plans h × w
};

__device__ __forceinline__ float arrondi(float v) {
  return __half2float(__float2half_rn(v));
}

// DepthToSpace(CRD, 2) lu à la volée : Z[c][Y][X] = Y[4c + 2(Y&1) + (X&1)][Y>>1][X>>1].
__device__ __forceinline__ float lire_z(const __half* y, int yh, int yw, int c, int zy, int zx) {
  const std::size_t canal = static_cast<std::size_t>(4 * c + 2 * (zy & 1) + (zx & 1));
  return __half2float(
      y[(canal * static_cast<std::size_t>(yh) + static_cast<std::size_t>(zy >> 1)) * yw + (zx >> 1)]);
}

// Resize bilinéaire half_pixel d'un facteur s de Z (canaux c0..c0+n-1) au pixel pleine résolution (px, py),
// arrondi FP16 comme la sortie du Resize du graphe.
template <int N>
__device__ __forceinline__ void agrandir(const SourceFlot& f, int c0, int px, int py, float (&r)[N]) {
  const int zh = 2 * f.yh;
  const int zw = 2 * f.yw;
  const float inv = 1.0F / static_cast<float>(f.s);
  float sy = (static_cast<float>(py) + 0.5F) * inv - 0.5F;
  float sx = (static_cast<float>(px) + 0.5F) * inv - 0.5F;
  sy = fminf(fmaxf(sy, 0.0F), static_cast<float>(zh - 1));
  sx = fminf(fmaxf(sx, 0.0F), static_cast<float>(zw - 1));
  const int y0 = static_cast<int>(sy);
  const int x0 = static_cast<int>(sx);
  const int y1 = min(y0 + 1, zh - 1);
  const int x1 = min(x0 + 1, zw - 1);
  const float wy = sy - static_cast<float>(y0);
  const float wx = sx - static_cast<float>(x0);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const float a = lire_z(f.y, f.yh, f.yw, c0 + i, y0, x0);
    const float b = lire_z(f.y, f.yh, f.yw, c0 + i, y0, x1);
    const float c = lire_z(f.y, f.yh, f.yw, c0 + i, y1, x0);
    const float e = lire_z(f.y, f.yh, f.yw, c0 + i, y1, x1);
    r[i] = arrondi((a * (1.0F - wx) + b * wx) * (1.0F - wy) + (c * (1.0F - wx) + e * wx) * wy);
  }
}

// Coordonnées de GridSample (align_corners = 1, padding border) depuis la grille normalisée.
struct Coord {
  int x0, x1, y0, y1;
  float wx, wy;
};

__device__ __forceinline__ Coord coord_grille(float gx, float gy, int w, int h) {
  float ix = (gx + 1.0F) * 0.5F * static_cast<float>(w - 1);
  float iy = (gy + 1.0F) * 0.5F * static_cast<float>(h - 1);
  ix = fminf(fmaxf(ix, 0.0F), static_cast<float>(w - 1));
  iy = fminf(fmaxf(iy, 0.0F), static_cast<float>(h - 1));
  Coord c;
  c.x0 = static_cast<int>(ix);
  c.y0 = static_cast<int>(iy);
  c.x1 = min(c.x0 + 1, w - 1);
  c.y1 = min(c.y0 + 1, h - 1);
  c.wx = ix - static_cast<float>(c.x0);
  c.wy = iy - static_cast<float>(c.y0);
  return c;
}

__device__ __forceinline__ float bilineaire(const __half* plan, const Coord& c, int w) {
  const float a = __half2float(plan[c.y0 * w + c.x0]);
  const float b = __half2float(plan[c.y0 * w + c.x1]);
  const float d = __half2float(plan[c.y1 * w + c.x0]);
  const float e = __half2float(plan[c.y1 * w + c.x1]);
  return (a * (1.0F - c.wx) + b * c.wx) * (1.0F - c.wy) + (d * (1.0F - c.wx) + e * c.wx) * c.wy;
}

// Noyau d'entrée : bloc 0, 15 canaux = moyenne 2 × 2 (d = 8) des images, de l'encode et du temps.
__global__ void __launch_bounds__(256) noyau_entree(ParamsEntree p) {
  const int ox = blockIdx.x * blockDim.x + threadIdx.x;
  const int oy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ox >= p.ow || oy >= p.oh) return;
  const std::size_t plan = static_cast<std::size_t>(p.h) * p.w;
  const std::size_t oplan = static_cast<std::size_t>(p.oh) * p.ow;
  const int base = p.d / 2 - 1;
  const int px = p.d * ox + base;
  const int py = p.d * oy + base;
  const std::size_t i00 = static_cast<std::size_t>(py) * p.w + px;
#pragma unroll
  for (int c = 0; c < 15; ++c) {
    const __half* src = c < 6 ? p.in + c * plan : (c < 14 ? p.enc + (c - 6) * plan : p.in + 6 * plan);
    const float v = (__half2float(src[i00]) + __half2float(src[i00 + 1]) + __half2float(src[i00 + p.w]) +
                     __half2float(src[i00 + p.w + 1])) *
                    0.25F;
    p.x[c * oplan + static_cast<std::size_t>(oy) * p.ow + ox] = __float2half_rn(v);
  }
}

// Noyau de colle d'un bloc : produit l'entrée (28 canaux) du bloc suivant.
__global__ void __launch_bounds__(128, 4) noyau_colle(ParamsColle p) {
  const int ox = blockIdx.x * blockDim.x + threadIdx.x;
  const int oy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ox >= p.ow || oy >= p.oh) return;
  const std::size_t plan = static_cast<std::size_t>(p.h) * p.w;
  const std::size_t oplan = static_cast<std::size_t>(p.oh) * p.ow;
  const int ntap = p.d > 1 ? 2 : 1;
  const int base = p.d > 1 ? p.d / 2 - 1 : 0;
  float acc[28];
#pragma unroll
  for (int c = 0; c < 28; ++c) acc[c] = 0.0F;

  for (int ty = 0; ty < ntap; ++ty) {
    for (int tx = 0; tx < ntap; ++tx) {
      const int px = p.d * ox + base + tx;
      const int py = p.d * oy + base + ty;
      const std::size_t i = static_cast<std::size_t>(py) * p.w + px;

      // Flot accumulé (Add_1, Add_3, Add_5 : sommes FP16).
      float fl[4];
      {
        float r[4];
        agrandir<4>(p.f[0], 0, px, py, r);
#pragma unroll
        for (int k = 0; k < 4; ++k) fl[k] = r[k] * static_cast<float>(p.f[0].s);
      }
#pragma unroll
      for (int s = 1; s < 4; ++s) {
        if (s < p.nflot) {
          float r[4];
          agrandir<4>(p.f[s], 0, px, py, r);
#pragma unroll
          for (int k = 0; k < 4; ++k) fl[k] = arrondi(fl[k] + r[k] * static_cast<float>(p.f[s].s));
        }
      }

      // Warp des deux images (7 canaux chacune : 3 de l'image, 4 de l'encode).
      const float gx0 = __half2float(p.in[7 * plan + i]);
      const float gy0 = __half2float(p.in[8 * plan + i]);
      const float sx = __half2float(p.in[9 * plan + i]);
      const float sy = __half2float(p.in[10 * plan + i]);
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const float gx = __fadd_rn(gx0, __fmul_rn(fl[2 * n], sx));
        const float gy = __fadd_rn(gy0, __fmul_rn(fl[2 * n + 1], sy));
        const Coord c = coord_grille(gx, gy, p.w, p.h);
#pragma unroll
        for (int ch = 0; ch < 7; ++ch) {
          const __half* src = ch < 3 ? p.in + (3 * n + ch) * plan : p.enc + (4 * n + ch - 3) * plan;
          const int dst = ch < 3 ? 3 * n + ch : 6 + 4 * n + ch - 3;
          acc[dst] += arrondi(bilineaire(src, c, p.w));
        }
      }
      acc[14] += __half2float(p.in[6 * plan + i]);
      // Masque et attributs du dernier bloc.
      {
        float r[9];
        agrandir<9>(p.f[p.nflot - 1], 4, px, py, r);
#pragma unroll
        for (int k = 0; k < 9; ++k) acc[15 + k] += r[k];
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[24 + k] += arrondi(fl[k] * p.echelle_flot);
    }
  }
  const float norme = 1.0F / static_cast<float>(ntap * ntap);
  const std::size_t o = static_cast<std::size_t>(oy) * p.ow + ox;
#pragma unroll
  for (int c = 0; c < 28; ++c) p.x[c * oplan + o] = __float2half_rn(acc[c] * norme);
}

// Noyau final : flot du bloc 4 ajouté, warp des deux images, masque sigmoïde, mélange, sortie 3 canaux.
__global__ void __launch_bounds__(256) noyau_fin(ParamsFin p) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= p.w || y >= p.h) return;
  const std::size_t plan = static_cast<std::size_t>(p.h) * p.w;
  const std::size_t i = static_cast<std::size_t>(y) * p.w + x;
  float fl[4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
    fl[k] = arrondi(__half2float(p.x4[(24 + k) * plan + i]) + lire_z(p.y4, p.y4h, p.y4w, k, y, x));
  const float masque = lire_z(p.y4, p.y4h, p.y4w, 4, y, x);
  const float sig = arrondi(1.0F / (1.0F + __expf(-masque)));
  const float gx0 = __half2float(p.in[7 * plan + i]);
  const float gy0 = __half2float(p.in[8 * plan + i]);
  const float sx = __half2float(p.in[9 * plan + i]);
  const float sy = __half2float(p.in[10 * plan + i]);
  float w2[2][3];
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const float gx = __fadd_rn(gx0, __fmul_rn(fl[2 * n], sx));
    const float gy = __fadd_rn(gy0, __fmul_rn(fl[2 * n + 1], sy));
    const Coord c = coord_grille(gx, gy, p.w, p.h);
#pragma unroll
    for (int ch = 0; ch < 3; ++ch) w2[n][ch] = arrondi(bilineaire(p.in + (3 * n + ch) * plan, c, p.w));
  }
  const float un_moins = arrondi(1.0F - sig);
#pragma unroll
  for (int ch = 0; ch < 3; ++ch)
    p.out[ch * plan + i] = __float2half_rn(arrondi(w2[0][ch] * sig) + arrondi(w2[1][ch] * un_moins));
}

}  // namespace rf
