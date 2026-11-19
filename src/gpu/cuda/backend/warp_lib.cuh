// Bibliothèque de warp CUDA (lot E, docs/warp-lib.md) : ce qui déplace des images par un flot, écrit une fois
// pour tous ses clients. RIFE fusionné (rife_fused_kernels.cu) s'en sert aujourd'hui ; TemporalFix (lot B),
// la composition par calques et le décalage des tuiles la reprendront.
//
// Trois briques, toutes en fonctions de périphérique :
//   - la prise (Tap) : position x + flot en FP32, bornée à une fenêtre, puis lecture bilinéaire ou
//   Catmull-Rom
//     d'un plan FP16 ;
//   - le flot à n'importe quelle échelle : un tenseur de flot plus petit que l'image (sortie d'un bloc de
//   RIFE
//     en DepthToSpace, flot NVOF en grille 4, carte de caméra), agrandi en bilinéaire demi-pixel aux seuls
//     points lus ; ou un mouvement affine (caméra) évalué au point ;
//   - le relèvement par bloc : la région d'un tenseur grossier que lit un bloc de fils, copiée une fois en
//     mémoire partagée par des lectures de 16 octets, puis lue là par chaque fil. C'est ce qui ôte la plus
//     grande part du coût de l'ancienne colle de RIFE : chaque fil y relisait les flots de tous les blocs
//     précédents par des lectures de 2 octets éparses.
//
// La réduction d'un facteur D (Resize 1/D demi-pixel du graphe de RIFE) s'évalue aux points échantillonnés :
// la moyenne des 2 × 2 pixels pleine résolution D x + D/2 − 1 + {0, 1}, jamais une image réduite d'abord.
//
// Méthode reprise de la description publique de libaji (rassemblement à l'échelle du bloc, warp aux points
// d'échantillonnage, spécialisation par forme) ; aucune ligne de son code (sans licence).
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kirari::gpu::cuda::warp {

__device__ __forceinline__ float rounded(float v) {
  return __half2float(__float2half_rn(v));
}

// La fenêtre lisible d'une image : [left, right[ × [top, bottom[.
struct Bounds {
  int left, top, right, bottom;
};

// Les quatre voisins et les poids d'une lecture bilinéaire à (x + fx, y + fy), bornée à la fenêtre (bords
// répétés). Somme en __fadd_rn : le même arrondi que la grille FP32 de RIFE (A03-20).
struct Tap {
  int x0, x1, y0, y1;
  float wx, wy;
};

__device__ __forceinline__ Tap tap_at(int x, int y, float fx, float fy, const Bounds& b) {
  float ix = __fadd_rn(static_cast<float>(x), fx);
  float iy = __fadd_rn(static_cast<float>(y), fy);
  ix = fminf(fmaxf(ix, static_cast<float>(b.left)), static_cast<float>(b.right - 1));
  iy = fminf(fmaxf(iy, static_cast<float>(b.top)), static_cast<float>(b.bottom - 1));
  Tap t;
  t.x0 = static_cast<int>(ix);
  t.y0 = static_cast<int>(iy);
  t.x1 = min(t.x0 + 1, b.right - 1);
  t.y1 = min(t.y0 + 1, b.bottom - 1);
  t.wx = ix - static_cast<float>(t.x0);
  t.wy = iy - static_cast<float>(t.y0);
  return t;
}

// Lecture bilinéaire d'un plan FP16 de largeur stride.
__device__ __forceinline__ float bilinear(const __half* plane, const Tap& t, int stride) {
  const __half* r0 = plane + t.y0 * stride;
  const __half* r1 = plane + t.y1 * stride;
  const float a = __half2float(r0[t.x0]);
  const float b = __half2float(r0[t.x1]);
  const float d = __half2float(r1[t.x0]);
  const float e = __half2float(r1[t.x1]);
  return (a * (1.0F - t.wx) + b * t.wx) * (1.0F - t.wy) + (d * (1.0F - t.wx) + e * t.wx) * t.wy;
}

__device__ __forceinline__ void catmull_rom_weights(float t, float (&w)[4]) {
  const float t2 = t * t;
  const float t3 = t2 * t;
  w[0] = -0.5F * t3 + t2 - 0.5F * t;
  w[1] = 1.5F * t3 - 2.5F * t2 + 1.0F;
  w[2] = -1.5F * t3 + 2.0F * t2 + 0.5F * t;
  w[3] = 0.5F * t3 - 0.5F * t2;
}

// Catmull-Rom (kernels/spec/warp.h : RifeWarp) sur 4 × 4 points de la fenêtre, bords répétés, sortie bornée
// aux quatre points centraux (anti-ringing local, C50-08).
__device__ __forceinline__ float catmull_rom(const __half* plane, const Tap& t, int stride, const Bounds& b) {
  float wx[4];
  float wy[4];
  catmull_rom_weights(t.wx, wx);
  catmull_rom_weights(t.wy, wy);
  float sum = 0.0F;
  float lo = 3.0e38F;
  float hi = -3.0e38F;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __half* row = plane + min(max(t.y0 - 1 + j, b.top), b.bottom - 1) * stride;
    float acc = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float v = __half2float(row[min(max(t.x0 - 1 + i, b.left), b.right - 1)]);
      acc += v * wx[i];
      if ((i == 1 || i == 2) && (j == 1 || j == 2)) {
        lo = fminf(lo, v);
        hi = fmaxf(hi, v);
      }
    }
    sum += acc * wy[j];
  }
  return fminf(fmaxf(sum, lo), hi);
}

// Agrandissement bilinéaire demi-pixel d'un facteur s (Resize du graphe) : la position dans la grille
// grossière de n points d'un point pleine résolution p, bornée.
__device__ __forceinline__ float coarse_at(int p, float inv_scale, int n) {
  const float s = (static_cast<float>(p) + 0.5F) * inv_scale - 0.5F;
  return fminf(fmaxf(s, 0.0F), static_cast<float>(n - 1));
}

// Les deux voisins grossiers et le poids le long d'un axe.
struct Axis {
  int i0, i1;
  float w;
};

__device__ __forceinline__ Axis coarse_axis(int p, float inv_scale, int n) {
  const float s = coarse_at(p, inv_scale, n);
  Axis a;
  a.i0 = static_cast<int>(s);
  a.i1 = min(a.i0 + 1, n - 1);
  a.w = s - static_cast<float>(a.i0);
  return a;
}

__device__ __forceinline__ float lerp2(float a, float b, float c, float e, const Axis& x, const Axis& y) {
  return (a * (1.0F - x.w) + b * x.w) * (1.0F - y.w) + (c * (1.0F - x.w) + e * x.w) * y.w;
}

// La région [lo, hi] (bornes comprises) d'une grille grossière de n points que lisent les points pleine
// résolution [p_lo, p_hi] agrandis d'un facteur 1 / inv_scale : la même arithmétique que coarse_axis, donc
// exacte.
struct Span {
  int lo, hi;
};

__device__ __forceinline__ Span coarse_span(int p_lo, int p_hi, float inv_scale, int n) {
  return {static_cast<int>(coarse_at(p_lo, inv_scale, n)),
          min(static_cast<int>(coarse_at(p_hi, inv_scale, n)) + 1, n - 1)};
}

// Un tenseur grossier relevé en mémoire partagée : une région rows × cols de pixels, chaque pixel portant
// stride demi-flottants (les canaux first .. first + stride − 1 de la source, dans l'ordre).
struct Staged {
  const __half* data;
  int row0, col0, cols, stride;
  __device__ __forceinline__ const __half* at(int row, int col) const {
    return data + ((row - row0) * cols + (col - col0)) * stride;
  }
};

// Relève dans dst la région [rows] × [cols] des canaux first .. first + count − 1 d'un tenseur FP16 de
// height × width pixels, rangé en plans (pitch 0) ou par pixel (pitch canaux, multiple de 8) ; dst garde
// stride canaux par pixel (multiple de 8, au moins count). Par pixel, first multiple de 8, lecture par 16
// octets. Tous les fils du bloc participent ; l'appelant synchronise ensuite.
__device__ __forceinline__ void stage_region(__half* dst, int stride, const __half* src, int height,
                                             int width, int pitch, int first, int count, Span rows,
                                             Span cols) {
  const int nr = rows.hi - rows.lo + 1;
  const int nc = cols.hi - cols.lo + 1;
  const int threads = static_cast<int>(blockDim.x * blockDim.y);
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  if (pitch > 0) {
    const int chunks = (count + 7) / 8;
    const int total = nr * nc * chunks;
    for (int i = tid; i < total; i += threads) {
      const int q = i % chunks;
      const int pixel = i / chunks;
      const int r = rows.lo + pixel / nc;
      const int c = cols.lo + pixel % nc;
      const auto* from = reinterpret_cast<const uint4*>(
          src + (static_cast<std::int64_t>(r) * width + c) * pitch + first + 8 * q);
      *reinterpret_cast<uint4*>(dst + pixel * stride + 8 * q) = *from;
    }
    return;
  }
  const std::int64_t plane = static_cast<std::int64_t>(height) * width;
  const int total = nr * nc * count;
  for (int i = tid; i < total; i += threads) {
    const int pixel = i % (nr * nc);
    const int ch = i / (nr * nc);
    const int r = rows.lo + pixel / nc;
    const int c = cols.lo + pixel % nc;
    dst[pixel * stride + ch] = src[(first + ch) * plane + static_cast<std::int64_t>(r) * width + c];
  }
}

// Somme des 2 × 2 points d'échantillonnage d'une réduction, portés par les fils lane, lane ^ 1, lane ^ 2 et
// lane ^ 3 d'un même warp : chacun en ressort avec la somme.
__device__ __forceinline__ float sum_quad(float v) {
  v += __shfl_xor_sync(0xFFFFFFFFU, v, 1);
  v += __shfl_xor_sync(0xFFFFFFFFU, v, 2);
  return v;
}

}  // namespace kirari::gpu::cuda::warp
