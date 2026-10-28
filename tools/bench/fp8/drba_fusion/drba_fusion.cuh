// Prototype : noyaux FP16 maison pour tout ce que DistilDRBA v2 lite (échelles 16, 8, 4) fait hors
// convolution.
//
// Les convolutions restent dans quatre moteurs TensorRT : l'encodeur (lot de 3 images au quart de la
// résolution) et les blocs 0, 1, 2 (entrée = Concat, sortie = 52 canaux de ConvTranspose avant DepthToSpace).
// Six noyaux font le reste :
//   pre   : réductions d'images 2 × 2 (au quart, au huitième, au seizième) et plans du temps,
//   post  : réductions 2 × 2 de l'encodage au demi et au quart de son échelle,
//   colle : DepthToSpace + Resize ×2 de la sortie du bloc précédent, flot ×2, grilles, warps des images et
//           des encodages (GridSample), qui complètent l'entrée du bloc suivant,
//   fin   : DepthToSpace + Resize ×4, flot ×4, sigmoïde du masque, warps pleine résolution, mélange.
//
// Les entrées X1 et X2 des blocs sont remplies en place par plusieurs producteurs (disposition ONNX) :
//   0-8 images 0, 1, 2 réduites, 9-56 encodages f0, f1, f2 réduits, 57-59 image 1 warpée, 60-62 image du
//   côté choisi warpée, 63-78 f1 warpé, 79-94 f du côté choisi warpé, 95 temps, 96 masque, 97-104
//   attributs, 105-108 flot.  X0 (58 canaux, 68 × 120) : 0-8 images, 9-56 encodages, 57 temps.
// Les tenseurs d'entrée de l'encodeur (3 × 3 plans) et de sortie (3 × 16 plans) sont adressés DANS X2
// (plans 0-8 et 9-56) : aucune copie de ces 57 canaux, et X2 est complet après la colle 2.
//
// Le côté choisi (img0 ou img2 selon tau) vient du Clip du graphe, lu sur le périphérique (tau change à
// chaque image) ; mélange du graphe = sélection exacte (le coefficient vaut 0 ou 1).
//
// Les arrondis FP16 du graphe (sortie des Resize, Mul, Add, Cast, Sigmoid...) sont reproduits aux mêmes
// endroits ; les Resize et GridSample du graphe sont en FP32 (îlots entre deux Cast).
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace df {

constexpr int kCanauxX = 109;  // blocs 1 et 2
constexpr int kCanauxX0 = 58;
constexpr int kPlanTemps = 95;

__device__ __forceinline__ float arrondi(float v) {
  return __half2float(__float2half_rn(v));
}

__device__ __forceinline__ float lire(const __half* p, std::size_t i) {
  return __half2float(p[i]);
}

// Le côté choisi du graphe : vrai si tau < 1 (img0), faux sinon (img2). tau = 2 × plan 6 de pair01.
__device__ __forceinline__ bool cote_avant(const __half* pair01, std::size_t plan) {
  const float tau = 2.0F * __half2float(pair01[6 * plan]);
  const float c = fminf(fmaxf((1.0F - tau) * 4096.0F, 0.0F), 1.0F);
  return c >= 0.5F;
}

// Sortie 52 canaux d'un bloc ; Z = DepthToSpace(CRD, 2)(Y) de zh × zw.
struct SourceZ {
  const __half* y;
  int zh, zw;
};

// DepthToSpace lu à la volée : Z[c][zy][zx] = Y[4c + 2(zy&1) + (zx&1)][zy>>1][zx>>1].
__device__ __forceinline__ float lire_z(const SourceZ& z, int c, int zy, int zx) {
  const std::size_t canal = static_cast<std::size_t>(4 * c + 2 * (zy & 1) + (zx & 1));
  return __half2float(z.y[(canal * static_cast<std::size_t>(z.zh / 2) + static_cast<std::size_t>(zy >> 1)) *
                              static_cast<std::size_t>(z.zw / 2) +
                          static_cast<std::size_t>(zx >> 1)]);
}

// Resize bilinéaire half_pixel d'un facteur s (puissance de 2) de Z (canaux c0..c0+N-1) au pixel (px, py) du
// plan agrandi ; arrondi FP16 comme la sortie du Cast qui suit le Resize FP32.
template <int N>
__device__ __forceinline__ void agrandir(const SourceZ& z, int c0, int s, int px, int py, float (&r)[N]) {
  const float inv = 1.0F / static_cast<float>(s);
  float sy = (static_cast<float>(py) + 0.5F) * inv - 0.5F;
  float sx = (static_cast<float>(px) + 0.5F) * inv - 0.5F;
  sy = fminf(fmaxf(sy, 0.0F), static_cast<float>(z.zh - 1));
  sx = fminf(fmaxf(sx, 0.0F), static_cast<float>(z.zw - 1));
  const int y0 = static_cast<int>(sy);
  const int x0 = static_cast<int>(sx);
  const int y1 = min(y0 + 1, z.zh - 1);
  const int x1 = min(x0 + 1, z.zw - 1);
  const float wy = sy - static_cast<float>(y0);
  const float wx = sx - static_cast<float>(x0);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const float a = lire_z(z, c0 + i, y0, x0);
    const float b = lire_z(z, c0 + i, y0, x1);
    const float c = lire_z(z, c0 + i, y1, x0);
    const float e = lire_z(z, c0 + i, y1, x1);
    r[i] = arrondi((a * (1.0F - wx) + b * wx) * (1.0F - wy) + (c * (1.0F - wx) + e * wx) * wy);
  }
}

// Grille normalisée du graphe : base FP16 (i × 2/(n-1) - 1) + flot FP16 × (2/(n-1) FP16), deux sommes FP16.
struct PasGrille {
  float c32;  // 2/(n-1) en FP32 (calcul de la base)
  float c16;  // le même arrondi en FP16 (échelle du flot)
};

__device__ __forceinline__ float grille(int i, float flot, PasGrille g) {
  const float base = arrondi(__fsub_rn(__fmul_rn(static_cast<float>(i), g.c32), 1.0F));
  return arrondi(base + arrondi(flot * g.c16));
}

// GridSample (bilinéaire, border, align_corners = 1) : coordonnées depuis la grille normalisée.
struct Coord {
  int i00, i01, i10, i11;  // indices dans un plan h × w
  float wx, wy;
};

__device__ __forceinline__ Coord coord_grille(float gx, float gy, int w, int h) {
  float ix = (gx + 1.0F) * static_cast<float>(w - 1) * 0.5F;
  float iy = (gy + 1.0F) * static_cast<float>(h - 1) * 0.5F;
  ix = fminf(fmaxf(ix, 0.0F), static_cast<float>(w - 1));
  iy = fminf(fmaxf(iy, 0.0F), static_cast<float>(h - 1));
  const int x0 = static_cast<int>(ix);
  const int y0 = static_cast<int>(iy);
  const int x1 = min(x0 + 1, w - 1);
  const int y1 = min(y0 + 1, h - 1);
  Coord c;
  c.i00 = y0 * w + x0;
  c.i01 = y0 * w + x1;
  c.i10 = y1 * w + x0;
  c.i11 = y1 * w + x1;
  c.wx = ix - static_cast<float>(x0);
  c.wy = iy - static_cast<float>(y0);
  return c;
}

__device__ __forceinline__ float bilineaire(const __half* plan, const Coord& c) {
  const float a = __half2float(plan[c.i00]);
  const float b = __half2float(plan[c.i01]);
  const float d = __half2float(plan[c.i10]);
  const float e = __half2float(plan[c.i11]);
  return (a * (1.0F - c.wx) + b * c.wx) * (1.0F - c.wy) + (d * (1.0F - c.wx) + e * c.wx) * c.wy;
}

// ---------------------------------------------------------------------------------------------------------
// Noyau pre : sections d'une même grille 1D.
//   a) images au quart -> X2[0..8]   b) au huitième -> X1[0..8]   c) au seizième -> X0[0..8]
//   d) plans du temps : X0[57], X1[95], X2[95]
struct ParamsPre {
  int h, w;
  const __half* pair01;
  const __half* pair12;
  __half* x0;
  __half* x1;
  __half* x2;
};

// Plan c (0..8) des trois images : img0 = pair01[0..2], img1 = pair01[3..5], img2 = pair12[3..5].
__device__ __forceinline__ const __half* plan_image(const ParamsPre& p, int c) {
  const std::size_t plan = static_cast<std::size_t>(p.h) * p.w;
  const int img = c / 3;
  const int k = c % 3;
  if (img == 0) return p.pair01 + k * plan;
  if (img == 1) return p.pair01 + (3 + k) * plan;
  return p.pair12 + (3 + k) * plan;
}

__global__ void __launch_bounds__(256) noyau_pre(ParamsPre p) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t plan = static_cast<std::size_t>(p.h) * p.w;
  // Tailles des sections (éléments).
  const int w4 = p.w / 4, h4 = p.h / 4, w8 = p.w / 8, h8 = p.h / 8, w16 = p.w / 16, h16 = p.h / 16;
  const std::size_t n4 = 9ull * h4 * w4, n8 = 9ull * h8 * w8, n16 = 9ull * h16 * w16;
  const std::size_t nt4 = static_cast<std::size_t>(h4) * w4, nt8 = static_cast<std::size_t>(h8) * w8,
                    nt16 = static_cast<std::size_t>(h16) * w16;
  std::size_t j = i;
  int d, ow, oh;
  __half* sortie;
  if (j < n4) {
    d = 4, ow = w4, oh = h4, sortie = p.x2;
  } else if (j < n4 + n8) {
    j -= n4, d = 8, ow = w8, oh = h8, sortie = p.x1;
  } else if (j < n4 + n8 + n16) {
    j -= n4 + n8, d = 16, ow = w16, oh = h16, sortie = p.x0;
  } else {
    j -= n4 + n8 + n16;
    // Plans du temps : u × 2, un pixel sur s de pair01[6].
    const __half* u = p.pair01 + 6 * plan;
    std::size_t n;
    int s, w2;
    __half* dst;
    if (j < nt4) {
      s = 4, w2 = w4, n = j, dst = p.x2 + kPlanTemps * nt4;
    } else if (j < nt4 + nt8) {
      j -= nt4, s = 8, w2 = w8, n = j, dst = p.x1 + kPlanTemps * nt8;
    } else if (j < nt4 + nt8 + nt16) {
      j -= nt4 + nt8, s = 16, w2 = w16, n = j, dst = p.x0 + 57 * nt16;
    } else {
      return;
    }
    const int oy = static_cast<int>(n / w2);
    const int ox = static_cast<int>(n % w2);
    dst[n] = __float2half_rn(2.0F * __half2float(u[static_cast<std::size_t>(s * oy) * p.w + s * ox]));
    return;
  }
  const std::size_t oplan = static_cast<std::size_t>(oh) * ow;
  const int c = static_cast<int>(j / oplan);
  const std::size_t r = j % oplan;
  const int oy = static_cast<int>(r / ow);
  const int ox = static_cast<int>(r % ow);
  // Resize d'échelle 1/d (half_pixel, linéaire) = moyenne 2 × 2 des pixels d·o + d/2 - 1 et +1.
  const int base = d / 2 - 1;
  const __half* src = plan_image(p, c);
  const std::size_t i00 = static_cast<std::size_t>(d * oy + base) * p.w + (d * ox + base);
  const float v = (__half2float(src[i00]) + __half2float(src[i00 + 1]) + __half2float(src[i00 + p.w]) +
                   __half2float(src[i00 + p.w + 1])) *
                  0.25F;
  sortie[static_cast<std::size_t>(c) * oplan + r] = __float2half_rn(v);
}

// ---------------------------------------------------------------------------------------------------------
// Noyau post : réductions 2 × 2 des trois encodages (sortie de l'encodeur, X2[9..56], 272 × 480) :
// au demi -> X1[9..56] (moyenne des pixels 2o et 2o+1), au quart de l'échelle -> X0[9..56] (4o+1 et 4o+2).
struct ParamsPost {
  int h4, w4;  // dimensions de l'encodage
  const __half* x2;
  __half* x1;
  __half* x0;
};

__global__ void __launch_bounds__(256) noyau_post(ParamsPost p) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int h2 = p.h4 / 2, w2 = p.w4 / 2, h1 = p.h4 / 4, w1 = p.w4 / 4;
  const std::size_t o2 = static_cast<std::size_t>(h2) * w2, o1 = static_cast<std::size_t>(h1) * w1;
  const std::size_t plan4 = static_cast<std::size_t>(p.h4) * p.w4;
  const std::size_t n2 = 48 * o2, n1 = 48 * o1;
  std::size_t j = i;
  int d, ow;
  std::size_t oplan;
  __half* sortie;
  if (j < n2) {
    d = 2, ow = w2, oplan = o2, sortie = p.x1 + 9 * o2;
  } else if (j < n2 + n1) {
    j -= n2, d = 4, ow = w1, oplan = o1, sortie = p.x0 + 9 * o1;
  } else {
    return;
  }
  const int c = static_cast<int>(j / oplan);
  const std::size_t r = j % oplan;
  const int oy = static_cast<int>(r / ow);
  const int ox = static_cast<int>(r % ow);
  const int base = d / 2 - 1;
  const __half* src = p.x2 + (9 + static_cast<std::size_t>(c)) * plan4;
  const std::size_t i00 = static_cast<std::size_t>(d * oy + base) * p.w4 + (d * ox + base);
  const float v = (__half2float(src[i00]) + __half2float(src[i00 + 1]) + __half2float(src[i00 + p.w4]) +
                   __half2float(src[i00 + p.w4 + 1])) *
                  0.25F;
  sortie[static_cast<std::size_t>(c) * oplan + r] = __float2half_rn(v);
}

// ---------------------------------------------------------------------------------------------------------
// Noyau colle : complète l'entrée Xk (k = 1, 2) : canaux 57-94 (warps), 96-104 (masque, attributs), 105-108
// (flot) à partir de la sortie du bloc précédent (agrandie ×2). Les canaux 0-56 et 95 sont déjà là.
struct ParamsColle {
  int h, w;   // résolution de Xk
  SourceZ z;  // DepthToSpace de la sortie du bloc précédent (zh = h/2)
  const __half* pair01;
  std::size_t plan_pleine;  // H × W de pair01 (lecture de tau)
  __half* x;
  PasGrille gx, gy;
};

__global__ void __launch_bounds__(128) noyau_colle(ParamsColle p) {
  const int ox = blockIdx.x * blockDim.x + threadIdx.x;
  const int oy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ox >= p.w || oy >= p.h) return;
  const std::size_t plan = static_cast<std::size_t>(p.h) * p.w;
  const std::size_t o = static_cast<std::size_t>(oy) * p.w + ox;
  const bool avant = cote_avant(p.pair01, p.plan_pleine);

  // Flot (canaux 105-108) : Resize ×2 arrondi FP16, puis × 2 (exact).
  float fl[4];
  {
    float r[4];
    agrandir<4>(p.z, 0, 2, ox, oy, r);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      fl[k] = arrondi(r[k] * 2.0F);
      p.x[(105 + k) * plan + o] = __float2half_rn(fl[k]);
    }
  }
  {
    float r[9];
    agrandir<9>(p.z, 4, 2, ox, oy, r);
#pragma unroll
    for (int k = 0; k < 9; ++k) p.x[(96 + k) * plan + o] = __float2half_rn(r[k]);
  }
  // Warps : n = 0 image 1 et f1 avec le flot 0-1 ; n = 1 image et f du côté choisi avec le flot 2-3.
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const Coord c = coord_grille(grille(ox, fl[2 * n], p.gx), grille(oy, fl[2 * n + 1], p.gy), p.w, p.h);
    const int img = n == 0 ? 3 : (avant ? 0 : 6);
    const int enc = n == 0 ? 25 : (avant ? 9 : 41);
#pragma unroll
    for (int k = 0; k < 3; ++k)
      p.x[(57 + 3 * n + k) * plan + o] = __float2half_rn(bilineaire(p.x + (img + k) * plan, c));
    for (int k = 0; k < 16; ++k)
      p.x[(63 + 16 * n + k) * plan + o] = __float2half_rn(bilineaire(p.x + (enc + k) * plan, c));
  }
}

// ---------------------------------------------------------------------------------------------------------
// Noyau fin : sortie du bloc 2 agrandie ×4 (flot × 4, masque), warps pleine résolution de img1 et de l'image
// du côté choisi, sigmoïde, mélange.
struct ParamsFin {
  int h, w;
  SourceZ z;
  const __half* pair01;
  const __half* pair12;
  __half* out;
  PasGrille gx, gy;
};

__global__ void __launch_bounds__(256) noyau_fin(ParamsFin p) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= p.w || y >= p.h) return;
  const std::size_t plan = static_cast<std::size_t>(p.h) * p.w;
  const std::size_t o = static_cast<std::size_t>(y) * p.w + x;
  const bool avant = cote_avant(p.pair01, plan);
  float r[5];
  agrandir<5>(p.z, 0, 4, x, y, r);
  float fl[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) fl[k] = arrondi(r[k] * 4.0F);
  const float sig = arrondi(1.0F / (1.0F + expf(-r[4])));
  const float un_moins = arrondi(1.0F - sig);
  float w2[2][3];
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const Coord c = coord_grille(grille(x, fl[2 * n], p.gx), grille(y, fl[2 * n + 1], p.gy), p.w, p.h);
    const __half* src = n == 0 ? p.pair01 + 3 * plan : (avant ? p.pair01 : p.pair12 + 3 * plan);
#pragma unroll
    for (int k = 0; k < 3; ++k) w2[n][k] = arrondi(bilineaire(src + k * plan, c));
  }
#pragma unroll
  for (int k = 0; k < 3; ++k)
    p.out[k * plan + o] = __float2half_rn(arrondi(w2[0][k] * sig) + arrondi(w2[1][k] * un_moins));
}

}  // namespace df
