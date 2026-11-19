// Les briques de périphérique de TemporalFix fusionné (tfix_fused_kernels.cu, tfix_blend_kernels.cu),
// au-dessus de la bibliothèque de warp (warp_lib.cuh) : le Resize du graphe (bilinéaire demi-pixel, tailles
// quelconques), le GridSample à zéros dehors, la sigmoïde.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "gpu/cuda/backend/tfix_fused_kernels.h"
#include "gpu/cuda/backend/warp_lib.cuh"

namespace kirari::gpu::cuda::tfix {

using warp::Axis;

// Le Resize linéaire demi-pixel du graphe d'une grille de n_in points vers n_out, le long d'un axe : les
// deux voisins et le poids du point p (position bornée à [0, n_in − 1], comme onnxruntime et TensorRT).
__device__ __forceinline__ Axis resize_axis(int p, int n_in, int n_out) {
  return warp::coarse_axis(p, static_cast<float>(n_in) / static_cast<float>(n_out), n_in);
}

__device__ __forceinline__ float at(const __half* plane, int y, int x, int width) {
  return __half2float(plane[static_cast<long long>(y) * width + x]);
}

// Le Resize d'un plan h × w lu au point (y, x) de la grille d'arrivée, dont les axes sont donnés.
__device__ __forceinline__ float resized(const __half* plane, int width, const Axis& ax, const Axis& ay) {
  return warp::lerp2(at(plane, ay.i0, ax.i0, width), at(plane, ay.i0, ax.i1, width),
                     at(plane, ay.i1, ax.i0, width), at(plane, ay.i1, ax.i1, width), ax, ay);
}

// GridSample bilinéaire, align_corners 0, zéros dehors : les quatre voisins de (px, py) en pixels et leurs
// poids, nuls pour un voisin hors de l'image (indice ramené à 0). valid : la somme des poids (le GridSample
// d'un plan de uns).
struct ZeroTap {
  int index[4];
  float weight[4];
  float valid;
};

__device__ __forceinline__ ZeroTap zero_tap(float px, float py, int height, int width) {
  // Bornée : au-delà, tous les voisins sont dehors ; un flot NaN ou démesuré n'atteint pas la conversion.
  px = fminf(fmaxf(px, -2.0F), static_cast<float>(width) + 1.0F);
  py = fminf(fmaxf(py, -2.0F), static_cast<float>(height) + 1.0F);
  const float fx = floorf(px);
  const float fy = floorf(py);
  const int x0 = static_cast<int>(fx);
  const int y0 = static_cast<int>(fy);
  const float ax = px - fx;
  const float ay = py - fy;
  ZeroTap t;
  t.valid = 0.0F;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int x = x0 + (i & 1);
    const int y = y0 + (i >> 1);
    const bool inside = x >= 0 && x < width && y >= 0 && y < height;
    const float w = ((i & 1) != 0 ? ax : 1.0F - ax) * ((i >> 1) != 0 ? ay : 1.0F - ay);
    t.index[i] = inside ? y * width + x : 0;
    t.weight[i] = inside ? w : 0.0F;
    t.valid += t.weight[i];
  }
  return t;
}

__device__ __forceinline__ float zero_sample(const __half* plane, const ZeroTap& t) {
  float sum = 0.0F;
#pragma unroll
  for (int i = 0; i < 4; ++i) sum += t.weight[i] * __half2float(plane[t.index[i]]);
  return sum;
}

__device__ __forceinline__ float sigmoid(float x) {
  return 1.0F / (1.0F + __expf(-x));
}

// Le flou binomial 5 × 5 suivi de la réduction exacte d'un facteur 2 (le niveau 2) revient à un seul noyau
// séparable de six points, (1 4 6 4 1 0 + 0 1 4 6 4 1) / 32, sur les lignes et colonnes 2 p − 2 à 2 p + 3.
__device__ __forceinline__ float half_kernel(int i) {
  constexpr float kWeights[6] = {1.0F / 32, 5.0F / 32, 10.0F / 32, 10.0F / 32, 5.0F / 32, 1.0F / 32};
  return kWeights[i];
}

__device__ __forceinline__ const __half* half_ptr(const void* p, std::size_t offset = 0) {
  return static_cast<const __half*>(p) + offset;
}
__device__ __forceinline__ __half* half_ptr(void* p, std::size_t offset = 0) {
  return static_cast<__half*>(p) + offset;
}

// L'image du réseau de la position r de la fenêtre : l'image de travail, ou en demi-taille l'image moyennée
// gardée.
__device__ __forceinline__ const __half* net_image(const TfixShape& s, int r) {
  return s.half ? half_ptr(s.table->packed[r], s.image) : half_ptr(s.table->rgb[r]);
}

// Le niveau l de la pyramide de la position r de la fenêtre (7 plans : image puis traits).
__device__ __forceinline__ const __half* pyramid(const TfixShape& s, int r, int level) {
  return half_ptr(s.table->packed[r], s.pyramid[level]);
}

}  // namespace kirari::gpu::cuda::tfix
