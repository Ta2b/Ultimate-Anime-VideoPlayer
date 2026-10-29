// Noyaux de RIFE fusionné (voir rife_fused_kernels.h).
//
// Disposition des 28 canaux d'entrée des blocs 1 et suivants (celle du graphe ONNX,
// kernels/spec/rife_fused.h) :
//   0-2 image A warpée, 3-5 image B warpée, 6-9 encodage de A warpé, 10-13 encodage de B warpé, 14 instant,
//   15 masque, 16-23 attributs, 24-27 flot accumulé × (1/d).
// Bloc 0 : 15 canaux (images A, B, encodages de A et de B, instant), tous moyennés sans warp.
#include <cstddef>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/rife_fused_kernels.h"

namespace ajp::gpu::cuda {
namespace {

__device__ __forceinline__ float rounded(float v) {
  return __half2float(__float2half_rn(v));
}

// DepthToSpace (CRD, 2) lu à la volée : Z[c][Y][X] = Y[4c + 2 (Y & 1) + (X & 1)][Y >> 1][X >> 1].
__device__ __forceinline__ float read_z(const __half* y, int yh, int yw, int c, int zy, int zx) {
  const auto channel = static_cast<std::size_t>(4 * c + 2 * (zy & 1) + (zx & 1));
  return __half2float(y[(channel * static_cast<std::size_t>(yh) + static_cast<std::size_t>(zy >> 1)) *
                            static_cast<std::size_t>(yw) +
                        static_cast<std::size_t>(zx >> 1)]);
}

// Resize bilinéaire demi-pixel de facteur s de Z (canaux c0 .. c0 + N - 1) au pixel pleine résolution
// (px, py), arrondi FP16 comme la sortie du Resize du graphe.
template <int N>
__device__ __forceinline__ void enlarge(const FusedFlowSource& f, int c0, int px, int py, float (&r)[N]) {
  const __half* y = static_cast<const __half*>(f.y);
  const int zh = 2 * f.height;
  const int zw = 2 * f.width;
  const float inv = 1.0F / static_cast<float>(f.scale);
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
    const float a = read_z(y, f.height, f.width, c0 + i, y0, x0);
    const float b = read_z(y, f.height, f.width, c0 + i, y0, x1);
    const float c = read_z(y, f.height, f.width, c0 + i, y1, x0);
    const float e = read_z(y, f.height, f.width, c0 + i, y1, x1);
    r[i] = rounded((a * (1.0F - wx) + b * wx) * (1.0F - wy) + (c * (1.0F - wx) + e * wx) * wy);
  }
}

// GridSample (coins alignés, bords répétés) : coordonnées depuis la grille normalisée.
struct Tap {
  int x0, x1, y0, y1;
  float wx, wy;
};

__device__ __forceinline__ Tap grid_tap(float gx, float gy, int w, int h) {
  float ix = (gx + 1.0F) * 0.5F * static_cast<float>(w - 1);
  float iy = (gy + 1.0F) * 0.5F * static_cast<float>(h - 1);
  ix = fminf(fmaxf(ix, 0.0F), static_cast<float>(w - 1));
  iy = fminf(fmaxf(iy, 0.0F), static_cast<float>(h - 1));
  Tap t;
  t.x0 = static_cast<int>(ix);
  t.y0 = static_cast<int>(iy);
  t.x1 = min(t.x0 + 1, w - 1);
  t.y1 = min(t.y0 + 1, h - 1);
  t.wx = ix - static_cast<float>(t.x0);
  t.wy = iy - static_cast<float>(t.y0);
  return t;
}

__device__ __forceinline__ float bilinear(const __half* plane, const Tap& t, int w) {
  const std::size_t stride = static_cast<std::size_t>(w);
  const float a =
      __half2float(plane[static_cast<std::size_t>(t.y0) * stride + static_cast<std::size_t>(t.x0)]);
  const float b =
      __half2float(plane[static_cast<std::size_t>(t.y0) * stride + static_cast<std::size_t>(t.x1)]);
  const float d =
      __half2float(plane[static_cast<std::size_t>(t.y1) * stride + static_cast<std::size_t>(t.x0)]);
  const float e =
      __half2float(plane[static_cast<std::size_t>(t.y1) * stride + static_cast<std::size_t>(t.x1)]);
  return (a * (1.0F - t.wx) + b * t.wx) * (1.0F - t.wy) + (d * (1.0F - t.wx) + e * t.wx) * t.wy;
}

// Bloc 0 : 15 canaux = moyenne 2 × 2 (ou d × d centrés) des images, de l'encodage et de l'instant.
__global__ void __launch_bounds__(256) fused_entry(FusedEntryParams p) {
  const int ox = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int oy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (ox >= p.out_width || oy >= p.out_height) return;
  const std::size_t plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  const std::size_t out_plane =
      static_cast<std::size_t>(p.out_height) * static_cast<std::size_t>(p.out_width);
  const int base = p.reduce / 2 - 1;
  const int px = p.reduce * ox + base;
  const int py = p.reduce * oy + base;
  const std::size_t i00 =
      static_cast<std::size_t>(py) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(px);
  const auto* input = static_cast<const __half*>(p.input);
  const auto* encode = static_cast<const __half*>(p.encode);
  auto* x = static_cast<__half*>(p.x);
#pragma unroll
  for (int c = 0; c < 15; ++c) {
    const __half* src = c < 6
                            ? input + static_cast<std::size_t>(c) * plane
                            : (c < 14 ? encode + static_cast<std::size_t>(c - 6) * plane : input + 6 * plane);
    const std::size_t w = static_cast<std::size_t>(p.width);
    const float v = (__half2float(src[i00]) + __half2float(src[i00 + 1]) + __half2float(src[i00 + w]) +
                     __half2float(src[i00 + w + 1])) *
                    0.25F;
    x[static_cast<std::size_t>(c) * out_plane +
      static_cast<std::size_t>(oy) * static_cast<std::size_t>(p.out_width) + static_cast<std::size_t>(ox)] =
        __float2half_rn(v);
  }
}

// Colle d'un bloc : l'entrée (28 canaux) du bloc suivant.
// Quatre blocs par SM (128 registres) suffisent à sm_75 à sm_89 ; sm_120 en demande davantage et déborde
// sinon (ptxas --warn-on-spills est une erreur de la compilation de livraison).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define AJP_GLUE_MIN_BLOCKS 2
#else
#define AJP_GLUE_MIN_BLOCKS 4
#endif
__global__ void __launch_bounds__(128, AJP_GLUE_MIN_BLOCKS) fused_glue(FusedGlueParams p) {
  const int ox = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int oy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (ox >= p.out_width || oy >= p.out_height) return;
  const std::size_t plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  const std::size_t out_plane =
      static_cast<std::size_t>(p.out_height) * static_cast<std::size_t>(p.out_width);
  const auto* input = static_cast<const __half*>(p.input);
  const auto* encode = static_cast<const __half*>(p.encode);
  const int taps = p.reduce > 1 ? 2 : 1;
  const int base = p.reduce > 1 ? p.reduce / 2 - 1 : 0;
  const float flow_scale = 1.0F / static_cast<float>(p.reduce);
  float acc[28];
#pragma unroll
  for (int c = 0; c < 28; ++c) acc[c] = 0.0F;

  for (int ty = 0; ty < taps; ++ty) {
    for (int tx = 0; tx < taps; ++tx) {
      const int px = p.reduce * ox + base + tx;
      const int py = p.reduce * oy + base + ty;
      const std::size_t i =
          static_cast<std::size_t>(py) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(px);

      // Flot accumulé (Add : sommes FP16 de chaque bloc, multiplié par l'échelle de son Resize).
      float fl[4];
      {
        float r[4];
        enlarge<4>(p.source[0], 0, px, py, r);
#pragma unroll
        for (int k = 0; k < 4; ++k) fl[k] = r[k] * static_cast<float>(p.source[0].scale);
      }
#pragma unroll
      for (int s = 1; s < 4; ++s) {
        if (s < p.flows) {
          float r[4];
          enlarge<4>(p.source[static_cast<std::size_t>(s)], 0, px, py, r);
#pragma unroll
          for (int k = 0; k < 4; ++k)
            fl[k] = rounded(fl[k] + r[k] * static_cast<float>(p.source[static_cast<std::size_t>(s)].scale));
        }
      }

      // Warp des deux images (7 canaux chacune : 3 de l'image, 4 de l'encodage).
      const float gx0 = __half2float(input[7 * plane + i]);
      const float gy0 = __half2float(input[8 * plane + i]);
      const float sx = __half2float(input[9 * plane + i]);
      const float sy = __half2float(input[10 * plane + i]);
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const float gx = __fadd_rn(gx0, __fmul_rn(fl[2 * n], sx));
        const float gy = __fadd_rn(gy0, __fmul_rn(fl[2 * n + 1], sy));
        const Tap tap = grid_tap(gx, gy, p.width, p.height);
#pragma unroll
        for (int ch = 0; ch < 7; ++ch) {
          const __half* src = ch < 3 ? input + static_cast<std::size_t>(3 * n + ch) * plane
                                     : encode + static_cast<std::size_t>(4 * n + ch - 3) * plane;
          const int dst = ch < 3 ? 3 * n + ch : 6 + 4 * n + ch - 3;
          acc[dst] += rounded(bilinear(src, tap, p.width));
        }
      }
      acc[14] += __half2float(input[6 * plane + i]);
      // Masque et attributs du dernier bloc calculé.
      {
        float r[9];
        enlarge<9>(p.source[static_cast<std::size_t>(p.flows - 1)], 4, px, py, r);
#pragma unroll
        for (int k = 0; k < 9; ++k) acc[15 + k] += r[k];
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[24 + k] += rounded(fl[k] * flow_scale);
    }
  }
  const float norm = 1.0F / static_cast<float>(taps * taps);
  const std::size_t o =
      static_cast<std::size_t>(oy) * static_cast<std::size_t>(p.out_width) + static_cast<std::size_t>(ox);
  auto* x = static_cast<__half*>(p.x);
#pragma unroll
  for (int c = 0; c < 28; ++c)
    x[static_cast<std::size_t>(c) * out_plane + o] = __float2half_rn(acc[c] * norm);
}

// Fin : flot du dernier bloc ajouté ; image (warp, masque sigmoïde, mélange) ou flot et masque.
__global__ void __launch_bounds__(256) fused_finish(FusedFinishParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x);
  const auto* input = static_cast<const __half*>(p.input);
  const auto* last_x = static_cast<const __half*>(p.x_last);
  const auto* last_y = static_cast<const __half*>(p.y_last);
  float fl[4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
    fl[k] = rounded(__half2float(last_x[static_cast<std::size_t>(24 + k) * plane + i]) +
                    read_z(last_y, p.y_height, p.y_width, k, y, x));
  const float mask = read_z(last_y, p.y_height, p.y_width, 4, y, x);
  const float sig = rounded(1.0F / (1.0F + __expf(-mask)));
  if (p.image == nullptr) {  // variante mouvement : le flot final et le masque
    auto* flow = static_cast<__half*>(p.flow);
    auto* out_mask = static_cast<__half*>(p.mask);
#pragma unroll
    for (int k = 0; k < 4; ++k) flow[static_cast<std::size_t>(k) * plane + i] = __float2half_rn(fl[k]);
    out_mask[i] = __float2half_rn(sig);
    return;
  }
  const float gx0 = __half2float(input[7 * plane + i]);
  const float gy0 = __half2float(input[8 * plane + i]);
  const float sx = __half2float(input[9 * plane + i]);
  const float sy = __half2float(input[10 * plane + i]);
  float warped[2][3];
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const float gx = __fadd_rn(gx0, __fmul_rn(fl[2 * n], sx));
    const float gy = __fadd_rn(gy0, __fmul_rn(fl[2 * n + 1], sy));
    const Tap tap = grid_tap(gx, gy, p.width, p.height);
#pragma unroll
    for (int ch = 0; ch < 3; ++ch)
      warped[n][ch] = rounded(bilinear(input + static_cast<std::size_t>(3 * n + ch) * plane, tap, p.width));
  }
  const float rest = rounded(1.0F - sig);
  auto* image = static_cast<__half*>(p.image);
#pragma unroll
  for (int ch = 0; ch < 3; ++ch)
    image[static_cast<std::size_t>(ch) * plane + i] =
        __float2half_rn(rounded(warped[0][ch] * sig) + rounded(warped[1][ch] * rest));
}

dim3 grid_for(int width, int height, dim3 block) {
  return dim3((static_cast<unsigned>(width) + block.x - 1U) / block.x,
              (static_cast<unsigned>(height) + block.y - 1U) / block.y);
}

}  // namespace

cudaError_t launch_fused_entry(const FusedEntryParams& params, cudaStream_t stream) {
  const dim3 block(32, 8);
  fused_entry<<<grid_for(params.out_width, params.out_height, block), block, 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_fused_glue(const FusedGlueParams& params, cudaStream_t stream) {
  const dim3 block(32, 4);
  fused_glue<<<grid_for(params.out_width, params.out_height, block), block, 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_fused_finish(const FusedFinishParams& params, cudaStream_t stream) {
  const dim3 block(32, 8);
  fused_finish<<<grid_for(params.width, params.height, block), block, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
