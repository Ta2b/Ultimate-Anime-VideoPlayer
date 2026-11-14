// Noyaux de RIFE fusionné (voir rife_fused_kernels.h).
//
// Disposition des 28 canaux d'entrée des blocs 1 et suivants (celle du graphe ONNX,
// kernels/spec/rife_fused.h) :
//   0-2 image A warpée, 3-5 image B warpée, 6-9 encodage de A warpé, 10-13 encodage de B warpé, 14 instant,
//   15 masque, 16-23 attributs, 24-27 flot accumulé × (1/d).
// Bloc 0 : 15 canaux (images A, B, encodages de A et de B, instant), tous moyennés sans warp.
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/rife_fused_kernels.h"

namespace ajp::gpu::cuda {
namespace {

__device__ __forceinline__ float rounded(float v) {
  return __half2float(__float2half_rn(v));
}

// L'élément (c, pixel) d'un tenseur de plane pixels : en plans (pitch 0) ou par pixel (pitch canaux).
__device__ __forceinline__ std::size_t element(int pitch, std::size_t plane, int c, std::size_t pixel) {
  return pitch == 0 ? static_cast<std::size_t>(c) * plane + pixel
                    : pixel * static_cast<std::size_t>(pitch) + static_cast<std::size_t>(c);
}

// DepthToSpace (CRD, 2) lu à la volée : Z[c][Y][X] = Y[4c + 2 (Y & 1) + (X & 1)][Y >> 1][X >> 1].
__device__ __forceinline__ float read_z(const __half* y, int yh, int yw, int pitch, int c, int zy, int zx) {
  const int channel = 4 * c + 2 * (zy & 1) + (zx & 1);
  const std::size_t pixel =
      static_cast<std::size_t>(zy >> 1) * static_cast<std::size_t>(yw) + static_cast<std::size_t>(zx >> 1);
  return __half2float(
      y[element(pitch, static_cast<std::size_t>(yh) * static_cast<std::size_t>(yw), channel, pixel)]);
}

// Les C valeurs d'un pixel de X, arrondies en FP16, en plans ou par pixel (pitch 0 ou P). Par pixel, les
// canaux C à P - 1 sont écrits à zéro (les poids que TensorRT y applique sont nuls, mais un NaN resté en
// mémoire passerait) et le pixel part en P / 8 écritures de 16 octets.
template <int C, int P>
__device__ __forceinline__ void store_x(__half* x, int pitch, std::size_t plane, std::size_t pixel,
                                        const float (&v)[C]) {
  if (pitch == 0) {
#pragma unroll
    for (int c = 0; c < C; ++c) x[static_cast<std::size_t>(c) * plane + pixel] = __float2half_rn(v[c]);
    return;
  }
  static_assert(P % 8 == 0 && P >= C, "P : multiple de 8, au moins C");
  std::uint32_t word[P / 2];
#pragma unroll
  for (int i = 0; i < P / 2; ++i) {
    const __half2 pair = __floats2half2_rn(2 * i < C ? v[2 * i] : 0.0F, 2 * i + 1 < C ? v[2 * i + 1] : 0.0F);
    word[i] = *reinterpret_cast<const std::uint32_t*>(&pair);
  }
  auto* out = reinterpret_cast<uint4*>(x + pixel * P);
#pragma unroll
  for (int j = 0; j < P / 8; ++j)
    out[j] = make_uint4(word[4 * j], word[4 * j + 1], word[4 * j + 2], word[4 * j + 3]);
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
    const float a = read_z(y, f.height, f.width, f.pitch, c0 + i, y0, x0);
    const float b = read_z(y, f.height, f.width, f.pitch, c0 + i, y0, x1);
    const float c = read_z(y, f.height, f.width, f.pitch, c0 + i, y1, x0);
    const float e = read_z(y, f.height, f.width, f.pitch, c0 + i, y1, x1);
    r[i] = rounded((a * (1.0F - wx) + b * wx) * (1.0F - wy) + (c * (1.0F - wx) + e * wx) * wy);
  }
}

// GridSample (coins alignés, bords répétés) depuis la position en pixels. La grille normalisée du graphe,
// base + flot × pas ramenée par (g + 1) / 2 × (n - 1), vaut exactement x + flot : calculée ainsi, en FP32,
// elle ne passe pas par le maillage de l'entrée v1 (plans 7 à 10), que le FP16 arrondit jusqu'à 0,16 px à
// 1280 de large et 0,23 px à 1920 (A03-20). Le fusionné s'écarte donc exprès du moteur entier v1, dont la
// grille reste celle de son entrée FP16.
struct Tap {
  int x0, x1, y0, y1;
  float wx, wy;
};

__device__ __forceinline__ Tap pixel_tap(int x, int y, float flow_x, float flow_y, int w, int h) {
  float ix = __fadd_rn(static_cast<float>(x), flow_x);
  float iy = __fadd_rn(static_cast<float>(y), flow_y);
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

// Catmull-Rom (kernels/spec/rife_fused.h : RifeWarp) sur 4 × 4 points en FP32 : la position du Tap (déjà
// ramenée dans l'image), points répétés au bord, sortie bornée aux quatre points centraux (anti-ringing
// local, C50-08).
__device__ __forceinline__ void catmull_rom(float t, float (&w)[4]) {
  const float t2 = t * t;
  const float t3 = t2 * t;
  w[0] = -0.5F * t3 + t2 - 0.5F * t;
  w[1] = 1.5F * t3 - 2.5F * t2 + 1.0F;
  w[2] = -1.5F * t3 + 2.0F * t2 + 0.5F * t;
  w[3] = 0.5F * t3 - 0.5F * t2;
}

// La fenêtre de l'image source dans le tenseur rempli : [left, right[ × [top, bottom[.
struct Bounds {
  int left, top, right, bottom;
};

// pixel_tap borné à la fenêtre de l'image au lieu du tenseur rempli (C03-03) : le warp final ne lit jamais
// la bordure noire du remplissage, comme le warp après SPAN (bilinear.cuh).
__device__ __forceinline__ Tap window_tap(int x, int y, float flow_x, float flow_y, const Bounds& b) {
  float ix = __fadd_rn(static_cast<float>(x), flow_x);
  float iy = __fadd_rn(static_cast<float>(y), flow_y);
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

__device__ __forceinline__ float catmull_rom_sample(const __half* plane, const Tap& t, int w,
                                                    const Bounds& b) {
  float wx[4];
  float wy[4];
  catmull_rom(t.wx, wx);
  catmull_rom(t.wy, wy);
  const std::size_t stride = static_cast<std::size_t>(w);
  float sum = 0.0F;
  float lo = 3.0e38F;
  float hi = -3.0e38F;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const auto yy = static_cast<std::size_t>(min(max(t.y0 - 1 + j, b.top), b.bottom - 1));
    float row = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const auto xx = static_cast<std::size_t>(min(max(t.x0 - 1 + i, b.left), b.right - 1));
      const float v = __half2float(plane[yy * stride + xx]);
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
  const auto* encode_b = static_cast<const __half*>(p.encode_b);
  const std::size_t w = static_cast<std::size_t>(p.width);
  float v[15];
#pragma unroll
  for (int c = 0; c < 15; ++c) {
    const __half* src = c < 6    ? input + static_cast<std::size_t>(c) * plane
                        : c < 10 ? encode + static_cast<std::size_t>(c - 6) * plane
                        : c < 14 ? encode_b + static_cast<std::size_t>(c - 10) * plane
                                 : input + 6 * plane;
    if (c == 14) {  // l'instant : l'élément 0 du plan 6 seul est tenu (étape 5.2, rife_time)
      v[c] = __half2float(src[0]);
      continue;
    }
    v[c] = (__half2float(src[i00]) + __half2float(src[i00 + 1]) + __half2float(src[i00 + w]) +
            __half2float(src[i00 + w + 1])) *
           0.25F;
  }
  store_x<15, 16>(
      static_cast<__half*>(p.x), p.x_pitch, out_plane,
      static_cast<std::size_t>(oy) * static_cast<std::size_t>(p.out_width) + static_cast<std::size_t>(ox), v);
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
  const auto* encode_b = static_cast<const __half*>(p.encode_b);
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
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const Tap tap = pixel_tap(px, py, fl[2 * n], fl[2 * n + 1], p.width, p.height);
#pragma unroll
        for (int ch = 0; ch < 7; ++ch) {
          const __half* src = ch < 3
                                  ? input + static_cast<std::size_t>(3 * n + ch) * plane
                                  : (n == 0 ? encode : encode_b) + static_cast<std::size_t>(ch - 3) * plane;
          const int dst = ch < 3 ? 3 * n + ch : 6 + 4 * n + ch - 3;
          acc[dst] += rounded(bilinear(src, tap, p.width));
        }
      }
      acc[14] += __half2float(input[6 * plane]);  // l'instant, à l'élément 0 (étape 5.2)
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
#pragma unroll
  for (int c = 0; c < 28; ++c) acc[c] *= norm;
  store_x<28, 32>(static_cast<__half*>(p.x), p.x_pitch, out_plane, o, acc);
}

// Fin : flot du dernier bloc ajouté ; image (warp bilinéaire ou Catmull-Rom, masque sigmoïde, mélange) ou
// flot et masque.
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
    fl[k] = rounded(__half2float(last_x[element(p.x_pitch, plane, 24 + k, i)]) +
                    read_z(last_y, p.y_height, p.y_width, p.y_pitch, k, y, x));
  const float mask = read_z(last_y, p.y_height, p.y_width, p.y_pitch, 4, y, x);
  const float sig = rounded(1.0F / (1.0F + __expf(-mask)));
  if (p.image == nullptr) {  // variante mouvement : le flot final et le masque
    auto* flow = static_cast<__half*>(p.flow);
    auto* out_mask = static_cast<__half*>(p.mask);
#pragma unroll
    for (int k = 0; k < 4; ++k) flow[static_cast<std::size_t>(k) * plane + i] = __float2half_rn(fl[k]);
    out_mask[i] = __float2half_rn(sig);
    return;
  }
  float warped[2][3];
  const Bounds window{.left = p.window_left,
                      .top = p.window_top,
                      .right = p.window_left + p.window_width,
                      .bottom = p.window_top + p.window_height};
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const Tap tap = window_tap(x, y, fl[2 * n], fl[2 * n + 1], window);
#pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
      const __half* source = input + static_cast<std::size_t>(3 * n + ch) * plane;
      warped[n][ch] = rounded(p.cubic != 0 ? catmull_rom_sample(source, tap, p.width, window)
                                           : bilinear(source, tap, p.width));
    }
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
  // Fenêtre vide (l'appelant ne la connaît pas) : le tenseur entier, comme le promet FusedFinishParams. Sans
  // ce repli, window_tap bornait à [0, -1] et lisait avant le plan (image fausse des tests des noyaux).
  FusedFinishParams p = params;
  if (p.window_width <= 0 || p.window_height <= 0) {
    p.window_left = 0;
    p.window_top = 0;
    p.window_width = p.width;
    p.window_height = p.height;
  }
  fused_finish<<<grid_for(p.width, p.height, block), block, 0, stream>>>(p);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
