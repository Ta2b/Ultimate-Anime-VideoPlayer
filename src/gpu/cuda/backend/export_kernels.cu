// Sortie YUV de l'export par puits CUDA (voir export_kernels.h).
#include <cuda_fp16.h>

#include "gpu/cuda/backend/export_kernels.h"
#include "gpu/cuda/backend/pixel_kernels.cuh"

namespace ajp::gpu::cuda {
namespace {

// 32 × 8 : une ligne de chroma par chaîne d'exécution, lectures contiguës le long de la ligne.
constexpr int kBlockX = 32;
constexpr int kBlockY = kPixelThreads / kBlockX;

// kernels/spec/resample.h mirror_index : miroir sans répétition du bord, puis clamp.
__device__ inline int mirror_index(int i, int n) {
  i = i < 0 ? -i - 1 : i;
  i = i >= n ? 2 * n - 1 - i : i;
  return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// kernels/spec/color.h bayer8_threshold.
__device__ inline float bayer8_threshold(int x, int y) {
  const auto m = [](int a, int b) { return 2 * (a ^ b) + b; };
  const int v = 16 * m(x & 1, y & 1) + 4 * m((x >> 1) & 1, (y >> 1) & 1) + m((x >> 2) & 1, (y >> 2) & 1);
  return (static_cast<float>(v) + 0.5F) / 64.0F;
}

// quantize (arrondi au pair) ou quantize_dithered (seuil de la position), en codes du conteneur.
__device__ inline unsigned short quantize(const YuvWriteParams& p, float raw, int x, int y) {
  const float v = raw / p.qdiv;
  float code = 0.0F;
  if (p.dither) {
    const float d = v + bayer8_threshold(x, y);
    code = d >= p.qmax ? p.qmax : (d > 0.0F ? floorf(d) : 0.0F);
  } else {
    code = v >= p.qmax ? p.qmax : (v > 0.0F ? rintf(v) : 0.0F);
  }
  return static_cast<unsigned short>(code * p.qdiv);
}

struct Yuv {
  float y;
  float u;
  float v;
};

// k_post_matrix (spec to_yuv) d'un pixel du tenseur.
__device__ inline Yuv to_yuv(const YuvWriteParams& p, const __half* rgb, std::size_t plane, std::size_t i) {
  const float r = __half2float(rgb[i]);
  const float g = __half2float(rgb[plane + i]);
  const float b = __half2float(rgb[2 * plane + i]);
  const float y = p.kr * r + (1.0F - p.kr - p.kb) * g + p.kb * b;
  return {.y = y, .u = (b - y) / (2.0F * (1.0F - p.kb)), .v = (r - y) / (2.0F * (1.0F - p.kr))};
}

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) yuv_rows(YuvWriteParams p) {
  const int cw = p.width / 2;
  const int cx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (cx >= cw || y >= p.height) return;
  const auto* rgb = static_cast<const __half*>(p.rgb);
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width);
  // La luma des deux pixels de la paire de colonnes.
  const int x = 2 * cx;
  const Yuv left = to_yuv(p, rgb, plane, row + static_cast<std::size_t>(x));
  const Yuv right = to_yuv(p, rgb, plane, row + static_cast<std::size_t>(x) + 1);
  const unsigned short y0 = quantize(p, left.y * p.yscale + p.yoff, x, y);
  const unsigned short y1 = quantize(p, right.y * p.yscale + p.yoff, x + 1, y);
  auto* luma = static_cast<unsigned char*>(p.luma) + static_cast<std::size_t>(y) * p.luma_pitch;
  if (p.ten_bit) {
    reinterpret_cast<ushort2*>(luma)[cx] = make_ushort2(y0, y1);
  } else {
    reinterpret_cast<uchar2*>(luma)[cx] =
        make_uchar2(static_cast<unsigned char>(y0), static_cast<unsigned char>(y1));
  }
  // Cb et Cr réduits en largeur, somme FP32 de la première prise à la dernière.
  const int start = p.x_start[cx];
  const float* w = p.x_weights + static_cast<std::size_t>(cx) * static_cast<std::size_t>(p.x_taps);
  float su = 0.0F;
  float sv = 0.0F;
  for (int t = 0; t < p.x_taps; ++t) {
    const int xi = mirror_index(start + t, p.width);
    const Yuv c = to_yuv(p, rgb, plane, row + static_cast<std::size_t>(xi));
    su += w[t] * c.u;
    sv += w[t] * c.v;
  }
  p.rows[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) + static_cast<std::size_t>(cx)] =
      make_float2(su, sv);
}

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) yuv_chroma(YuvWriteParams p) {
  const int cw = p.width / 2;
  const int ch = p.height / 2;
  const int cx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int cy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (cx >= cw || cy >= ch) return;
  const int start = p.y_start[cy];
  const float* w = p.y_weights + static_cast<std::size_t>(cy) * static_cast<std::size_t>(p.y_taps);
  float su = 0.0F;
  float sv = 0.0F;
  for (int t = 0; t < p.y_taps; ++t) {
    const int yi = mirror_index(start + t, p.height);
    const float2 c =
        p.rows[static_cast<std::size_t>(yi) * static_cast<std::size_t>(cw) + static_cast<std::size_t>(cx)];
    su += w[t] * c.x;
    sv += w[t] * c.y;
  }
  const unsigned short u = quantize(p, su * p.cscale + p.coff, cx, cy);
  const unsigned short v = quantize(p, sv * p.cscale + p.coff, cx, cy);
  auto* chroma = static_cast<unsigned char*>(p.chroma) + static_cast<std::size_t>(cy) * p.chroma_pitch;
  if (p.ten_bit) {
    reinterpret_cast<ushort2*>(chroma)[cx] = make_ushort2(u, v);
  } else {
    reinterpret_cast<uchar2*>(chroma)[cx] =
        make_uchar2(static_cast<unsigned char>(u), static_cast<unsigned char>(v));
  }
}

dim3 grid_for(int width, int height) {
  return dim3(static_cast<unsigned>((width + kBlockX - 1) / kBlockX),
              static_cast<unsigned>((height + kBlockY - 1) / kBlockY));
}

}  // namespace

std::size_t yuv_rows_bytes(int width, int height) {
  return static_cast<std::size_t>(width / 2) * static_cast<std::size_t>(height) * sizeof(float2);
}

cudaError_t launch_write_yuv(const YuvWriteParams& params, cudaStream_t stream) {
  const dim3 block(kBlockX, kBlockY);
  yuv_rows<<<grid_for(params.width / 2, params.height), block, 0, stream>>>(params);
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  yuv_chroma<<<grid_for(params.width / 2, params.height / 2), block, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
