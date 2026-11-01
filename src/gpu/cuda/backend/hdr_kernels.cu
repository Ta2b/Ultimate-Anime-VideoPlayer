// Noyaux de RTX HDR de la tranche verticale (voir hdr_kernels.h).
#include <cuda_fp16.h>

#include "gpu/cuda/backend/hdr_kernels.h"
#include "gpu/cuda/backend/pixel_kernels.cuh"

namespace ajp::gpu::cuda {
namespace {

constexpr int kBlock = 16;
// Un point de bande noire : aucun canal au-dessus de 2 % (le noir et le bruit de codage autour).
constexpr float kBarBlack = 0.02F;

dim3 grid_for(int width, int height) {
  return dim3(static_cast<unsigned>((width + kBlock - 1) / kBlock),
              static_cast<unsigned>((height + kBlock - 1) / kBlock));
}

// Bruit uniforme dans [0, 1) tiré d'un hachage entier (PCG) du pixel, du canal et de la graine.
__device__ float uniform(std::uint32_t x, std::uint32_t y, std::uint32_t salt) {
  std::uint32_t v = x * 0x9E3779B1U ^ y * 0x85EBCA77U ^ salt * 0xC2B2AE3DU;
  v = v * 747796405U + 2891336453U;
  v = ((v >> ((v >> 28U) + 4U)) ^ v) * 277803737U;
  v = (v >> 22U) ^ v;
  return static_cast<float>(v >> 8U) * (1.0F / 16777216.0F);
}

__device__ unsigned int dither8(float value, std::uint32_t x, std::uint32_t y, std::uint32_t salt) {
  const float noise = uniform(x, y, salt) - uniform(x, y, salt ^ 0x5bd1e995U);  // triangulaire, ±1
  float code = value * 255.0F + noise + 0.5F;
  code = code != code ? 0.0F : fminf(fmaxf(code, 0.0F), 255.0F);  // NaN donne 0
  return static_cast<unsigned int>(code);
}

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) to_rgba8(ToRgba8Params p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x);
  const auto* rgb = static_cast<const __half*>(p.rgb);
  const auto ux = static_cast<std::uint32_t>(x);
  const auto uy = static_cast<std::uint32_t>(y);
  const unsigned int r = dither8(__half2float(rgb[i]), ux, uy, p.seed * 3U);
  const unsigned int g = dither8(__half2float(rgb[plane + i]), ux, uy, p.seed * 3U + 1U);
  const unsigned int b = dither8(__half2float(rgb[2 * plane + i]), ux, uy, p.seed * 3U + 2U);
  auto* row = reinterpret_cast<unsigned int*>(static_cast<unsigned char*>(p.out) +
                                              static_cast<std::size_t>(y) * p.pitch);
  row[x] = r | (g << 8U) | (b << 16U) | (255U << 24U);
}

__device__ unsigned short widen10(unsigned int code) {
  return static_cast<unsigned short>((code * 65535U + 511U) / 1023U);
}

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm)
    rgb10a2_to_surface(Rgb10a2ToSurfaceParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const auto* row = reinterpret_cast<const unsigned int*>(static_cast<const unsigned char*>(p.in) +
                                                          static_cast<std::size_t>(y) * p.pitch);
  const unsigned int pixel = row[x];
  const ushort4 out = make_ushort4(widen10(pixel & 0x3FFU), widen10((pixel >> 10U) & 0x3FFU),
                                   widen10((pixel >> 20U) & 0x3FFU), 65535);
  surf2Dwrite(out, p.surface, x * static_cast<int>(sizeof(ushort4)), y);
}

// Statistiques de luminance : un thread par point d'une grille au pas de 4 ; histogramme de 64 cases en
// mémoire partagée par bloc, puis ajouté au global ; somme des luminances par warp.
__global__ void __launch_bounds__(kBlock* kBlock) mean_luma_partial(MeanLumaParams p) {
  __shared__ unsigned int bins[kLumaBins];
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  if (tid < kLumaBins) bins[tid] = 0U;
  __syncthreads();
  const int gx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int gy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int w4 = p.width / 4;
  const int h4 = p.height / 4;
  float value = 0.0F;
  float blue = 0.0F;
  float chroma = 0.0F;
  if (gx < w4 && gy < h4) {
    const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
    const std::size_t i = static_cast<std::size_t>(gy * 4) * static_cast<std::size_t>(p.width) +
                          static_cast<std::size_t>(gx * 4);
    const auto* rgb = static_cast<const __half*>(p.rgb);
    const float r = __half2float(rgb[i]);
    const float g = __half2float(rgb[plane + i]);
    const float b = __half2float(rgb[2 * plane + i]);
    bool inside = true;
    if (p.bars != nullptr) {
      // Un point non noir marque sa ligne et sa colonne : l'image suivante ne mesure que leur rectangle.
      if (fmaxf(r, fmaxf(g, b)) > kBarBlack) {
        p.bars[4 + gy] = 1U;
        p.bars[4 + h4 + gx] = 1U;
      }
      if (p.bars[2] != 0U)
        inside = gx >= static_cast<int>(p.bars[0]) && gy >= static_cast<int>(p.bars[1]) &&
                 gx < static_cast<int>(p.bars[2]) && gy < static_cast<int>(p.bars[3]);
    }
    if (inside) {
      value = 0.2126F * r + 0.7152F * g + 0.0722F * b;
      value = value != value ? 0.0F : fminf(fmaxf(value, 0.0F), 1.0F);
      blue = fmaxf(0.0F, b - fmaxf(r, g));
      chroma = fmaxf(r, fmaxf(g, b)) - fminf(r, fminf(g, b));
      blue = blue != blue ? 0.0F : fminf(blue, 1.0F);
      chroma = chroma != chroma ? 0.0F : fminf(chroma, 1.0F);
      atomicAdd(&bins[min(kLumaBins - 1, static_cast<int>(value * static_cast<float>(kLumaBins)))], 1U);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    blue += __shfl_down_sync(0xFFFFFFFFU, blue, offset);
    chroma += __shfl_down_sync(0xFFFFFFFFU, chroma, offset);
  }
  if (tid % 32 == 0) {
    atomicAdd(&p.sum[0], value);
    atomicAdd(&p.sum[1], blue);
    atomicAdd(&p.sum[2], chroma);
  }
  __syncthreads();
  if (tid < kLumaBins && bins[tid] != 0U) atomicAdd(&p.histogram[tid], bins[tid]);
}

// Le centile q (0 à 1) de l'histogramme, au milieu de sa case.
__device__ float percentile(const unsigned int* histogram, float count, float q) {
  const float target = q * count;
  float seen = 0.0F;
  for (int b = 0; b < kLumaBins; ++b) {
    seen += static_cast<float>(histogram[b]);
    if (seen >= target) return (static_cast<float>(b) + 0.5F) / static_cast<float>(kLumaBins);
  }
  return 1.0F;
}

// Le rectangle de l'image hors bandes noires, d'après les marques de l'image mesurée, pour la suivante ; les
// marques sont remises à zéro. Rien de marqué (image noire, fondu) ou moins de la moitié de l'image dans une
// dimension (un plan sombre, pas des bandes) : l'image entière.
__device__ void next_bars(unsigned int* bars, int w4, int h4) {
  unsigned int* rows = bars + 4;
  unsigned int* cols = rows + h4;
  int top = 0;
  while (top < h4 && rows[top] == 0U) ++top;
  int bottom = h4;
  while (bottom > top && rows[bottom - 1] == 0U) --bottom;
  int left = 0;
  while (left < w4 && cols[left] == 0U) ++left;
  int right = w4;
  while (right > left && cols[right - 1] == 0U) --right;
  const bool bands =
      2 * (bottom - top) >= h4 && 2 * (right - left) >= w4 && (bottom - top < h4 || right - left < w4);
  bars[0] = bands ? static_cast<unsigned int>(left) : 0U;
  bars[1] = bands ? static_cast<unsigned int>(top) : 0U;
  bars[2] = bands ? static_cast<unsigned int>(right) : 0U;
  bars[3] = bands ? static_cast<unsigned int>(bottom) : 0U;
  for (int y = 0; y < h4; ++y) rows[y] = 0U;
  for (int x = 0; x < w4; ++x) cols[x] = 0U;
}

__global__ void mean_luma_finish(MeanLumaParams p) {
  const int w4 = p.width / 4;
  const int h4 = p.height / 4;
  // Les points mesurés : le rectangle hors bandes de cette image, sinon l'image entière.
  float count = static_cast<float>(w4) * static_cast<float>(h4);
  if (p.bars != nullptr && p.bars[2] != 0U)
    count = static_cast<float>(p.bars[2] - p.bars[0]) * static_cast<float>(p.bars[3] - p.bars[1]);
  if (count > 0.0F) {
    float dark = 0.0F;
    float bright = 0.0F;
    for (int b = 0; b < kLumaBins; ++b) {
      const float centre = (static_cast<float>(b) + 0.5F) / static_cast<float>(kLumaBins);
      if (centre < 0.10F) dark += static_cast<float>(p.histogram[b]);
      if (centre > 0.80F) bright += static_cast<float>(p.histogram[b]);
    }
    p.out[0] = p.sum[0] / count;
    p.out[1] = percentile(p.histogram, count, 0.05F);
    p.out[2] = percentile(p.histogram, count, 0.50F);
    p.out[3] = percentile(p.histogram, count, 0.95F);
    p.out[4] = percentile(p.histogram, count, 0.995F);
    p.out[5] = dark / count;
    p.out[6] = bright / count;
    p.out[7] = p.sum[1] / count;
    p.out[8] = p.sum[2] / count;
    p.out[9] += 1.0F;  // numéro de mesure : la dernière écriture
  }
  p.sum[0] = 0.0F;
  p.sum[1] = 0.0F;
  p.sum[2] = 0.0F;
  for (int b = 0; b < kLumaBins; ++b) p.histogram[b] = 0U;
  if (p.bars != nullptr) next_bars(p.bars, w4, h4);
}

}  // namespace

cudaError_t launch_mean_luma(const MeanLumaParams& params, cudaStream_t stream) {
  if (params.width < 4 || params.height < 4) return cudaSuccess;
  const dim3 grid(static_cast<unsigned>((params.width / 4 + kBlock - 1) / kBlock),
                  static_cast<unsigned>((params.height / 4 + kBlock - 1) / kBlock));
  mean_luma_partial<<<grid, dim3(kBlock, kBlock), 0, stream>>>(params);
  mean_luma_finish<<<1, 1, 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_to_rgba8(const ToRgba8Params& params, cudaStream_t stream) {
  to_rgba8<<<grid_for(params.width, params.height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_rgb10a2_to_surface(const Rgb10a2ToSurfaceParams& params, cudaStream_t stream) {
  rgb10a2_to_surface<<<grid_for(params.width, params.height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
