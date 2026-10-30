// Noyaux de RTX HDR de la tranche verticale (voir hdr_kernels.h).
#include <cuda_fp16.h>

#include "gpu/cuda/backend/hdr_kernels.h"
#include "gpu/cuda/backend/pixel_kernels.cuh"

namespace ajp::gpu::cuda {
namespace {

constexpr int kBlock = 16;

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
  float value = 0.0F;
  if (gx < p.width / 4 && gy < p.height / 4) {
    const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
    const std::size_t i = static_cast<std::size_t>(gy * 4) * static_cast<std::size_t>(p.width) +
                          static_cast<std::size_t>(gx * 4);
    const auto* rgb = static_cast<const __half*>(p.rgb);
    value = 0.2126F * __half2float(rgb[i]) + 0.7152F * __half2float(rgb[plane + i]) +
            0.0722F * __half2float(rgb[2 * plane + i]);
    value = value != value ? 0.0F : fminf(fmaxf(value, 0.0F), 1.0F);
    atomicAdd(&bins[min(kLumaBins - 1, static_cast<int>(value * static_cast<float>(kLumaBins)))], 1U);
  }
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  if (tid % 32 == 0) atomicAdd(p.sum, value);
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

__global__ void mean_luma_finish(MeanLumaParams p) {
  const float count = static_cast<float>(p.width / 4) * static_cast<float>(p.height / 4);
  if (count > 0.0F) {
    float dark = 0.0F;
    float bright = 0.0F;
    for (int b = 0; b < kLumaBins; ++b) {
      const float centre = (static_cast<float>(b) + 0.5F) / static_cast<float>(kLumaBins);
      if (centre < 0.10F) dark += static_cast<float>(p.histogram[b]);
      if (centre > 0.80F) bright += static_cast<float>(p.histogram[b]);
    }
    p.out[0] = *p.sum / count;
    p.out[1] = percentile(p.histogram, count, 0.05F);
    p.out[2] = percentile(p.histogram, count, 0.50F);
    p.out[3] = percentile(p.histogram, count, 0.95F);
    p.out[4] = percentile(p.histogram, count, 0.995F);
    p.out[5] = dark / count;
    p.out[6] = bright / count;
    p.out[7] += 1.0F;  // numéro de mesure : la dernière écriture
  }
  *p.sum = 0.0F;
  for (int b = 0; b < kLumaBins; ++b) p.histogram[b] = 0U;
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
