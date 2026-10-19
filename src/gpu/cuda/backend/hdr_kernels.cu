// Noyaux de RTX HDR de la tranche verticale (voir hdr_kernels.h).
#include <cuda_fp16.h>

#include "gpu/cuda/backend/hdr_kernels.h"

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

__global__ void to_rgba8(ToRgba8Params p) {
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

__global__ void rgb10a2_to_surface(Rgb10a2ToSurfaceParams p) {
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

}  // namespace

cudaError_t launch_to_rgba8(const ToRgba8Params& params, cudaStream_t stream) {
  to_rgba8<<<grid_for(params.width, params.height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_rgb10a2_to_surface(const Rgb10a2ToSurfaceParams& params, cudaStream_t stream) {
  rgb10a2_to_surface<<<grid_for(params.width, params.height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
