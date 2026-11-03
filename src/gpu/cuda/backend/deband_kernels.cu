// Débandage avant SPAN (voir deband_kernels.h).
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/deband_kernels.cuh"
#include "gpu/cuda/backend/deband_kernels.h"
#include "gpu/cuda/backend/pixel_kernels.cuh"

namespace ajp::gpu::cuda {
namespace {

constexpr int kBlock = 16;

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) deband(DebandParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const int2 offset = deband_offset(x, y);
  const int dx = offset.x;
  const int dy = offset.y;
  // Quatre points en croix autour du pixel, ramenés dans l'image.
  const auto at = [&](int ox, int oy) {
    const int sx = min(max(x + ox, 0), p.width - 1);
    const int sy = min(max(y + oy, 0), p.height - 1);
    return static_cast<std::size_t>(sy) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(sx);
  };
  const std::size_t i = at(0, 0);
  const std::size_t a = at(dx, dy);
  const std::size_t b = at(-dx, -dy);
  const std::size_t d = at(-dy, dx);
  const std::size_t e = at(dy, -dx);
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const auto* in = static_cast<const __half*>(p.in);
  auto* out = static_cast<__half*>(p.out);
  for (int k = 0; k < 3; ++k) {
    const __half* src = in + static_cast<std::size_t>(k) * plane;
    out[static_cast<std::size_t>(k) * plane + i] =
        __float2half(deband_rule(__half2float(src[i]), __half2float(src[a]), __half2float(src[b]),
                                 __half2float(src[d]), __half2float(src[e])));
  }
}

}  // namespace

cudaError_t launch_deband(const DebandParams& params, cudaStream_t stream) {
  if (params.in == nullptr || params.out == nullptr || params.in == params.out) return cudaErrorInvalidValue;
  const dim3 grid(static_cast<unsigned>((params.width + kBlock - 1) / kBlock),
                  static_cast<unsigned>((params.height + kBlock - 1) / kBlock));
  deband<<<grid, dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
