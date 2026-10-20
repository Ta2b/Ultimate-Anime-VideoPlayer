// Noyaux du softmax splatting de GMFSS (voir gmfss_kernels.h). Même calcul que
// kernels/glsl/gmfss_splat.comp et gmfss_normalize.comp ; référence kernels/ref/gmfss.cpp.
#include <cuda_fp16.h>

#include "gpu/cuda/backend/gmfss_kernels.h"
#include "kernels/spec/gmfss.h"

namespace ajp::gpu::cuda {
namespace {

constexpr int kBlock = 16;
// Plans d'un groupe de l'axe z : la géométrie d'un pixel se refait par groupe.
constexpr int kGroup = 16;

__device__ std::size_t at(int x, int y, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

__global__ void splat(SoftSplatParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i = at(x, y, p.width);
  const auto* motion = static_cast<const __half*>(p.motion);
  const float tx = static_cast<float>(x) + p.s * __half2float(motion[i]);
  const float ty = static_cast<float>(y) + p.s * __half2float(motion[plane + i]);
  // Hors de ]−1, taille[, aucun coin n'est dans le plan ; NaN et infinis échouent aussi.
  if (!(tx > -1.0F && tx < static_cast<float>(p.width) && ty > -1.0F && ty < static_cast<float>(p.height)))
    return;
  const float scale = expf(p.s * __half2float(motion[2 * plane + i]));
  const int x0 = static_cast<int>(floorf(tx));
  const int y0 = static_cast<int>(floorf(ty));
  const float east = tx - static_cast<float>(x0);
  const float west = static_cast<float>(x0 + 1) - tx;
  const float south = ty - static_cast<float>(y0);
  const float north = static_cast<float>(y0 + 1) - ty;
  const int cx[4] = {x0, x0 + 1, x0, x0 + 1};
  const int cy[4] = {y0, y0, y0 + 1, y0 + 1};
  const float share[4] = {west * north, east * north, west * south, east * south};
  const int first = static_cast<int>(blockIdx.z) * kGroup;
  const int last = min(p.channels, first + kGroup);
  const auto* source = static_cast<const __half*>(p.source);
  for (int k = 0; k < 4; ++k) {
    if (cx[k] < 0 || cx[k] >= p.width || cy[k] < 0 || cy[k] >= p.height) continue;
    const std::size_t j = at(cx[k], cy[k], p.width);
    const float weight = scale * share[k];
    for (int c = first; c < last; ++c) {
      atomicAdd(&p.sums[static_cast<std::size_t>(c) * plane + j],
                __half2float(source[static_cast<std::size_t>(c) * plane + i]) * weight);
    }
    if (blockIdx.z == 0) atomicAdd(&p.sums[static_cast<std::size_t>(p.channels) * plane + j], weight);
  }
}

__global__ void normalize(SoftSplatParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i = at(x, y, p.width);
  const float total =
      p.sums[static_cast<std::size_t>(p.channels) * plane + i] + kernels::kGmfssNormalizeEpsilon;
  auto* out = static_cast<__half*>(p.out);
  for (int c = 0; c < p.channels; ++c) {
    const std::size_t k = static_cast<std::size_t>(c) * plane + i;
    out[k] = __float2half_rn(p.sums[k] / total);
  }
}

}  // namespace

std::size_t soft_splat_sums_bytes(int width, int height, int channels) {
  return static_cast<std::size_t>(channels + 1) * static_cast<std::size_t>(width) *
         static_cast<std::size_t>(height) * sizeof(float);
}

cudaError_t launch_soft_splat(const SoftSplatParams& params, cudaStream_t stream) {
  const cudaError_t cleared = cudaMemsetAsync(
      params.sums, 0, soft_splat_sums_bytes(params.width, params.height, params.channels), stream);
  if (cleared != cudaSuccess) return cleared;
  const dim3 block(kBlock, kBlock);
  const auto gx = static_cast<unsigned>((params.width + kBlock - 1) / kBlock);
  const auto gy = static_cast<unsigned>((params.height + kBlock - 1) / kBlock);
  splat<<<dim3(gx, gy, static_cast<unsigned>((params.channels + kGroup - 1) / kGroup)), block, 0, stream>>>(
      params);
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  normalize<<<dim3(gx, gy), block, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
