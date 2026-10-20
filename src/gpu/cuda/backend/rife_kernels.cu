// Noyaux de RIFE de la tranche verticale (voir rife_kernels.h).
#include <cuda_fp16.h>

#include "gpu/cuda/backend/rife_kernels.h"

namespace ajp::gpu::cuda {
namespace {

constexpr int kBlock = 16;
constexpr int kTile = 32;     // bloc des dessins tenus (kernels::kHeldBlock)
constexpr int kTileRows = 8;  // fils par colonne du bloc : 32 × 8 fils, 4 lignes chacun
constexpr float kRed = 0.2126F;
constexpr float kGreen = 0.7152F;
constexpr float kBlue = 0.0722F;

dim3 grid_for(int width, int height) {
  return dim3(static_cast<unsigned>((width + kBlock - 1) / kBlock),
              static_cast<unsigned>((height + kBlock - 1) / kBlock));
}

__device__ std::size_t at(int x, int y, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

__global__ void rife_input(RifeInputParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.padded_width || y >= p.padded_height) return;
  const std::size_t plane =
      static_cast<std::size_t>(p.padded_width) * static_cast<std::size_t>(p.padded_height);
  const std::size_t i = at(x, y, p.padded_width);
  auto* out = static_cast<__half*>(p.input);
  const int ix = x - p.left;
  const int iy = y - p.top;
  const bool inside = ix >= 0 && iy >= 0 && ix < p.width && iy < p.height;
  const std::size_t source_plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t j = inside ? at(ix, iy, p.width) : 0;
  const auto* a = static_cast<const __half*>(p.a);
  const auto* b = static_cast<const __half*>(p.b);
  const __half black = __float2half_rn(p.black);
  for (int c = 0; c < 3 && p.frames; ++c) {
    out[static_cast<std::size_t>(c) * plane + i] =
        inside ? a[static_cast<std::size_t>(c) * source_plane + j] : black;
    out[static_cast<std::size_t>(3 + c) * plane + i] =
        inside ? b[static_cast<std::size_t>(c) * source_plane + j] : black;
  }
  if (p.planes <= 6) return;  // GMFSS : les deux images seules
  out[6 * plane + i] = __float2half_rn(p.time);
  if (p.planes < 11) return;  // RIFE v2 : la grille est dans le graphe
  // kernels::rife_mesh et rife_mesh_step.
  out[7 * plane + i] =
      __float2half_rn(2.0F * static_cast<float>(x) / static_cast<float>(p.padded_width - 1) - 1.0F);
  out[8 * plane + i] =
      __float2half_rn(2.0F * static_cast<float>(y) / static_cast<float>(p.padded_height - 1) - 1.0F);
  out[9 * plane + i] = __float2half_rn(2.0F / static_cast<float>(p.padded_width - 1));
  out[10 * plane + i] = __float2half_rn(2.0F / static_cast<float>(p.padded_height - 1));
}

__global__ void rife_crop(RifeCropParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t in_plane =
      static_cast<std::size_t>(p.padded_width) * static_cast<std::size_t>(p.padded_height);
  const std::size_t out_plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const auto* in = static_cast<const __half*>(p.padded);
  auto* out = static_cast<__half*>(p.out);
  const std::size_t from = at(x + p.left, y + p.top, p.padded_width);
  const std::size_t to = at(x, y, p.width);
  for (int c = 0; c < 3; ++c)
    out[static_cast<std::size_t>(c) * out_plane + to] = in[static_cast<std::size_t>(c) * in_plane + from];
}

__device__ float luma(const __half* rgb, std::size_t plane, std::size_t i) {
  return kRed * __half2float(rgb[i]) + kGreen * __half2float(rgb[plane + i]) +
         kBlue * __half2float(rgb[2 * plane + i]);
}

// Un bloc CUDA par bloc de 32 × 32 pixels, tronqué aux bords.
__global__ void pair_diff(PairDiffParams p) {
  __shared__ float sums[kTile * kTileRows];
  const int x = static_cast<int>(blockIdx.x) * kTile + static_cast<int>(threadIdx.x);
  const int y0 = static_cast<int>(blockIdx.y) * kTile;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const auto* a = static_cast<const __half*>(p.a);
  const auto* b = static_cast<const __half*>(p.b);
  float sum = 0.0F;
  if (x < p.width) {
    for (int y = y0 + static_cast<int>(threadIdx.y); y < min(y0 + kTile, p.height); y += kTileRows) {
      const std::size_t i = at(x, y, p.width);
      sum += fabsf(luma(a, plane, i) - luma(b, plane, i));
    }
  }
  const int t = static_cast<int>(threadIdx.y) * kTile + static_cast<int>(threadIdx.x);
  sums[t] = sum;
  __syncthreads();
  for (int stride = kTile * kTileRows / 2; stride > 0; stride /= 2) {
    if (t < stride) sums[t] += sums[t + stride];
    __syncthreads();
  }
  if (t == 0) {
    const int w = min(kTile, p.width - static_cast<int>(blockIdx.x) * kTile);
    const int h = min(kTile, p.height - y0);
    atomicAdd(p.total, static_cast<double>(sums[0]));
    const float mean = sums[0] / static_cast<float>(w * h);
    atomicMax(p.worst, static_cast<unsigned int>(mean * p.levels * 256.0F));
  }
}

}  // namespace

cudaError_t launch_rife_input(const RifeInputParams& params, cudaStream_t stream) {
  rife_input<<<grid_for(params.padded_width, params.padded_height), dim3(kBlock, kBlock), 0, stream>>>(
      params);
  return cudaGetLastError();
}

cudaError_t launch_rife_crop(const RifeCropParams& params, cudaStream_t stream) {
  rife_crop<<<grid_for(params.width, params.height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_pair_diff(const PairDiffParams& params, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>((params.width + kTile - 1) / kTile),
                  static_cast<unsigned>((params.height + kTile - 1) / kTile));
  pair_diff<<<grid, dim3(kTile, kTileRows), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
