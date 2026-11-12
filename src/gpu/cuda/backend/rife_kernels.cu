// Noyaux de RIFE de la tranche verticale (voir rife_kernels.h).
#include <cuda_fp16.h>

#include "gpu/cuda/backend/pixel_kernels.cuh"
#include "gpu/cuda/backend/rife_kernels.h"

namespace ajp::gpu::cuda {
namespace {

constexpr int kBlock = 16;
constexpr int kTile = 16;      // bloc des dessins tenus (kernels::kHeldBlock)
constexpr int kTileRows = 16;  // fils par colonne du bloc : 16 × 16 fils, une ligne chacun
static_assert(kBlock * kBlock == kPixelThreads && kTile * kTileRows == kPixelThreads);

dim3 grid_for(int width, int height) {
  return dim3(static_cast<unsigned>((width + kBlock - 1) / kBlock),
              static_cast<unsigned>((height + kBlock - 1) / kBlock));
}

__device__ std::size_t at(int x, int y, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) rife_input(RifeInputParams p) {
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
  for (int c = 0; c < 3; ++c) {
    out[static_cast<std::size_t>(c) * plane + i] =
        inside ? a[static_cast<std::size_t>(c) * source_plane + j] : black;
    out[static_cast<std::size_t>(3 + c) * plane + i] =
        inside ? b[static_cast<std::size_t>(c) * source_plane + j] : black;
  }
  if (p.planes <= 6) return;  // les deux images seules
  out[6 * plane + i] = __float2half_rn(p.time);
  if (p.planes < 11 || !p.mesh) return;  // RIFE v2 : la grille est dans le graphe
  // kernels::rife_mesh et rife_mesh_step.
  out[7 * plane + i] =
      __float2half_rn(2.0F * static_cast<float>(x) / static_cast<float>(p.padded_width - 1) - 1.0F);
  out[8 * plane + i] =
      __float2half_rn(2.0F * static_cast<float>(y) / static_cast<float>(p.padded_height - 1) - 1.0F);
  out[9 * plane + i] = __float2half_rn(2.0F / static_cast<float>(p.padded_width - 1));
  out[10 * plane + i] = __float2half_rn(2.0F / static_cast<float>(p.padded_height - 1));
}

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) rife_crop(RifeCropParams p) {
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

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) area_reduce(AreaReduceParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.out_width || y >= p.out_height) return;
  const float sx = static_cast<float>(p.in_width) / static_cast<float>(p.out_width);
  const float sy = static_cast<float>(p.in_height) / static_cast<float>(p.out_height);
  const float x0 = static_cast<float>(x) * sx;
  const float y0 = static_cast<float>(y) * sy;
  const float x1 = x0 + sx;
  const float y1 = y0 + sy;
  const int last_x = min(static_cast<int>(ceilf(x1)), p.in_width);
  const int last_y = min(static_cast<int>(ceilf(y1)), p.in_height);
  const std::size_t in_plane = static_cast<std::size_t>(p.in_width) * static_cast<std::size_t>(p.in_height);
  const std::size_t out_plane =
      static_cast<std::size_t>(p.out_width) * static_cast<std::size_t>(p.out_height);
  const auto* in = static_cast<const __half*>(p.in);
  auto* out = static_cast<__half*>(p.out);
  float sum[3] = {0.0F, 0.0F, 0.0F};
#pragma unroll 1
  for (int j = static_cast<int>(floorf(y0)); j < last_y; ++j) {
    const float wy = fminf(y1, static_cast<float>(j + 1)) - fmaxf(y0, static_cast<float>(j));
#pragma unroll 1
    for (int i = static_cast<int>(floorf(x0)); i < last_x; ++i) {
      const float w = wy * (fminf(x1, static_cast<float>(i + 1)) - fmaxf(x0, static_cast<float>(i)));
      const std::size_t from = at(i, j, p.in_width);
      for (int c = 0; c < 3; ++c) sum[c] += __half2float(in[static_cast<std::size_t>(c) * in_plane + from]) * w;
    }
  }
  const float norm = 1.0F / (sx * sy);
  const std::size_t to = at(x, y, p.out_width);
  for (int c = 0; c < 3; ++c) out[static_cast<std::size_t>(c) * out_plane + to] = __float2half_rn(sum[c] * norm);
}

// Luma du pixel pointé dans le plan R de trois plans RGB consécutifs, aux coefficients de la source.
__device__ float luma(const __half* red, std::size_t plane, float kr, float kb) {
  return kr * __half2float(red[0]) + (1.0F - kr - kb) * __half2float(red[plane]) +
         kb * __half2float(red[2 * plane]);
}

// Un bloc CUDA par bloc de 16 × 16 pixels, tronqué aux bords.
__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) pair_diff(PairDiffParams p) {
  const int x = static_cast<int>(blockIdx.x) * kTile + static_cast<int>(threadIdx.x);
  const int y0 = static_cast<int>(blockIdx.y) * kTile;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const auto* a = static_cast<const __half*>(p.a);
  const auto* b = static_cast<const __half*>(p.b);
  float sum = 0.0F;
  if (x < p.width) {
    // Pointeurs avancés d'une ligne de fils : sm_120 garde sinon les 24 adresses 64 bits des quatre
    // lignes déroulées et déborde sous la borne de 64 registres.
    const int last = min(y0 + kTile, p.height);
    const int first = y0 + static_cast<int>(threadIdx.y);
    const std::size_t step = static_cast<std::size_t>(kTileRows) * static_cast<std::size_t>(p.width);
    const std::size_t start = first < last ? at(x, first, p.width) : 0;
    const __half* pa = a + start;
    const __half* pb = b + start;
    for (int y = first; y < last; y += kTileRows) {
      sum += fabsf(luma(pa, plane, p.kr, p.kb) - luma(pb, plane, p.kr, p.kb));
      pa += step;
      pb += step;
    }
  }
  sum = block_reduce(sum, 0.0F, Sum{});
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const int w = min(kTile, p.width - static_cast<int>(blockIdx.x) * kTile);
    const int h = min(kTile, p.height - y0);
    atomicAdd(p.total, static_cast<double>(sum));
    const float mean = sum / static_cast<float>(w * h);
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

cudaError_t launch_area_reduce(const AreaReduceParams& params, cudaStream_t stream) {
  area_reduce<<<grid_for(params.out_width, params.out_height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_pair_diff(const PairDiffParams& params, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>((params.width + kTile - 1) / kTile),
                  static_cast<unsigned>((params.height + kTile - 1) / kTile));
  pair_diff<<<grid, dim3(kTile, kTileRows), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
