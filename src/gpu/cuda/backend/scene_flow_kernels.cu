// Noyaux de la carte de la scène (scene_flow_kernels.h), calqués sur kernels/ref/scene_flow.cpp.
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/scene_flow_kernels.h"

namespace kirari::gpu::cuda {
namespace {

constexpr int kBlock = 16;
constexpr int kTile = kernels::kSceneTileFlowPixels;  // pixels du flux par côté de tuile
constexpr int kCells = kernels::kSceneTileCells;
constexpr int kGrid = kernels::kSceneFlowGrid;
constexpr int kCostHigh = kernels::kSceneCostHigh;
static_assert(kTile == 16 && kCells == 4);
static_assert(kCells * kGrid == kTile);

__device__ float channel(const SceneLumaParams& p, std::size_t index) {
  if (p.fp16) return __half2float(static_cast<const __half*>(p.rgb)[index]);
  return static_cast<const float*>(p.rgb)[index];
}

__global__ void scene_luma(SceneLumaParams p) {
  const int out_width = p.width / p.divisor;
  const int out_height = p.height / p.divisor;
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= out_width || y >= out_height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const float kg = 1.0F - p.kr - p.kb;
  float sum = 0.0F;
  for (int j = 0; j < p.divisor; ++j) {
    for (int i = 0; i < p.divisor; ++i) {
      const std::size_t at = static_cast<std::size_t>(y * p.divisor + j) * static_cast<std::size_t>(p.width) +
                             static_cast<std::size_t>(x * p.divisor + i);
      // Mêmes produits et même ordre que la référence, sans contraction.
      sum = __fadd_rn(
          sum, __fadd_rn(__fadd_rn(__fmul_rn(p.kr, channel(p, at)), __fmul_rn(kg, channel(p, plane + at))),
                         __fmul_rn(p.kb, channel(p, 2 * plane + at))));
    }
  }
  const float area = 1.0F / static_cast<float>(p.divisor * p.divisor);
  const float luma = fminf(fmaxf(__fmul_rn(sum, area), 0.0F), 1.0F);
  p.luma[static_cast<std::size_t>(y) * p.pitch + static_cast<std::size_t>(x)] =
      static_cast<unsigned char>(static_cast<int>(__fadd_rn(__fmul_rn(255.0F, luma), 0.5F)));
}

struct LumaSums {
  int count = 0;
  int gradient = 0;
  int absolute = 0;
  int signed_sum = 0;
};

__device__ LumaSums luma_group(const SceneTilesParams& p, int x, int y, bool aligned) {
  if (y >= p.height) return {};
  const auto* a = p.a + static_cast<std::size_t>(y) * p.a_pitch;
  const auto* b = p.b + static_cast<std::size_t>(y) * p.b_pitch;
  if (aligned && x + 3 < p.width) {
    const auto sa = *reinterpret_cast<const unsigned*>(a + x);
    const auto sb = *reinterpret_cast<const unsigned*>(b + x);
    const unsigned right = x + 4 < p.width ? a[x + 4] : a[x + 3];
    const unsigned beside = (sa >> 8U) | (right << 24U);
    const auto below = y + 1 < p.height ? *reinterpret_cast<const unsigned*>(a + p.a_pitch + x) : sa;
    constexpr unsigned kOnes = 0x01010101U;
    return {.count = 4,
            .gradient = static_cast<int>(__vsadu4(sa, beside) + __vsadu4(sa, below)),
            .absolute = static_cast<int>(__vsadu4(sa, sb)),
            .signed_sum = static_cast<int>(__dp4a(sb, kOnes, 0U)) - static_cast<int>(__dp4a(sa, kOnes, 0U))};
  }
  LumaSums out;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int px = x + i;
    if (px >= p.width) break;
    const int va = a[px], vb = b[px];
    if (px + 1 < p.width) out.gradient += abs(int{a[px + 1]} - va);
    if (y + 1 < p.height) out.gradient += abs(int{a[p.a_pitch + px]} - va);
    out.absolute += abs(vb - va);
    out.signed_sum += vb - va;
    ++out.count;
  }
  return out;
}

__device__ int warp_sum(int value) {
  constexpr unsigned kAll = 0xffffffffU;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_add_sync(kAll, value);
#else
#pragma unroll
  for (int offset = 16; offset != 0; offset >>= 1) value += __shfl_down_sync(kAll, value, offset);
  return value;  // le résultat est lu dans le premier fil du warp
#endif
}

// Les deux warps combinent les lumas avec une seule barrière, sans atomiques.
__global__ void scene_tiles(SceneTilesParams p) {
  __shared__ LumaSums bottom;
  const int tid = static_cast<int>(threadIdx.x);
  const int col = static_cast<int>(blockIdx.x), row = static_cast<int>(blockIdx.y);
  const int tx = tid % kCells, ty = tid / kCells;
  const auto alignment =
      reinterpret_cast<std::uintptr_t>(p.a) | reinterpret_cast<std::uintptr_t>(p.b) | p.a_pitch | p.b_pitch;
  const auto luma = luma_group(p, col * kTile + 4 * tx, row * kTile + ty, (alignment & 3U) == 0U);
  const LumaSums top{warp_sum(luma.count), warp_sum(luma.gradient), warp_sum(luma.absolute),
                     warp_sum(luma.signed_sum)};
  if (tid == 32) bottom = top;
  int valid = 0, vx = 0, vy = 0, cost = 0;
  if (tid < 32) {
    const int cells_x = (p.width + kGrid - 1) / kGrid, cells_y = (p.height + kGrid - 1) / kGrid;
    const int cx = col * kCells + tx, cy = row * kCells + ty;
    if (ty < kCells && cx < cells_x && cy < cells_y) {
      const auto* f =
          reinterpret_cast<const unsigned char*>(p.flow) + static_cast<std::size_t>(cy) * p.flow_pitch;
      const short2 v = reinterpret_cast<const short2*>(f)[cx];
      valid = 1;
      vx = v.x;
      vy = v.y;
      cost = p.cost[static_cast<std::size_t>(cy) * p.cost_pitch + static_cast<std::size_t>(cx)];
    }
    const int x2 = vx * vx, y2 = vy * vy;
    const int cells = warp_sum(valid);
    const int sum_x = warp_sum(vx), sum_y = warp_sum(vy);
    // Seize carrés de S10.5 peuvent dépasser un entier signé. En sommant
    // à part les quatre bits bas, chaque réduction reste bornée par 2^30.
    const int cx_hi = warp_sum(x2 >> 4), cy_hi = warp_sum(y2 >> 4);
    const int cx_lo = warp_sum(x2 & 15), cy_lo = warp_sum(y2 & 15);
    const int total_cost = warp_sum(cost);
    const int unmatched = warp_sum(cost >= kCostHigh ? 1 : 0);
    if (tid == 0) {
      const auto sx = (static_cast<std::uint64_t>(cx_hi) << 4U) + static_cast<std::uint64_t>(cx_lo);
      const auto sy = (static_cast<std::uint64_t>(cy_hi) << 4U) + static_cast<std::uint64_t>(cy_lo);
      constexpr float unit = 32.0F;
      auto& out = p.out[static_cast<std::size_t>(row) * static_cast<std::size_t>(p.cols) +
                        static_cast<std::size_t>(col)];
      out.cells = static_cast<float>(cells);
      out.sum_x = static_cast<float>(sum_x) / unit;
      out.sum_y = static_cast<float>(sum_y) / unit;
      out.sum_xx = static_cast<float>(sx) / (unit * unit);
      out.sum_yy = static_cast<float>(sy) / (unit * unit);
      out.cost = static_cast<float>(total_cost);
      out.unmatched = static_cast<float>(unmatched);
    }
  }
  __syncthreads();
  if (tid == 0) {
    auto& out = p.out[static_cast<std::size_t>(row) * static_cast<std::size_t>(p.cols) +
                      static_cast<std::size_t>(col)];
    out.pixels = static_cast<float>(top.count + bottom.count);
    out.texture = static_cast<float>(top.gradient + bottom.gradient);
    out.diff = static_cast<float>(top.absolute + bottom.absolute);
    out.signed_diff = static_cast<float>(top.signed_sum + bottom.signed_sum);
  }
}

}  // namespace

cudaError_t launch_scene_luma(const SceneLumaParams& params, cudaStream_t stream) {
  const int width = params.width / params.divisor;
  const int height = params.height / params.divisor;
  if (width <= 0 || height <= 0) return cudaErrorInvalidValue;
  const dim3 block(kBlock, kBlock);
  const dim3 grid(static_cast<unsigned>((width + kBlock - 1) / kBlock),
                  static_cast<unsigned>((height + kBlock - 1) / kBlock));
  scene_luma<<<grid, block, 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_scene_tiles(const SceneTilesParams& params, cudaStream_t stream) {
  if (params.cols <= 0 || params.rows <= 0) return cudaErrorInvalidValue;
  scene_tiles<<<dim3(static_cast<unsigned>(params.cols), static_cast<unsigned>(params.rows)), kTile * kCells,
                0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
