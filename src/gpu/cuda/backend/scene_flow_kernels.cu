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
static_assert(kTile == kBlock, "un fil par pixel de la tuile");
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

// Une tuile par bloc, un fil par pixel du flux ; les 4 × 4 premiers fils lisent aussi un vecteur. Des
// additions atomiques entières en mémoire partagée : leur ordre ne change rien.
__global__ void scene_tiles(SceneTilesParams p) {
  __shared__ int pixels;
  __shared__ int texture;
  __shared__ int diff;
  __shared__ int signed_diff;
  __shared__ int cells;
  __shared__ int sum_x;
  __shared__ int sum_y;
  __shared__ unsigned long long sum_xx;
  __shared__ unsigned long long sum_yy;
  __shared__ int cost;
  __shared__ int unmatched;
  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  if (tx == 0 && ty == 0) {
    pixels = texture = diff = signed_diff = cells = sum_x = sum_y = cost = unmatched = 0;
    sum_xx = sum_yy = 0ULL;
  }
  __syncthreads();
  const int col = static_cast<int>(blockIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  const int x = col * kTile + tx;
  const int y = row * kTile + ty;
  if (x < p.width && y < p.height) {
    const int va = p.a[static_cast<std::size_t>(y) * p.a_pitch + static_cast<std::size_t>(x)];
    const int vb = p.b[static_cast<std::size_t>(y) * p.b_pitch + static_cast<std::size_t>(x)];
    int grad = 0;
    if (x + 1 < p.width)
      grad += abs(int{p.a[static_cast<std::size_t>(y) * p.a_pitch + static_cast<std::size_t>(x + 1)]} - va);
    if (y + 1 < p.height)
      grad += abs(int{p.a[static_cast<std::size_t>(y + 1) * p.a_pitch + static_cast<std::size_t>(x)]} - va);
    atomicAdd(&pixels, 1);
    atomicAdd(&texture, grad);
    atomicAdd(&diff, abs(vb - va));
    atomicAdd(&signed_diff, vb - va);
  }
  const int cells_x = (p.width + kGrid - 1) / kGrid;
  const int cells_y = (p.height + kGrid - 1) / kGrid;
  const int cx = col * kCells + tx;
  const int cy = row * kCells + ty;
  if (tx < kCells && ty < kCells && cx < cells_x && cy < cells_y) {
    const short2 v = *reinterpret_cast<const short2*>(reinterpret_cast<const unsigned char*>(p.flow) +
                                                      static_cast<std::size_t>(cy) * p.flow_pitch +
                                                      static_cast<std::size_t>(cx) * sizeof(short2));
    const int c = p.cost[static_cast<std::size_t>(cy) * p.cost_pitch + static_cast<std::size_t>(cx)];
    atomicAdd(&cells, 1);
    atomicAdd(&sum_x, int{v.x});
    atomicAdd(&sum_y, int{v.y});
    atomicAdd(&sum_xx, static_cast<unsigned long long>(int{v.x} * int{v.x}));
    atomicAdd(&sum_yy, static_cast<unsigned long long>(int{v.y} * int{v.y}));
    atomicAdd(&cost, c);
    atomicAdd(&unmatched, c >= kCostHigh ? 1 : 0);
  }
  __syncthreads();
  if (tx != 0 || ty != 0) return;
  constexpr float kUnit = 32.0F;
  kernels::SceneTileSums& out =
      p.out[static_cast<std::size_t>(row) * static_cast<std::size_t>(p.cols) + static_cast<std::size_t>(col)];
  out.cells = static_cast<float>(cells);
  out.sum_x = static_cast<float>(sum_x) / kUnit;
  out.sum_y = static_cast<float>(sum_y) / kUnit;
  out.sum_xx = static_cast<float>(sum_xx) / (kUnit * kUnit);
  out.sum_yy = static_cast<float>(sum_yy) / (kUnit * kUnit);
  out.cost = static_cast<float>(cost);
  out.unmatched = static_cast<float>(unmatched);
  out.pixels = static_cast<float>(pixels);
  out.texture = static_cast<float>(texture);
  out.diff = static_cast<float>(diff);
  out.signed_diff = static_cast<float>(signed_diff);
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
  scene_tiles<<<dim3(static_cast<unsigned>(params.cols), static_cast<unsigned>(params.rows)),
                dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
