#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <limits>
#include <math_constants.h>

#include "gpu/cuda/backend/gmflow_convex_kernels.h"
#include "gpu/cuda/backend/gmflow_convex_math.h"

namespace kirari::gpu::cuda {
namespace {

using namespace gmflow_convex;
constexpr int kThreads = 128;
constexpr int kDirections = 2;
constexpr int kInputPairs = kInput / 2;
constexpr int kFeaturePairs = kFeatures / 2;
static_assert(kSubpixels == 16);

// L'arrondi du flot pour les convolutions du masque se fait ici. Le flot à agrandir garde sa copie FP32.
__global__ void input_kernel(Grid grid, const __half2* features, const float* flow, __half2* out) {
  const int thread = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int cell = thread / kInputPairs;
  if (cell >= grid.pixels()) return;
  const int c = thread % kInputPairs;
  const int at = static_cast<int>(blockIdx.y) * grid.pixels() + cell;
  __half2 value = __floats2half2_rn(0.F, 0.F);
  if (c == 0) {
    value = __floats2half2_rn(flow[2 * at], flow[2 * at + 1]);
  } else if (c <= kFeaturePairs) {
    value = features[at * kFeaturePairs + c - 1];
  }
  out[at * kInputPairs + c] = value;
}

// Un demi-warp par cellule du flot, un fil par sous-pixel. Le demi-warp lit les neuf vecteurs une fois
// (un voisin sur chacun des neuf premiers fils) et les distribue à ses 16 sorties.
__global__ void __launch_bounds__(kThreads)
    expand_kernel(Grid grid, const __half* mask, const float* flow, float* out) {
  const int thread = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int cell = thread / kSubpixels;
  if (cell >= grid.pixels()) return;
  const int sub = thread % kSubpixels;
  const int side = static_cast<int>(blockIdx.y);
  const int at = side * grid.pixels() + cell;
  const __half* logits = mask + static_cast<std::size_t>(at) * kChannels;
  const float* side_flow = flow + 2 * side * grid.pixels();
  // kThreads et la taille traitée sont multiples de 16 : tous les fils nommés participent.
  const unsigned group = 0xffffU << (static_cast<unsigned>(threadIdx.x) & 16U);
  float neighbor_x = 0.F, neighbor_y = 0.F;
  if (sub < kNeighbors) {
    const int neighbor = grid.neighbor(cell, sub);
    if (neighbor >= 0) {
      neighbor_x = kFactor * side_flow[2 * neighbor];
      neighbor_y = kFactor * side_flow[2 * neighbor + 1];
    }
  }

  float values[kNeighbors];
  float peak = -CUDART_INF_F;
#pragma unroll
  for (int tap = 0; tap < kNeighbors; ++tap) {
    values[tap] = __half2float(logits[mask_channel(tap, sub)]);
    peak = fmaxf(peak, values[tap]);
  }
  float denominator = 0.F;
#pragma unroll
  for (int tap = 0; tap < kNeighbors; ++tap) {
    values[tap] = expf(values[tap] - peak);
    denominator += values[tap];
  }
  const float inverse = 1.F / denominator;
  float x = 0.F, y = 0.F;
#pragma unroll
  for (int tap = 0; tap < kNeighbors; ++tap) {
    const float weight = values[tap] * inverse;
    x = fmaf(weight, __shfl_sync(group, neighbor_x, tap, kSubpixels), x);
    y = fmaf(weight, __shfl_sync(group, neighbor_y, tap, kSubpixels), y);
  }
  const int plane = grid.pixels() * kSubpixels;
  const int pixel = grid.expanded(cell, sub);
  out[(2 * side) * plane + pixel] = x;
  out[(2 * side + 1) * plane + pixel] = y;
}

bool grid_fits(int height, int width) {
  const auto pixels = static_cast<std::int64_t>(height) * width;
  return height > 0 && width > 0 &&
         pixels <= (std::numeric_limits<int>::max() - kThreads) / (kDirections * kInputPairs);
}

dim3 launch_grid(Grid grid, int per_cell) {
  return {static_cast<unsigned>((grid.pixels() * per_cell + kThreads - 1) / kThreads), kDirections};
}

bool aligned(const void* p, std::size_t boundary) {
  return p != nullptr && reinterpret_cast<std::uintptr_t>(p) % boundary == 0;
}

}  // namespace

cudaError_t launch_gmflow_convex_input(cudaStream_t stream, const void* features, const float* flow,
                                       int height, int width, void* packed) {
  if (!grid_fits(height, width) || !aligned(features, alignof(__half2)) || !aligned(flow, alignof(float)) ||
      !aligned(packed, alignof(__half2)) || features == packed || flow == packed)
    return cudaErrorInvalidValue;
  const Grid grid{height, width};
  input_kernel<<<launch_grid(grid, kInputPairs), kThreads, 0, stream>>>(
      grid, static_cast<const __half2*>(features), flow, static_cast<__half2*>(packed));
  return cudaGetLastError();
}

cudaError_t launch_gmflow_convex(cudaStream_t stream, const void* mask, const float* flow, int height,
                                 int width, float* output) {
  if (!grid_fits(height, width) || !aligned(mask, alignof(__half)) || !aligned(flow, alignof(float)) ||
      !aligned(output, alignof(float)) || mask == output || flow == output)
    return cudaErrorInvalidValue;
  const Grid grid{height, width};
  expand_kernel<<<launch_grid(grid, kSubpixels), kThreads, 0, stream>>>(
      grid, static_cast<const __half*>(mask), flow, output);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
