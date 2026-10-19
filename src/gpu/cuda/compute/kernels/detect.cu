// Mesures d'une paire de lumas : coupe de scène et pire bloc des dessins tenus, en une lecture.
// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:698-743 (k_scd_diff) et 923-959
// (k_block_diff), réécrits sur kernels/spec/detect.h. Chaque terme |a − b| × (1 / crête) se
// calcule en float comme dans la référence ; la somme passe en double par tuile, puis dans un
// ordre fixe : le score ne dépend pas de l'ordonnancement (k_scd_diff sommait en float par
// atomicAdd). Un bloc CUDA de 32 × 8 fils couvre une tuile de 32 × 32, quatre lignes par fil.
// Tests : tests/unit/cuda_compute/detect_test.cpp.
#include "gpu/cuda/compute/kernels/device.cuh"
#include "gpu/cuda/compute/kernels/launch.h"

namespace ajp::gpu::cuda::dev {
namespace {

constexpr int kTile = 32;
constexpr int kThreads = kBlockX * kBlockY;
constexpr int kFinishThreads = 256;

// Réduction en arbre d'un tableau partagé de kThreads valeurs ; le fil 0 garde le total.
template <class V>
__device__ void reduce(V* values, int tid) {
  for (int n = kThreads / 2; n > 0; n >>= 1) {
    if (tid < n) values[tid] += values[tid + n];
    __syncthreads();
  }
}

template <class S>
__global__ void k_pair_tiles(Luma a, Luma b, float norm, std::uint32_t scale, double* partials,
                             PairResult* result) {
  __shared__ double sums[kThreads];
  __shared__ std::uint32_t codes[kThreads];
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const int x0 = static_cast<int>(blockIdx.x) * kTile;
  const int y0 = static_cast<int>(blockIdx.y) * kTile;
  const int x = x0 + static_cast<int>(threadIdx.x);
  double sum = 0.0;
  std::uint32_t code_sum = 0;
  if (x < a.width) {
    for (int y = y0 + static_cast<int>(threadIdx.y); y < min(y0 + kTile, a.height); y += kBlockY) {
      const int pa = row_of<S>(a.plane, y)[x];
      const int pb = row_of<S>(b.plane, y)[x];
      sum += static_cast<double>(fabsf(static_cast<float>(pa) - static_cast<float>(pb)) * norm);
      code_sum += static_cast<std::uint32_t>(abs(pa - pb));
    }
  }
  sums[tid] = sum;
  codes[tid] = code_sum;
  __syncthreads();
  reduce(sums, tid);
  reduce(codes, tid);
  if (tid != 0) return;
  partials[blockIdx.y * gridDim.x + blockIdx.x] = sums[0];
  const auto n =
      static_cast<std::uint32_t>((min(x0 + kTile, a.width) - x0) * (min(y0 + kTile, a.height) - y0));
  atomicMax(&result->worst, codes[0] * scale / n);
}

__global__ void k_pair_finish(const double* partials, int count, double area, PairResult* result) {
  __shared__ double sums[kFinishThreads];
  const int tid = static_cast<int>(threadIdx.x);
  double sum = 0.0;
  for (int i = tid; i < count; i += kFinishThreads) sum += partials[i];
  sums[tid] = sum;
  __syncthreads();
  for (int n = kFinishThreads / 2; n > 0; n >>= 1) {
    if (tid < n) sums[tid] += sums[tid + n];
    __syncthreads();
  }
  if (tid == 0) {
    result->score = sums[0] / area;
    result->identical = sums[0] == 0.0 ? 1U : 0U;
  }
}

dim3 tile_grid(int width, int height) {
  return {static_cast<unsigned>((width + kTile - 1) / kTile),
          static_cast<unsigned>((height + kTile - 1) / kTile), 1};
}

}  // namespace

int tile_count(int width, int height) {
  const dim3 grid = tile_grid(width, height);
  return static_cast<int>(grid.x * grid.y);
}

cudaError_t pair_tiles(Luma a, Luma b, float norm, std::uint32_t scale, double* partials, PairResult* result,
                       cudaStream_t stream) {
  const dim3 grid = tile_grid(a.width, a.height);
  if (a.sample == Sample::U8) {
    k_pair_tiles<std::uint8_t><<<grid, block2d(), 0, stream>>>(a, b, norm, scale, partials, result);
  } else {
    k_pair_tiles<std::uint16_t><<<grid, block2d(), 0, stream>>>(a, b, norm, scale, partials, result);
  }
  return cudaGetLastError();
}

cudaError_t pair_finish(const double* partials, int count, double area, PairResult* result,
                        cudaStream_t stream) {
  k_pair_finish<<<1, kFinishThreads, 0, stream>>>(partials, count, area, result);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda::dev
