// Port CUDA de kernels/ref/motion_doubt.cpp : un fil par point des plans du mouvement.
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/doubt_kernels.h"
#include "gpu/cuda/backend/pixel_kernels.cuh"
#include "kernels/spec/motion_doubt.h"

namespace kirari::gpu::cuda {
namespace {

constexpr int kBlock = 16;

// kernels::doubt_at, doubt_coord, doubt_flow et doubt_mask réécrits ici : nvcc n'appelle pas une fonction
// constexpr de l'hôte.
__device__ float doubt_at(const std::uint8_t* weights, int cols, int rows, float u, float v) {
  const float cu = fminf(fmaxf(u, 0.0F), static_cast<float>(cols - 1));
  const float cv = fminf(fmaxf(v, 0.0F), static_cast<float>(rows - 1));
  const int c0 = static_cast<int>(cu);
  const int r0 = static_cast<int>(cv);
  const int c1 = min(c0 + 1, cols - 1);
  const int r1 = min(r0 + 1, rows - 1);
  const float fx = cu - static_cast<float>(c0);
  const float fy = cv - static_cast<float>(r0);
  const float top = __fmul_rn(static_cast<float>(weights[r0 * cols + c0]), 1.0F - fx) +
                    __fmul_rn(static_cast<float>(weights[r0 * cols + c1]), fx);
  const float bottom = __fmul_rn(static_cast<float>(weights[r1 * cols + c0]), 1.0F - fx) +
                       __fmul_rn(static_cast<float>(weights[r1 * cols + c1]), fx);
  return (__fmul_rn(top, 1.0F - fy) + __fmul_rn(bottom, fy)) / static_cast<float>(kernels::kDoubtFull);
}

__device__ float doubt_coord(int x, int cells, int window) {
  return __fmul_rn(static_cast<float>(x) + 0.5F, static_cast<float>(cells)) / static_cast<float>(window) -
         0.5F;
}

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) motion_doubt(MotionDoubtParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.plane_width || y >= p.plane_height) return;
  const std::size_t plane =
      static_cast<std::size_t>(p.plane_width) * static_cast<std::size_t>(p.plane_height);
  const std::size_t at =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.plane_width) + static_cast<std::size_t>(x);
  const auto* flow = static_cast<const __half*>(p.flow);
  const auto* mask = static_cast<const __half*>(p.mask);
  auto* out_flow = static_cast<__half*>(p.out_flow);
  auto* out_mask = static_cast<__half*>(p.out_mask);
  const int wx = x - p.window_x;
  const int wy = y - p.window_y;
  float w = 0.0F;
  if (wx >= 0 && wy >= 0 && wx < p.window_width && wy < p.window_height) {
    w = doubt_at(p.weights, p.cols, p.rows, doubt_coord(wx, p.cols, p.window_width),
                 doubt_coord(wy, p.rows, p.window_height));
  }
  if (w <= 0.0F) {  // recopié au bit près, hors de la fenêtre compris
    for (int c = 0; c < 4; ++c) out_flow[c * plane + at] = flow[c * plane + at];
    out_mask[at] = mask[at];
    return;
  }
  for (int c = 0; c < 4; ++c) {
    const float f = w >= 1.0F ? 0.0F : __fmul_rn(__half2float(flow[c * plane + at]), 1.0F - w);
    out_flow[c * plane + at] = __float2half_rn(f);
  }
  const float m = __half2float(mask[at]);
  out_mask[at] = __float2half_rn(w >= 1.0F ? p.fade : m + __fmul_rn(p.fade - m, w));
}

}  // namespace

cudaError_t launch_motion_doubt(const MotionDoubtParams& params, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>((params.plane_width + kBlock - 1) / kBlock),
                  static_cast<unsigned>((params.plane_height + kBlock - 1) / kBlock));
  motion_doubt<<<grid, dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
