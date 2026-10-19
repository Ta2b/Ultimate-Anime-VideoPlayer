// Noyaux du flux optique matériel (optical_flow_kernels.h), calqués sur kernels/ref/optical_flow.cpp.
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/optical_flow_kernels.h"
#include "kernels/spec/optical_flow.h"

namespace ajp::gpu::cuda {
namespace {

constexpr int kBlock = 16;

dim3 grid_for(int width, int height) {
  return dim3(static_cast<unsigned>((width + kBlock - 1) / kBlock),
              static_cast<unsigned>((height + kBlock - 1) / kBlock));
}

__global__ void luma8(Luma8Params p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const auto* rgb = static_cast<const __half*>(p.rgb);
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x);
  const float luma = 0.2126F * __half2float(rgb[i]) + 0.7152F * __half2float(rgb[plane + i]) +
                     0.0722F * __half2float(rgb[2 * plane + i]);
  const auto code = static_cast<unsigned char>(__float2int_rn(255.0F * fminf(fmaxf(luma, 0.0F), 1.0F)));
  static_cast<unsigned char*>(
      p.first)[static_cast<std::size_t>(y) * p.first_pitch + static_cast<std::size_t>(x)] = code;
  static_cast<unsigned char*>(
      p.second)[static_cast<std::size_t>(y) * p.second_pitch + static_cast<std::size_t>(x)] = code;
}

struct Vec {
  float x;
  float y;
};

__device__ Vec add(Vec a, Vec b) {
  return {a.x + b.x, a.y + b.y};
}
__device__ Vec scale(float s, Vec a) {
  return {s * a.x, s * a.y};
}

// Un flot S10.5 lu en bilinéaire borné à la grille, en pixels de la grille (at de la référence).
struct Field {
  const unsigned char* base;
  std::size_t pitch;
  int w;
  int h;
  int grid;

  // kernels::flow_to_grid, réécrit ici : nvcc n'appelle pas une fonction constexpr de l'hôte
  // depuis l'appareil sans --expt-relaxed-constexpr.
  __device__ Vec cell(int i, int j) const {
    const auto* v = reinterpret_cast<const std::int16_t*>(base + static_cast<std::size_t>(j) * pitch) + 2 * i;
    const float unit = kernels::kFlowFraction * static_cast<float>(grid);
    return {static_cast<float>(v[0]) / unit, static_cast<float>(v[1]) / unit};
  }

  __device__ Vec at(Vec p) const {
    const float x = fminf(fmaxf(p.x, 0.0F), static_cast<float>(w - 1));
    const float y = fminf(fmaxf(p.y, 0.0F), static_cast<float>(h - 1));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = min(x0 + 1, w - 1);
    const int y1 = min(y0 + 1, h - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const Vec top = add(scale(1.0F - fx, cell(x0, y0)), scale(fx, cell(x1, y0)));
    const Vec bottom = add(scale(1.0F - fx, cell(x0, y1)), scale(fx, cell(x1, y1)));
    return add(scale(1.0F - fy, top), scale(fy, bottom));
  }
};

__device__ float confidence(float error) {
  const float r = error / kernels::kFlowConsistency;
  return 1.0F / (1.0F + r * r);
}

// Désaccord du déplacement v vu de p avec a (ea) et avec b (eb) de la spécification.
__device__ float disagreement_a(const Field& forward, Vec p, Vec v) {
  const Vec e = add(forward.at(add(p, scale(-0.5F, v))), scale(-1.0F, v));
  return hypotf(e.x, e.y);
}
__device__ float disagreement_b(const Field& backward, Vec p, Vec v) {
  const Vec e = add(backward.at(add(p, scale(0.5F, v))), v);
  return hypotf(e.x, e.y);
}

__global__ void mid_motion(MidMotionParams p) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= p.cells_x || j >= p.cells_y) return;
  const Field forward{static_cast<const unsigned char*>(p.forward), p.forward_pitch, p.cells_x, p.cells_y,
                      p.grid};
  const Field backward{static_cast<const unsigned char*>(p.backward), p.backward_pitch, p.cells_x, p.cells_y,
                       p.grid};
  const Vec at{static_cast<float>(i), static_cast<float>(j)};
  Vec m = forward.at(at);
  Vec n = scale(-1.0F, backward.at(at));
  for (int k = 0; k < kernels::kMidFlowIterations; ++k) {
    m = forward.at(add(at, scale(-0.5F, m)));
    n = scale(-1.0F, backward.at(add(at, scale(0.5F, n))));
  }
  const float sm = confidence(disagreement_a(forward, at, m)) * confidence(disagreement_b(backward, at, m));
  const float sn = confidence(disagreement_a(forward, at, n)) * confidence(disagreement_b(backward, at, n));
  const Vec mid = sm >= sn ? m : n;
  const float va = confidence(disagreement_a(forward, at, mid));
  const float vb = confidence(disagreement_b(backward, at, mid));
  const std::size_t plane = static_cast<std::size_t>(p.cells_x) * static_cast<std::size_t>(p.cells_y);
  const std::size_t o =
      static_cast<std::size_t>(j) * static_cast<std::size_t>(p.cells_x) + static_cast<std::size_t>(i);
  auto* flow = static_cast<__half*>(p.flow);
  flow[o] = __float2half(-0.5F * mid.x);
  flow[plane + o] = __float2half(-0.5F * mid.y);
  flow[2 * plane + o] = __float2half(0.5F * mid.x);
  flow[3 * plane + o] = __float2half(0.5F * mid.y);
  static_cast<__half*>(p.mask)[o] = __float2half(va / (va + vb));
}

}  // namespace

cudaError_t launch_luma8(const Luma8Params& params, cudaStream_t stream) {
  luma8<<<grid_for(params.width, params.height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_mid_motion(const MidMotionParams& params, cudaStream_t stream) {
  mid_motion<<<grid_for(params.cells_x, params.cells_y), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
