// Noyaux du warp (warp_kernels.h), calqués sur kernels/ref/warp.cpp.
#include <cuda_fp16.h>

#include "gpu/cuda/backend/warp_kernels.h"
#include "kernels/spec/warp.h"

namespace ajp::gpu::cuda {
namespace {

constexpr int kBlock = 16;

dim3 grid_for(int width, int height) {
  return dim3(static_cast<unsigned>((width + kBlock - 1) / kBlock),
              static_cast<unsigned>((height + kBlock - 1) / kBlock));
}

// Échantillon bilinéaire du plan c d'un tenseur FP16 (w × h par plan), coordonnées relatives au
// coin (ox, oy) d'une fenêtre fw × fh, bornées à la fenêtre (bilinear_at de ref/warp.cpp).
__device__ float bilinear(const __half* t, int plane_w, int plane_h, int c, int ox, int oy, int fw, int fh,
                          float x, float y) {
  x = fminf(fmaxf(x, 0.0F), static_cast<float>(fw - 1));
  y = fminf(fmaxf(y, 0.0F), static_cast<float>(fh - 1));
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = min(x0 + 1, fw - 1);
  const int y1 = min(y0 + 1, fh - 1);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const __half* p =
      t + static_cast<std::size_t>(c) * static_cast<std::size_t>(plane_w) * static_cast<std::size_t>(plane_h);
  const auto at = [&](int xx, int yy) {
    return __half2float(p[static_cast<std::size_t>(oy + yy) * static_cast<std::size_t>(plane_w) +
                          static_cast<std::size_t>(ox + xx)]);
  };
  const float top = at(x0, y0) * (1.0F - fx) + at(x1, y0) * fx;
  const float bottom = at(x0, y1) * (1.0F - fx) + at(x1, y1) * fx;
  return top * (1.0F - fy) + bottom * fy;
}

__global__ void warp_blend(WarpBlendParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const auto* flow = static_cast<const __half*>(p.flow);
  const auto* mask = static_cast<const __half*>(p.mask);
  const int pw = p.plane_width;
  const int ph = p.plane_height;
  const int wx = p.window_x;
  const int wy = p.window_y;
  const int fw = p.window_width;
  const int fh = p.window_height;
  const float sx = static_cast<float>(p.width) / static_cast<float>(fw);
  const float sy = static_cast<float>(p.height) / static_cast<float>(fh);
  const float u = (static_cast<float>(x) + 0.5F) / sx - 0.5F;
  const float v = (static_cast<float>(y) + 0.5F) / sy - 0.5F;
  float ax;
  float ay;
  float bx;
  float by;
  float weight;
  if (!p.from_middle) {
    ax = bilinear(flow, pw, ph, 0, wx, wy, fw, fh, u, v);
    ay = bilinear(flow, pw, ph, 1, wx, wy, fw, fh, u, v);
    bx = bilinear(flow, pw, ph, 2, wx, wy, fw, fh, u, v);
    by = bilinear(flow, pw, ph, 3, wx, wy, fw, fh, u, v);
    weight = bilinear(mask, pw, ph, 0, wx, wy, fw, fh, u, v);
  } else {
    const float t = p.time;
    const float a0x = bilinear(flow, pw, ph, 0, wx, wy, fw, fh, u, v);
    const float a0y = bilinear(flow, pw, ph, 1, wx, wy, fw, fh, u, v);
    const float a1x = bilinear(flow, pw, ph, 2, wx, wy, fw, fh, u, v);
    const float a1y = bilinear(flow, pw, ph, 3, wx, wy, fw, fh, u, v);
    const float u2 = u - ((1.0F - t) * a0x + t * a1x);
    const float v2 = v - ((1.0F - t) * a0y + t * a1y);
    const float tx =
        bilinear(flow, pw, ph, 2, wx, wy, fw, fh, u2, v2) - bilinear(flow, pw, ph, 0, wx, wy, fw, fh, u2, v2);
    const float ty =
        bilinear(flow, pw, ph, 3, wx, wy, fw, fh, u2, v2) - bilinear(flow, pw, ph, 1, wx, wy, fw, fh, u2, v2);
    const float mm = fminf(fmaxf(bilinear(mask, pw, ph, 0, wx, wy, fw, fh, u2, v2), kernels::kMaskEpsilon),
                           1.0F - kernels::kMaskEpsilon);
    const float z = kernels::kBlendLawA * logf(mm / (1.0F - mm)) +
                    kernels::kBlendLawB * logf((1.0F - t) / t) + kernels::kBlendLawC;
    ax = -t * tx;
    ay = -t * ty;
    bx = (1.0F - t) * tx;
    by = (1.0F - t) * ty;
    weight = 1.0F / (1.0F + expf(-z));
  }
  const auto* a = static_cast<const __half*>(p.a);
  const auto* b = static_cast<const __half*>(p.b);
  auto* out = static_cast<__half*>(p.out);
  const float fx = static_cast<float>(x);
  const float fy = static_cast<float>(y);
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x);
  for (int c = 0; c < 3; ++c) {
    const float va = bilinear(a, p.width, p.height, c, 0, 0, p.width, p.height, fx + sx * ax, fy + sy * ay);
    const float vb = bilinear(b, p.width, p.height, c, 0, 0, p.width, p.height, fx + sx * bx, fy + sy * by);
    out[static_cast<std::size_t>(c) * plane + i] = __float2half_rn(va * weight + vb * (1.0F - weight));
  }
}

__global__ void flow_max(FlowMaxParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  float m = 0.0F;
  if (x < p.window_width && y < p.window_height) {
    const auto* flow = static_cast<const __half*>(p.flow);
    const std::size_t plane =
        static_cast<std::size_t>(p.plane_width) * static_cast<std::size_t>(p.plane_height);
    const std::size_t i = static_cast<std::size_t>(p.window_y + y) * static_cast<std::size_t>(p.plane_width) +
                          static_cast<std::size_t>(p.window_x + x);
    for (int c = 0; c < 4; ++c)
      m = fmaxf(m, fabsf(__half2float(flow[static_cast<std::size_t>(c) * plane + i])));
  }
  // Maximum du bloc, puis un atomicMax par bloc (flottants positifs : ordre des entiers).
  __shared__ float block_max[kBlock * kBlock];
  const int t = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  block_max[t] = m != m ? 0.0F : m;
  __syncthreads();
  for (int stride = kBlock * kBlock / 2; stride > 0; stride /= 2) {
    if (t < stride) block_max[t] = fmaxf(block_max[t], block_max[t + stride]);
    __syncthreads();
  }
  if (t == 0) atomicMax(p.result, __float_as_uint(block_max[0]));
}

}  // namespace

cudaError_t launch_warp_blend(const WarpBlendParams& params, cudaStream_t stream) {
  warp_blend<<<grid_for(params.width, params.height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_flow_max(const FlowMaxParams& params, cudaStream_t stream) {
  flow_max<<<grid_for(params.window_width, params.window_height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
