// Noyaux du warp (warp_kernels.h), calqués sur kernels/ref/warp.cpp.
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/bilinear.cuh"
#include "gpu/cuda/backend/pixel_kernels.cuh"
#include "gpu/cuda/backend/warp_kernels.h"
#include "kernels/spec/poses.h"
#include "kernels/spec/warp.h"

namespace ajp::gpu::cuda {
namespace {

constexpr int kBlock = 16;

dim3 grid_for(int width, int height) {
  return dim3(static_cast<unsigned>((width + kBlock - 1) / kBlock),
              static_cast<unsigned>((height + kBlock - 1) / kBlock));
}

__device__ Planes motion_planes(const WarpBlendParams& p, const void* data, cudaTextureObject_t texture) {
  return Planes{.data = static_cast<const __half*>(data),
                .texture = texture,
                .plane_w = p.plane_width,
                .plane_h = p.plane_height,
                .ox = p.window_x,
                .oy = p.window_y,
                .fw = p.window_width,
                .fh = p.window_height};
}

__device__ Planes image_planes(const WarpBlendParams& p, const void* data, cudaTextureObject_t texture) {
  return Planes{.data = static_cast<const __half*>(data),
                .texture = texture,
                .plane_w = p.width,
                .plane_h = p.height,
                .fw = p.width,
                .fh = p.height};
}

// L'instant réglé du pixel (u, v) pour garder les poses (keeping_poses de ref/warp.cpp).
template <typename S>
__device__ float pose_time_at(const WarpBlendParams& p, float u, float v) {
  const Planes flow = motion_planes(p, p.flow, p.textures.flow);
  const Planes previous = motion_planes(p, p.previous_flow, p.textures.previous_flow);
  const float t = p.time;
  float tp = t;
  for (int k = 0; k < kernels::kPoseIterations; ++k) {
    const float u2 = u - ((1.0F - tp) * S::sample(flow, 0, u, v) + tp * S::sample(flow, 2, u, v));
    const float v2 = v - ((1.0F - tp) * S::sample(flow, 1, u, v) + tp * S::sample(flow, 3, u, v));
    const float dx = S::sample(flow, 2, u2, v2) - S::sample(flow, 0, u2, v2);
    const float dy = S::sample(flow, 3, u2, v2) - S::sample(flow, 1, u2, v2);
    float before = 0.0F;
    if (p.previous_flow != nullptr) {
      const float ax = u - tp * dx;
      const float ay = v - tp * dy;
      float qx = ax;
      float qy = ay;
      for (int step = 0; step < 2; ++step) {
        const float nx = ax - S::sample(previous, 2, qx, qy);
        const float ny = ay - S::sample(previous, 3, qx, qy);
        qx = nx;
        qy = ny;
      }
      before = hypotf(S::sample(previous, 2, qx, qy) - S::sample(previous, 0, qx, qy),
                      S::sample(previous, 3, qx, qy) - S::sample(previous, 1, qx, qy));
    }
    const float now = hypotf(dx, dy);
    const float r = (before + kernels::kPoseEpsilon) / (before + now + 2.0F * kernels::kPoseEpsilon);
    // kernels::pose_time, réécrit ici : nvcc n'appelle pas une fonction constexpr de l'hôte.
    const float timed = t < 0.5F ? 2.0F * t * r : 1.0F - 2.0F * (1.0F - t) * r;
    tp = fminf(fmaxf(timed, kernels::kPoseTimeLimit), 1.0F - kernels::kPoseTimeLimit);
  }
  return tp;
}

// Les trois canaux du pixel (x, y) de la sortie : mouvement lu dans la fenêtre, puis les deux
// images déformées et mélangées.
template <typename S>
__device__ void warp_pixel(const WarpBlendParams& p, int x, int y, float* rgb) {
  const Planes flow = motion_planes(p, p.flow, p.textures.flow);
  const Planes mask = motion_planes(p, p.mask, p.textures.mask);
  const float sx = static_cast<float>(p.width) / static_cast<float>(p.window_width);
  const float sy = static_cast<float>(p.height) / static_cast<float>(p.window_height);
  const float u = (static_cast<float>(x) + 0.5F) / sx - 0.5F;
  const float v = (static_cast<float>(y) + 0.5F) / sy - 0.5F;
  float ax;
  float ay;
  float bx;
  float by;
  float weight;
  if (!p.from_middle) {
    ax = S::sample(flow, 0, u, v);
    ay = S::sample(flow, 1, u, v);
    bx = S::sample(flow, 2, u, v);
    by = S::sample(flow, 3, u, v);
    weight = S::sample(mask, 0, u, v);
  } else {
    // L'instant réglé se lit toujours exactement : le rapport des deux mouvements amplifie la
    // fraction sur 8 bits de la texture (émulée sur les images du test : écart porté de 2 à 10/1024).
    const float t = p.keep_poses ? pose_time_at<Exact>(p, u, v) : p.time;
    const float u2 = u - ((1.0F - t) * S::sample(flow, 0, u, v) + t * S::sample(flow, 2, u, v));
    const float v2 = v - ((1.0F - t) * S::sample(flow, 1, u, v) + t * S::sample(flow, 3, u, v));
    const float tx = S::sample(flow, 2, u2, v2) - S::sample(flow, 0, u2, v2);
    const float ty = S::sample(flow, 3, u2, v2) - S::sample(flow, 1, u2, v2);
    const float mm =
        fminf(fmaxf(S::sample(mask, 0, u2, v2), kernels::kMaskEpsilon), 1.0F - kernels::kMaskEpsilon);
    const float z = kernels::kBlendLawA * logf(mm / (1.0F - mm)) +
                    kernels::kBlendLawB * logf((1.0F - t) / t) + kernels::kBlendLawC;
    ax = -t * tx;
    ay = -t * ty;
    bx = (1.0F - t) * tx;
    by = (1.0F - t) * ty;
    weight = 1.0F / (1.0F + expf(-z));
  }
  const Planes a = image_planes(p, p.a, p.textures.a);
  const Planes b = image_planes(p, p.b, p.textures.b);
  const float fx = static_cast<float>(x);
  const float fy = static_cast<float>(y);
  for (int c = 0; c < 3; ++c) {
    const float va = S::sample(a, c, fx + sx * ax, fy + sy * ay);
    const float vb = S::sample(b, c, fx + sx * bx, fy + sy * by);
    rgb[c] = va * weight + vb * (1.0F - weight);
  }
}

__global__ void __launch_bounds__(kPixelThreads, kHeavyBlocksPerSm) warp_blend(WarpBlendParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  float rgb[3];
  warp_pixel<Exact>(p, x, y, rgb);
  if (p.surface != 0) {
    store_rgba16(p.surface, x, y, __float2half_rn(rgb[0]), __float2half_rn(rgb[1]), __float2half_rn(rgb[2]));
    return;
  }
  auto* out = static_cast<__half*>(p.out);
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x);
  for (int c = 0; c < 3; ++c) out[static_cast<std::size_t>(c) * plane + i] = __float2half_rn(rgb[c]);
}

// Deux pixels voisins par fil, lus par les objets texture : un bloc 16 × 16 couvre 32 × 16 pixels.
__global__ void __launch_bounds__(kPixelThreads, kHeavyBlocksPerSm) warp_blend_filtered(WarpBlendParams p) {
  const int x = 2 * static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;  // largeur paire : x + 1 est dans l'image
  // Un pixel après l'autre : déroulés, les deux calculs entrelacés débordent de la borne lourde
  // sur sm_86 et sm_89 (les poses lues exactement).
  __half2 pair[3];
#pragma unroll 1
  for (int k = 0; k < 2; ++k) {
    float rgb[3];
    warp_pixel<Filtered>(p, x + k, y, rgb);
    for (int c = 0; c < 3; ++c) {
      const __half v = __float2half_rn(rgb[c]);
      pair[c] = k == 0 ? __halves2half2(v, v) : __halves2half2(__low2half(pair[c]), v);
    }
  }
  if (p.surface != 0) {
    store_rgba16(p.surface, x, y, __low2half(pair[0]), __low2half(pair[1]), __low2half(pair[2]));
    store_rgba16(p.surface, x + 1, y, __high2half(pair[0]), __high2half(pair[1]), __high2half(pair[2]));
    return;
  }
  auto* out = static_cast<__half2*>(p.out);
  const std::size_t plane = static_cast<std::size_t>(p.width) / 2 * static_cast<std::size_t>(p.height);
  const std::size_t i =
      (static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x)) / 2;
  for (int c = 0; c < 3; ++c) out[static_cast<std::size_t>(c) * plane + i] = pair[c];
}

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) flow_max(FlowMaxParams p) {
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
  m = block_reduce(m != m ? 0.0F : m, 0.0F, Max{});
  if (threadIdx.x == 0 && threadIdx.y == 0) atomicMax(p.result, __float_as_uint(m));
}

}  // namespace

cudaError_t launch_warp_blend(const WarpBlendParams& params, cudaStream_t stream) {
  const WarpBlendParams::Textures& t = params.textures;
  const bool previous = params.keep_poses && params.previous_flow != nullptr;
  if (t.a != 0 && t.b != 0 && t.flow != 0 && t.mask != 0 && (!previous || t.previous_flow != 0) &&
      params.width % 2 == 0 && reinterpret_cast<std::uintptr_t>(params.out) % 4 == 0) {
    warp_blend_filtered<<<grid_for(params.width / 2, params.height), dim3(kBlock, kBlock), 0, stream>>>(
        params);
  } else {
    warp_blend<<<grid_for(params.width, params.height), dim3(kBlock, kBlock), 0, stream>>>(params);
  }
  return cudaGetLastError();
}

cudaError_t launch_flow_max(const FlowMaxParams& params, cudaStream_t stream) {
  flow_max<<<grid_for(params.window_width, params.window_height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
