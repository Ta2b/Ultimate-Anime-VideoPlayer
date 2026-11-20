#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <limits>

#include "gpu/cuda/backend/gmfss_pair_flow_glue.h"
#include "gpu/cuda/backend/warp_lib.cuh"

namespace kirari::gpu::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kSides = 2;
constexpr int kRgb = 3;
constexpr int kAxes = 2;

struct Images {
  const __half* rgb[kSides];
};

struct Flows {
  const __half* xy[kSides];
};

int blocks_for(int n) {
  return (n + kThreads - 1) / kThreads;
}

bool usable_size(int height, int width) {
  return height >= 4 && width >= 4 && height % 4 == 0 && width % 4 == 0 &&
         height <= std::numeric_limits<int>::max() / 24 / width;
}

__device__ float half_sample(const __half* src, int w, int x, int y) {
  const __half* a = src + 2 * (y * w + x);
  const __half* b = a + w;
  return warp::lerp2(__half2float(a[0]), __half2float(a[1]), __half2float(b[0]), __half2float(b[1]),
                     {0, 1, .5F}, {0, 1, .5F});
}

// Resize(0,5) des images entières. GMFlow et la fusion reçoivent les mêmes demi-images.
__global__ void reduce_rgb(Images in, int h, int w, __half* out0, __half* out1, __half* copy) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int side = static_cast<int>(blockIdx.y);
  const int plane = h * w;
  if (i >= kRgb * plane) return;
  const int c = i / plane, p = i % plane;
  const __half value = __float2half_rn(half_sample(in.rgb[side] + c * 4 * plane, 2 * w, p % w, p / w));
  __half* dest = side == 0 ? out0 : out1;
  dest[i] = value;
  if (copy != nullptr) copy[side * kRgb * plane + i] = value;
}

// En s05, un fil produit un carré de quatre pixels pour la fusion et son pixel GMFlow.
// En lecture, les deux Resize FP16 du graphe séparent les deux réductions et la normalisation.
template <bool Normalize>
__global__ void reduce_rgb_s05(Images in, int h, int w, __half* half0, __half* half1, __half* quarter,
                               GmfssImageNormalization normalization) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int side = static_cast<int>(blockIdx.y);
  const int plane = h * w, coarse_plane = plane / 4, cw = w / 2;
  if (i >= kRgb * coarse_plane) return;
  const int channel = i / coarse_plane, p = i % coarse_plane;
  const int x = 2 * (p % cw), y = 2 * (p / cw);
  const __half* src = in.rgb[side] + channel * 4 * plane;
  __half* dst = (side == 0 ? half0 : half1) + channel * plane + y * w + x;
  const float a = half_sample(src, 2 * w, x, y);
  const float b = half_sample(src, 2 * w, x + 1, y);
  const float c = half_sample(src, 2 * w, x, y + 1);
  const float d = half_sample(src, 2 * w, x + 1, y + 1);
  const __half2 top = __floats2half2_rn(a, b);
  const __half2 bottom = __floats2half2_rn(c, d);
  *reinterpret_cast<__half2*>(dst) = top;
  *reinterpret_cast<__half2*>(dst + w) = bottom;
  float v;
  if constexpr (Normalize) {
    // Les pixels de half0/1 et du Resize suivant sont des tenseurs Half dans le moteur TensorRT.
    const float2 top_rgb = __half22float2(top), bottom_rgb = __half22float2(bottom);
    const float q = warp::rounded(
        warp::lerp2(top_rgb.x, top_rgb.y, bottom_rgb.x, bottom_rgb.y, {0, 1, .5F}, {0, 1, .5F}));
    v = __fdiv_rn(__fsub_rn(q, normalization.mean[channel]), normalization.deviation[channel]);
  } else {
    v = warp::lerp2(a, b, c, d, {0, 1, .5F}, {0, 1, .5F});
  }
  quarter[side * kRgb * coarse_plane + i] = __float2half_rn(v);
}

template <bool Half>
__global__ void flow_l1(const float* from, int h, int w, __half* flow0, __half* flow1) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int side = static_cast<int>(blockIdx.y);
  const int plane = h * w;
  if (i >= plane) return;
  const int source_plane = Half ? plane / 4 : plane;
  const float* src = from + side * kAxes * source_plane;
  __half* dst = side == 0 ? flow0 : flow1;
  if constexpr (Half) {
    const int sw = w / 2, sh = h / 2;
    const warp::Axis ax = warp::coarse_axis(i % w, .5F, sw), ay = warp::coarse_axis(i / w, .5F, sh);
#pragma unroll
    for (int c = 0; c < kAxes; ++c) {
      const float* f = src + c * source_plane;
      const float v =
          warp::lerp2(warp::rounded(f[ay.i0 * sw + ax.i0]), warp::rounded(f[ay.i0 * sw + ax.i1]),
                      warp::rounded(f[ay.i1 * sw + ax.i0]), warp::rounded(f[ay.i1 * sw + ax.i1]), ax, ay);
      dst[c * plane + i] = __float2half_rn(2.F * warp::rounded(v));
    }
  } else {
#pragma unroll
    for (int c = 0; c < kAxes; ++c) dst[c * plane + i] = __float2half_rn(src[c * plane + i]);
  }
}

template <int Reduce>
__global__ void flow_reduced(Flows top, int h, int w, __half* flow0, __half* flow1) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int side = static_cast<int>(blockIdx.y);
  const int reduced_w = w / Reduce, reduced_plane = (h / Reduce) * reduced_w;
  if (i >= reduced_plane) return;
  const int x = (i % reduced_w) * Reduce + Reduce / 2 - 1;
  const int y = (i / reduced_w) * Reduce + Reduce / 2 - 1;
  const int p = y * w + x;
  const int plane = h * w;
  __half* dst = side == 0 ? flow0 : flow1;
  const __half* src = top.xy[side];
#pragma unroll
  for (int c = 0; c < kAxes; ++c) {
    const __half* f = src + c * plane + p;
    const float reduced = warp::lerp2(__half2float(f[0]), __half2float(f[1]), __half2float(f[w]),
                                      __half2float(f[w + 1]), {0, 1, .5F}, {0, 1, .5F});
    dst[c * reduced_plane + i] = __float2half_rn(warp::rounded(reduced) / Reduce);
  }
}

// Deux expressions du graphe. Le backwarp des images divise le flot par le demi-axe puis l'ajoute
// à la grille normalisée. La cohérence construit d'abord x + flot puis normalise cette position.
template <bool Consistency>
__device__ float grid_position(int p, float f, float span) {
  float grid;
  if constexpr (Consistency) {
    grid = __fadd_rn(2.F * __fadd_rn(static_cast<float>(p), f) / span, -1.F);
  } else {
    const float base = __fadd_rn(__fmul_rn(static_cast<float>(p), 2.F / span), -1.F);
    grid = __fadd_rn(base, f / (.5F * span));
  }
  return __fmul_rn(__fadd_rn(grid, 1.F) * .5F, span);
}

struct ZeroTap {
  int at[4];
  float wx, wy;
};

template <bool Consistency>
__device__ ZeroTap grid_tap(int x, int y, float2 f, int h, int w) {
  const float u = grid_position<Consistency>(x, f.x, static_cast<float>(w - 1));
  const float v = grid_position<Consistency>(y, f.y, static_cast<float>(h - 1));
  ZeroTap tap{{-1, -1, -1, -1}, 0.F, 0.F};
  if (!(u > -1.F && u < static_cast<float>(w) && v > -1.F && v < static_cast<float>(h))) return tap;
  const int x0 = __float2int_rd(u), y0 = __float2int_rd(v);
  tap.wx = u - static_cast<float>(x0);
  tap.wy = v - static_cast<float>(y0);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int px = x0 + (k & 1), py = y0 + (k >> 1);
    if (px >= 0 && px < w && py >= 0 && py < h) tap.at[k] = py * w + px;
  }
  return tap;
}

__device__ float sample(const __half* from, const ZeroTap& t) {
  float values[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) values[i] = t.at[i] < 0 ? 0.F : __half2float(from[t.at[i]]);
  return warp::rounded(warp::lerp2(values[0], values[1], values[2], values[3], {0, 1, t.wx}, {0, 1, t.wy}));
}

__device__ float half_norm(float2 f) {
  return warp::rounded(sqrtf(f.x * f.x + f.y * f.y));
}

// Concat des 14 plans du modèle d'origine avec les mêmes arrondis FP16 aux interfaces.
__global__ void metric_inputs(Images images, Flows f, int h, int w, __half* out) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int plane = h * w;
  if (p >= plane) return;
  const int x = p % w, y = p / w;
  const float2 a = make_float2(__half2float(f.xy[0][p]), __half2float(f.xy[0][plane + p]));
  const float2 b = make_float2(__half2float(f.xy[1][p]), __half2float(f.xy[1][plane + p]));
  constexpr float kAlpha = .01000213623046875F;  // la valeur FP16 de 0,01 dans l'ONNX
  const float bound = warp::rounded(.5F + warp::rounded(kAlpha * warp::rounded(half_norm(a) + half_norm(b))));
  const float hx = warp::rounded(.5F * static_cast<float>(w - 1));
  const float hy = warp::rounded(.5F * static_cast<float>(h - 1));
#pragma unroll
  for (int side = 0; side < kSides; ++side) {
    const float2 flow = side == 0 ? a : b;
    const int other = 1 - side;
    const ZeroTap photometric = grid_tap<false>(x, y, flow, h, w);
    const ZeroTap consistency = grid_tap<true>(x, y, flow, h, w);
    float error = 0.F;
#pragma unroll
    for (int c = 0; c < kRgb; ++c) {
      const __half rgb = images.rgb[side][c * plane + p];
      out[(kRgb * side + c) * plane + p] = rgb;
      const float remote = sample(images.rgb[other] + c * plane, photometric);
      error += fabsf(warp::rounded(__half2float(rgb) - remote));
    }
    const float2 residual = make_float2(warp::rounded(flow.x + sample(f.xy[other], consistency)),
                                        warp::rounded(flow.y + sample(f.xy[other] + plane, consistency)));
    out[(6 + side) * plane + p] = __float2half_rn(-error / 3.F);
    out[(8 + 2 * side) * plane + p] = __float2half_rn(flow.x / hx);
    out[(9 + 2 * side) * plane + p] = __float2half_rn(flow.y / hy);
    out[(12 + side) * plane + p] = __float2half_rn(half_norm(residual) > bound ? 1.F : 0.F);
  }
}

}  // namespace

cudaError_t launch_gmfss_pair_flow_input(cudaStream_t stream, const void* frames, int height, int width,
                                         bool half_flow, void* gmflow_images,
                                         const std::array<void*, 2>& half_images,
                                         const GmfssImageNormalization* normalization) {
  if (!usable_size(height, width) || frames == nullptr || gmflow_images == nullptr ||
      std::ranges::any_of(half_images, [](const void* ptr) { return ptr == nullptr; }))
    return cudaErrorInvalidValue;
  const auto* src = static_cast<const __half*>(frames);
  auto* first = static_cast<__half*>(half_images[0]);
  auto* second = static_cast<__half*>(half_images[1]);
  auto* input = static_cast<__half*>(gmflow_images);
  const int plane = height * width;
  const Images images{{src, src + kRgb * 4 * plane}};
  if (half_flow) {
    const dim3 grid(blocks_for(kRgb * plane / 4), kSides);
    if (normalization != nullptr)
      reduce_rgb_s05<true>
          <<<grid, kThreads, 0, stream>>>(images, height, width, first, second, input, *normalization);
    else
      reduce_rgb_s05<false><<<grid, kThreads, 0, stream>>>(images, height, width, first, second, input, {});
  } else {
    reduce_rgb<<<dim3(blocks_for(kRgb * plane), kSides), kThreads, 0, stream>>>(images, height, width, first,
                                                                                second, input);
  }
  return cudaGetLastError();
}

cudaError_t launch_gmfss_pair_flow_output(cudaStream_t stream, const float* flow, int height, int width,
                                          bool half_flow, const std::array<const void*, 2>& half_images,
                                          const std::array<void*, 6>& flows, void* metric_input) {
  if (!usable_size(height, width) || flow == nullptr || metric_input == nullptr ||
      std::ranges::any_of(half_images, [](const void* ptr) { return ptr == nullptr; }) ||
      std::ranges::any_of(flows, [](const void* ptr) { return ptr == nullptr; }))
    return cudaErrorInvalidValue;
  const int plane = height * width;
  const auto out = [&](int i) { return static_cast<__half*>(flows[static_cast<std::size_t>(i)]); };
  const dim3 grid(blocks_for(plane), kSides);
  if (half_flow)
    flow_l1<true><<<grid, kThreads, 0, stream>>>(flow, height, width, out(0), out(3));
  else
    flow_l1<false><<<grid, kThreads, 0, stream>>>(flow, height, width, out(0), out(3));
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  const Flows levels{{out(0), out(3)}};
  flow_reduced<2>
      <<<dim3(blocks_for(plane / 4), kSides), kThreads, 0, stream>>>(levels, height, width, out(1), out(4));
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  flow_reduced<4>
      <<<dim3(blocks_for(plane / 16), kSides), kThreads, 0, stream>>>(levels, height, width, out(2), out(5));
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  const Images images{
      {static_cast<const __half*>(half_images[0]), static_cast<const __half*>(half_images[1])}};
  metric_inputs<<<blocks_for(plane), kThreads, 0, stream>>>(images, levels, height, width,
                                                            static_cast<__half*>(metric_input));
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
