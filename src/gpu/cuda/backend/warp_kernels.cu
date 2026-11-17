// Noyaux du warp (warp_kernels.h), calqués sur kernels/ref/warp.cpp.
#include <cstdint>
#include <cuda_fp16.h>
#include <type_traits>

#include "gpu/cuda/backend/bilinear.cuh"
#include "gpu/cuda/backend/pixel_kernels.cuh"
#include "gpu/cuda/backend/warp_kernels.h"
#include "kernels/spec/poses.h"
#include "kernels/spec/warp.h"
#include "kernels/spec/warp_pan.h"

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

// L'instant réglé du pixel (u, v) pour garder les poses (keeping_poses de ref/warp.cpp). at_uv : les quatre
// canaux du flot lus en (u, v), les mêmes à chaque itération ; chaque point se lit en une fois (sample_n),
// même résultat que canal par canal.
template <typename S>
__device__ float pose_time_at(const WarpBlendParams& p, float u, float v, const float (&at_uv)[4]) {
  const Planes flow = motion_planes(p, p.flow, p.textures.flow);
  const Planes previous = motion_planes(p, p.previous_flow, p.textures.previous_flow);
  const float t = p.time;
  float tp = t;
  for (int k = 0; k < kernels::kPoseIterations; ++k) {
    const float u2 = u - ((1.0F - tp) * at_uv[0] + tp * at_uv[2]);
    const float v2 = v - ((1.0F - tp) * at_uv[1] + tp * at_uv[3]);
    float at2[4];
    S::sample_n(flow, 0, u2, v2, at2);
    const float dx = at2[2] - at2[0];
    const float dy = at2[3] - at2[1];
    float before = 0.0F;
    if (p.previous_flow != nullptr) {
      const float ax = u - tp * dx;
      const float ay = v - tp * dy;
      float qx = ax;
      float qy = ay;
      for (int step = 0; step < 2; ++step) {
        float back[2];
        S::sample_n(previous, 2, qx, qy, back);
        qx = ax - back[0];
        qy = ay - back[1];
      }
      float at_q[4];
      S::sample_n(previous, 0, qx, qy, at_q);
      before = hypotf(at_q[2] - at_q[0], at_q[3] - at_q[1]);
    }
    const float now = hypotf(dx, dy);
    const float r = (before + kernels::kPoseEpsilon) / (before + now + 2.0F * kernels::kPoseEpsilon);
    // kernels::pose_time, réécrit ici : nvcc n'appelle pas une fonction constexpr de l'hôte.
    const float timed = t < 0.5F ? 2.0F * t * r : 1.0F - 2.0F * (1.0F - t) * r;
    tp = fminf(fmaxf(timed, kernels::kPoseTimeLimit), 1.0F - kernels::kPoseTimeLimit);
  }
  return tp;
}

// Le déplacement d'un pixel de la sortie vers a et vers b, en pixels de la fenêtre du mouvement, et le poids
// de a au mélange.
struct Shift {
  float ax = 0.0F;
  float ay = 0.0F;
  float bx = 0.0F;
  float by = 0.0F;
  float weight = 0.5F;
};

// Les deux images lues au déplacement du pixel (x, y), mises à l'échelle de la sortie, puis mélangées.
template <typename S, bool kCubic>
__device__ void blend_images(const WarpBlendParams& p, int x, int y, const Shift& m, float* rgb) {
  const float sx = static_cast<float>(p.width) / static_cast<float>(p.window_width);
  const float sy = static_cast<float>(p.height) / static_cast<float>(p.window_height);
  const Planes a = image_planes(p, p.a, p.textures.a);
  const Planes b = image_planes(p, p.b, p.textures.b);
  const float fx = static_cast<float>(x);
  const float fy = static_cast<float>(y);
  using Image = std::conditional_t<kCubic, CatmullRom, S>;
  for (int c = 0; c < 3; ++c) {
    const float va = Image::sample(a, c, fx + sx * m.ax, fy + sy * m.ay);
    const float vb = Image::sample(b, c, fx + sx * m.bx, fy + sy * m.by);
    rgb[c] = va * m.weight + vb * (1.0F - m.weight);
  }
}

// Mouvement d'ensemble (kernels/spec/warp_pan.h) : le flot lu f remplacé par la translation de la paire au
// milieu, en pixels de la fenêtre, là où il est petit et proche d'elle.
__device__ void pan_substitute(const WarpBlendParams& p, float sx, float sy, float (&f)[4]) {
  if (p.pan == nullptr || __ldg(p.pan + 2) < 0.5F) return;
  const float dx = __ldg(p.pan);
  const float dy = __ldg(p.pan + 1);
  const float g[4] = {-0.5F * dx / sx, -0.5F * dy / sy, 0.5F * dx / sx, 0.5F * dy / sy};
  float m = 0.0F;
  float e = 0.0F;
  for (int c = 0; c < 4; ++c) {
    m = fmaxf(m, fabsf(f[c]));
    e = fmaxf(e, fabsf(f[c] - g[c]));
  }
  if (m < kernels::kPanFlowLimit && e < kernels::kPanAgreement)
    for (int c = 0; c < 4; ++c) f[c] = g[c];
}

// kernels::slow_fade_weight, réécrit ici : nvcc n'appelle pas une fonction constexpr de l'hôte.
__device__ float slow_fade(float d) {
  const float s =
      fminf(fmaxf((d - kernels::kSlowFadeLo) / (kernels::kSlowFadeHi - kernels::kSlowFadeLo), 0.0F), 1.0F);
  return s * s * (3.0F - 2.0F * s);
}

// Les trois canaux du pixel (x, y) de la sortie : mouvement lu dans la fenêtre, puis les deux
// images déformées et mélangées. kCubic : les images lues par Catmull-Rom (option cubic, éteinte par
// défaut), dans des noyaux à eux.
template <typename S, bool kCubic>
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
  float at_uv[4];
  S::sample_n(flow, 0, u, v, at_uv);
  const bool pan = p.pan != nullptr && !p.keep_poses;
  if (pan) pan_substitute(p, sx, sy, at_uv);
  if (!p.from_middle) {
    ax = at_uv[0];
    ay = at_uv[1];
    bx = at_uv[2];
    by = at_uv[3];
    weight = S::sample(mask, 0, u, v);
    if (pan) {  // sous un demi-pixel, le masque glisse vers le fondu (ln((1 − t) / t), nul à t = 1/2)
      const float mm = fminf(fmaxf(weight, kernels::kMaskEpsilon), 1.0F - kernels::kMaskEpsilon);
      const float w = slow_fade(hypotf(bx - ax, by - ay));
      const float t = p.time;
      weight = 1.0F / (1.0F + expf(-(w * logf(mm / (1.0F - mm)) + (1.0F - w) * logf((1.0F - t) / t))));
    }
  } else {
    // L'instant réglé se lit toujours exactement : le rapport des deux mouvements amplifie la
    // fraction sur 8 bits de la texture (émulée sur les images du test : écart porté de 2 à 10/1024).
    float t = p.time;
    if (p.keep_poses) {
      if constexpr (std::is_same_v<S, Exact>) {
        t = pose_time_at<Exact>(p, u, v, at_uv);
      } else {
        float exact_uv[4];
        Exact::sample_n(flow, 0, u, v, exact_uv);
        t = pose_time_at<Exact>(p, u, v, exact_uv);
      }
    }
    const float u2 = u - ((1.0F - t) * at_uv[0] + t * at_uv[2]);
    const float v2 = v - ((1.0F - t) * at_uv[1] + t * at_uv[3]);
    float at2[4];
    S::sample_n(flow, 0, u2, v2, at2);
    if (pan) pan_substitute(p, sx, sy, at2);
    const float tx = at2[2] - at2[0];
    const float ty = at2[3] - at2[1];
    const float mm =
        fminf(fmaxf(S::sample(mask, 0, u2, v2), kernels::kMaskEpsilon), 1.0F - kernels::kMaskEpsilon);
    float z = kernels::kBlendLawA * logf(mm / (1.0F - mm)) + kernels::kBlendLawB * logf((1.0F - t) / t) +
              kernels::kBlendLawC;
    if (pan) {
      const float w = slow_fade(hypotf(tx, ty));
      z = w * z + (1.0F - w) * logf((1.0F - t) / t);
    }
    ax = -t * tx;
    ay = -t * ty;
    bx = (1.0F - t) * tx;
    by = (1.0F - t) * ty;
    weight = 1.0F / (1.0F + expf(-z));
  }
  blend_images<S, kCubic>(p, x, y, Shift{.ax = ax, .ay = ay, .bx = bx, .by = by, .weight = weight}, rgb);
}

// Les noyaux de Catmull-Rom ont leur propre borne : ses 16 lectures par canal ne tiennent pas dans les
// registres de la borne lourde (nvcc débordait de 8 octets en mémoire locale, --warn-on-spills), et le
// chemin bilinéaire, celui par défaut, garde la sienne.
inline constexpr int kCubicBlocksPerSm = 2;
template <bool kCubic>
inline constexpr int kWarpBlocksPerSm = kCubic ? kCubicBlocksPerSm : kHeavyBlocksPerSm;

template <bool kCubic>
__global__ void __launch_bounds__(kPixelThreads, kWarpBlocksPerSm<kCubic>) warp_blend(WarpBlendParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  float rgb[3];
  warp_pixel<Exact, kCubic>(p, x, y, rgb);
  if (p.surface.object != 0) {
    store_ring(p.surface, x, y, __float2half_rn(rgb[0]), __float2half_rn(rgb[1]), __float2half_rn(rgb[2]));
    return;
  }
  auto* out = static_cast<__half*>(p.out);
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x);
  for (int c = 0; c < 3; ++c) out[static_cast<std::size_t>(c) * plane + i] = __float2half_rn(rgb[c]);
}

// Deux pixels voisins par fil, lus par les objets texture : un bloc 16 × 16 couvre 32 × 16 pixels.
template <bool kCubic>
__global__ void __launch_bounds__(kPixelThreads, kWarpBlocksPerSm<kCubic>)
    warp_blend_filtered(WarpBlendParams p) {
  const int x = 2 * static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;  // largeur paire : x + 1 est dans l'image
  // Un pixel après l'autre : déroulés, les deux calculs entrelacés débordent de la borne lourde
  // sur sm_86 et sm_89 (les poses lues exactement).
  __half2 pair[3];
#pragma unroll 1
  for (int k = 0; k < 2; ++k) {
    float rgb[3];
    warp_pixel<Filtered, kCubic>(p, x + k, y, rgb);
    for (int c = 0; c < 3; ++c) {
      const __half v = __float2half_rn(rgb[c]);
      pair[c] = k == 0 ? __halves2half2(v, v) : __halves2half2(__low2half(pair[c]), v);
    }
  }
  if (p.surface.object != 0) {
    store_ring(p.surface, x, y, __low2half(pair[0]), __low2half(pair[1]), __low2half(pair[2]));
    store_ring(p.surface, x + 1, y, __high2half(pair[0]), __high2half(pair[1]), __high2half(pair[2]));
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
  __shared__ unsigned int bins[kernels::kFlowBins];
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  for (int b = tid; b < kernels::kFlowBins; b += static_cast<int>(blockDim.x * blockDim.y)) bins[b] = 0U;
  __syncthreads();
  if (x < p.window_width && y < p.window_height) {
    float m = 0.0F;
    const auto* flow = static_cast<const __half*>(p.flow);
    const std::size_t plane =
        static_cast<std::size_t>(p.plane_width) * static_cast<std::size_t>(p.plane_height);
    const std::size_t i = static_cast<std::size_t>(p.window_y + y) * static_cast<std::size_t>(p.plane_width) +
                          static_cast<std::size_t>(p.window_x + x);
    for (int c = 0; c < 4; ++c)
      m = fmaxf(m, fabsf(__half2float(flow[static_cast<std::size_t>(c) * plane + i])));
    // Même case que kernels::flow_bin (le quart de pixel est exact en flottant).
    const float scaled = m * 4.0F;
    atomicAdd(&bins[scaled >= 255.0F ? 255 : static_cast<int>(scaled)], 1U);
  }
  __syncthreads();
  // L'histogramme du bloc s'ajoute à celui de la paire.
  for (int b = tid; b < kernels::kFlowBins; b += static_cast<int>(blockDim.x * blockDim.y)) {
    if (bins[b] != 0U) atomicAdd(&p.result[b], bins[b]);
  }
}

}  // namespace

// Mouvement d'ensemble (kernels/spec/warp_pan.h) : un bloc de 16 × 16 fils, kPanGrid² points, réduction en
// mémoire partagée. La validité ne dépend que des gradients de a : la même à chaque itération.
__device__ float pan_luma(const __half* img, int plane, int width, int x, int y) {
  const int i = y * width + x;
  return kernels::kPanLumaR * __half2float(img[i]) + kernels::kPanLumaG * __half2float(img[plane + i]) +
         kernels::kPanLumaB * __half2float(img[2 * plane + i]);
}

__device__ float pan_luma_bilinear(const __half* img, int plane, int width, int height, float x, float y) {
  x = fminf(fmaxf(x, 0.0F), static_cast<float>(width - 1));
  y = fminf(fmaxf(y, 0.0F), static_cast<float>(height - 1));
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = min(x0 + 1, width - 1);
  const int y1 = min(y0 + 1, height - 1);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const float top =
      pan_luma(img, plane, width, x0, y0) * (1.0F - fx) + pan_luma(img, plane, width, x1, y0) * fx;
  const float bottom =
      pan_luma(img, plane, width, x0, y1) * (1.0F - fx) + pan_luma(img, plane, width, x1, y1) * fx;
  return top * (1.0F - fy) + bottom * fy;
}

// kernels::pan_point, réécrit ici (constexpr de l'hôte).
__device__ int pan_point(int i, int n) {
  const int m = max(4, n / 12);
  return m + (i * (n - 2 * m)) / kernels::kPanGrid;
}

constexpr int kPanThreads = kBlock * kBlock;
constexpr int kPanPerThread = kernels::kPanGrid * kernels::kPanGrid / kPanThreads;

__global__ void __launch_bounds__(kPanThreads) pan_global(WarpBlendParams p, float* out) {
  __shared__ float sums[5][kPanThreads];
  __shared__ float shift[3];
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const auto* a = static_cast<const __half*>(p.a);
  const auto* b = static_cast<const __half*>(p.b);
  const int plane = p.width * p.height;
  float la[kPanPerThread];
  float gx[kPanPerThread];
  float gy[kPanPerThread];
  int px[kPanPerThread];
  int py[kPanPerThread];
  for (int k = 0; k < kPanPerThread; ++k) {
    const int q = tid + k * kPanThreads;
    px[k] = pan_point(q % kernels::kPanGrid, p.width);
    py[k] = pan_point(q / kernels::kPanGrid, p.height);
    la[k] = pan_luma(a, plane, p.width, px[k], py[k]);
    gx[k] = 0.5F *
            (pan_luma(a, plane, p.width, px[k] + 1, py[k]) - pan_luma(a, plane, p.width, px[k] - 1, py[k]));
    gy[k] = 0.5F *
            (pan_luma(a, plane, p.width, px[k], py[k] + 1) - pan_luma(a, plane, p.width, px[k], py[k] - 1));
  }
  if (tid == 0) {
    shift[0] = 0.0F;
    shift[1] = 0.0F;
    shift[2] = 0.0F;
  }
  __syncthreads();
  for (int it = 0; it < kernels::kPanIterations; ++it) {
    const float dx = shift[0];
    const float dy = shift[1];
    float s[5] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (int k = 0; k < kPanPerThread; ++k) {
      const float r = pan_luma_bilinear(b, plane, p.width, p.height, static_cast<float>(px[k]) + dx,
                                        static_cast<float>(py[k]) + dy) -
                      la[k];
      s[0] += gx[k] * gx[k];
      s[1] += gx[k] * gy[k];
      s[2] += gy[k] * gy[k];
      s[3] += gx[k] * r;
      s[4] += gy[k] * r;
    }
    for (int c = 0; c < 5; ++c) sums[c][tid] = s[c];
    __syncthreads();
    for (int h = kPanThreads / 2; h > 0; h >>= 1) {
      if (tid < h)
        for (int c = 0; c < 5; ++c) sums[c][tid] += sums[c][tid + h];
      __syncthreads();
    }
    if (tid == 0) {
      const float sxx = sums[0][0];
      const float sxy = sums[1][0];
      const float syy = sums[2][0];
      const float det = sxx * syy - sxy * sxy;
      const bool valid = det > kernels::kPanDetRatio * (sxx + syy) * (sxx + syy) && det > 0.0F;
      if (valid) {
        shift[0] = dx - (syy * sums[3][0] - sxy * sums[4][0]) / det;
        shift[1] = dy - (sxx * sums[4][0] - sxy * sums[3][0]) / det;
      }
      shift[2] = valid ? 1.0F : 0.0F;
    }
    __syncthreads();
    if (shift[2] < 0.5F) break;
  }
  if (tid == 0) {
    out[0] = shift[0];
    out[1] = shift[1];
    out[2] = shift[2] > 0.5F && hypotf(shift[0], shift[1]) < kernels::kPanMaxShift ? 1.0F : 0.0F;
  }
}

cudaError_t launch_pan_global(const WarpBlendParams& params, float* out, cudaStream_t stream) {
  if (params.width < 16 || params.height < 16) return cudaMemsetAsync(out, 0, 3 * sizeof(float), stream);
  pan_global<<<1, dim3(kBlock, kBlock), 0, stream>>>(params, out);
  return cudaGetLastError();
}

cudaError_t launch_warp_blend(const WarpBlendParams& params, cudaStream_t stream) {
  const WarpBlendParams::Textures& t = params.textures;
  const bool previous = params.keep_poses && params.previous_flow != nullptr;
  if (t.a != 0 && t.b != 0 && t.flow != 0 && t.mask != 0 && (!previous || t.previous_flow != 0) &&
      params.width % 2 == 0 && reinterpret_cast<std::uintptr_t>(params.out) % 4 == 0) {
    const dim3 grid = grid_for(params.width / 2, params.height);
    if (params.cubic)
      warp_blend_filtered<true><<<grid, dim3(kBlock, kBlock), 0, stream>>>(params);
    else
      warp_blend_filtered<false><<<grid, dim3(kBlock, kBlock), 0, stream>>>(params);
  } else {
    const dim3 grid = grid_for(params.width, params.height);
    if (params.cubic)
      warp_blend<true><<<grid, dim3(kBlock, kBlock), 0, stream>>>(params);
    else
      warp_blend<false><<<grid, dim3(kBlock, kBlock), 0, stream>>>(params);
  }
  return cudaGetLastError();
}

cudaError_t launch_flow_max(const FlowMaxParams& params, cudaStream_t stream) {
  flow_max<<<grid_for(params.window_width, params.window_height), dim3(kBlock, kBlock), 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
