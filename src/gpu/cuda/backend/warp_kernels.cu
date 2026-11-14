// Noyaux du warp (warp_kernels.h), calqués sur kernels/ref/warp.cpp.
#include <cstdint>
#include <cuda_fp16.h>
#include <type_traits>

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
  if (!p.from_middle) {
    ax = at_uv[0];
    ay = at_uv[1];
    bx = at_uv[2];
    by = at_uv[3];
    weight = S::sample(mask, 0, u, v);
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
    const float tx = at2[2] - at2[0];
    const float ty = at2[3] - at2[1];
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

// Mouvement par tuile (WarpBlendParams::tiled_motion, essai) : le mouvement estimé au milieu, mis à l'instant
// de la sortie, se calcule une fois par point du flot que couvre le bloc, en mémoire partagée, au lieu d'une
// fois par pixel de la sortie (neuf pixels par point en 4K pour un flot en 1280). Chaque pixel interpole
// ensuite tx, ty et le poids entre les quatre points qui l'entourent : le calcul n'étant pas linéaire (second
// prélèvement, sigmoïde), l'image change un peu. Sans les poses ; sortie au moins aussi grande que la
// fenêtre.
inline constexpr int kTilePoints = kBlock + 2;  // un bloc de 16 pixels couvre au plus 17 points du flot

struct TilePoint {
  float tx;
  float ty;
  float weight;
};

__device__ TilePoint middle_point(const WarpBlendParams& p, float u, float v) {
  const Planes flow = motion_planes(p, p.flow, 0);
  const Planes mask = motion_planes(p, p.mask, 0);
  const float t = p.time;
  float at_uv[4];
  Exact::sample_n(flow, 0, u, v, at_uv);
  const float u2 = u - ((1.0F - t) * at_uv[0] + t * at_uv[2]);
  const float v2 = v - ((1.0F - t) * at_uv[1] + t * at_uv[3]);
  float at2[4];
  Exact::sample_n(flow, 0, u2, v2, at2);
  const float tx = at2[2] - at2[0];
  const float ty = at2[3] - at2[1];
  const float mm =
      fminf(fmaxf(Exact::sample(mask, 0, u2, v2), kernels::kMaskEpsilon), 1.0F - kernels::kMaskEpsilon);
  const float z = kernels::kBlendLawA * logf(mm / (1.0F - mm)) + kernels::kBlendLawB * logf((1.0F - t) / t) +
                  kernels::kBlendLawC;
  return {.tx = tx, .ty = ty, .weight = 1.0F / (1.0F + expf(-z))};
}

// Le premier point du flot qu'un bord de bloc lit, comme Exact::sample (position bornée, partie entière).
__device__ int first_point(int pixel, float scale, int size) {
  const float u = (static_cast<float>(pixel) + 0.5F) / scale - 0.5F;
  return static_cast<int>(fminf(fmaxf(u, 0.0F), static_cast<float>(size - 1)));
}

template <bool kCubic>
__global__ void __launch_bounds__(kPixelThreads, kWarpBlocksPerSm<kCubic>)
    warp_blend_tiled(WarpBlendParams p) {
  __shared__ TilePoint tile[kTilePoints][kTilePoints];
  const float sx = static_cast<float>(p.width) / static_cast<float>(p.window_width);
  const float sy = static_cast<float>(p.height) / static_cast<float>(p.window_height);
  const int x_first = static_cast<int>(blockIdx.x * blockDim.x);
  const int y_first = static_cast<int>(blockIdx.y * blockDim.y);
  const int gx = first_point(x_first, sx, p.window_width);
  const int gy = first_point(y_first, sy, p.window_height);
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  for (int k = tid; k < kTilePoints * kTilePoints; k += kPixelThreads) {
    const int i = k % kTilePoints;
    const int j = k / kTilePoints;
    const int px = min(gx + i, p.window_width - 1);
    const int py = min(gy + j, p.window_height - 1);
    tile[j][i] = middle_point(p, static_cast<float>(px), static_cast<float>(py));
  }
  __syncthreads();
  const int x = x_first + static_cast<int>(threadIdx.x);
  const int y = y_first + static_cast<int>(threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const float u =
      fminf(fmaxf((static_cast<float>(x) + 0.5F) / sx - 0.5F, 0.0F), static_cast<float>(p.window_width - 1));
  const float v =
      fminf(fmaxf((static_cast<float>(y) + 0.5F) / sy - 0.5F, 0.0F), static_cast<float>(p.window_height - 1));
  const int x0 = static_cast<int>(u);
  const int y0 = static_cast<int>(v);
  const float fx = u - static_cast<float>(x0);
  const float fy = v - static_cast<float>(y0);
  const int i0 = min(x0 - gx, kTilePoints - 2);
  const int j0 = min(y0 - gy, kTilePoints - 2);
  const auto lerp = [&](float TilePoint::* field) {
    const float top = tile[j0][i0].*field * (1.0F - fx) + tile[j0][i0 + 1].*field * fx;
    const float bottom = tile[j0 + 1][i0].*field * (1.0F - fx) + tile[j0 + 1][i0 + 1].*field * fx;
    return top * (1.0F - fy) + bottom * fy;
  };
  const float t = p.time;
  const float tx = lerp(&TilePoint::tx);
  const float ty = lerp(&TilePoint::ty);
  float rgb[3];
  blend_images<Exact, kCubic>(p, x, y,
                              Shift{.ax = -t * tx,
                                    .ay = -t * ty,
                                    .bx = (1.0F - t) * tx,
                                    .by = (1.0F - t) * ty,
                                    .weight = lerp(&TilePoint::weight)},
                              rgb);
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

cudaError_t launch_warp_blend(const WarpBlendParams& params, cudaStream_t stream) {
  if (params.tiled_motion && params.from_middle && !params.keep_poses &&
      params.width >= params.window_width && params.height >= params.window_height) {
    const dim3 grid = grid_for(params.width, params.height);
    if (params.cubic)
      warp_blend_tiled<true><<<grid, dim3(kBlock, kBlock), 0, stream>>>(params);
    else
      warp_blend_tiled<false><<<grid, dim3(kBlock, kBlock), 0, stream>>>(params);
    return cudaGetLastError();
  }
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
