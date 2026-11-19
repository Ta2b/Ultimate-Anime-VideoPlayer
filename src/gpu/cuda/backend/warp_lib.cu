// Le warp générique de la bibliothèque (warp_lib.h), sur les briques de warp_lib.cuh. Un fil par pixel de
// sortie ; réduite, il parcourt ses 2 × 2 points d'échantillonnage. Pas de relèvement en mémoire partagée :
// les clients de ce point d'entrée ne sont pas sur le chemin le plus chaud (la colle de RIFE a son noyau).
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/warp_lib.cuh"
#include "gpu/cuda/backend/warp_lib.h"

namespace kirari::gpu::cuda {
namespace {

constexpr int kMaxChannels = 16;

// Le canal c d'un tenseur FP16 de height × width points, en plans ou par point.
__device__ __forceinline__ float element(const __half* t, int height, int width, int pitch, int c, int y,
                                         int x) {
  const std::int64_t point = static_cast<std::int64_t>(y) * width + x;
  return __half2float(pitch == 0 ? t[static_cast<std::int64_t>(c) * height * width + point]
                                 : t[point * pitch + c]);
}

// Le canal c d'une grille grossière agrandie au point pleine résolution (px, py).
__device__ __forceinline__ float enlarged(const __half* t, int height, int width, int pitch, int c,
                                          const warp::Axis& ax, const warp::Axis& ay) {
  return warp::lerp2(element(t, height, width, pitch, c, ay.i0, ax.i0),
                     element(t, height, width, pitch, c, ay.i0, ax.i1),
                     element(t, height, width, pitch, c, ay.i1, ax.i0),
                     element(t, height, width, pitch, c, ay.i1, ax.i1), ax, ay);
}

// Spécialisé sur le filtre, et __maxnreg__ au lieu de __launch_bounds__ : nvcc 13 débordait sinon de quelques
// octets (--warn-on-spills) à 64 registres sur sm_89 et sm_120.
template <bool Cubic>
__global__ void __maxnreg__(128)
    warp_planes(WarpPlanesParams p, warp::Bounds window, int out_width, int out_height) {
  const int ox = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int oy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (ox >= out_width || oy >= out_height) return;
  const int d = p.reduce;
  const int taps = d > 1 ? 2 : 1;
  const int base = d > 1 ? d / 2 - 1 : 0;
  const auto* image = static_cast<const __half*>(p.image);
  const auto* fallback = static_cast<const __half*>(p.fallback);
  const auto* flow = static_cast<const __half*>(p.flow.planes);
  const auto* confidence = static_cast<const __half*>(p.confidence);
  const std::int64_t plane = static_cast<std::int64_t>(p.height) * p.width;
  const float inv = 1.0F / static_cast<float>(p.flow.scale);

  float acc[kMaxChannels];
#pragma unroll
  for (int c = 0; c < kMaxChannels; ++c) acc[c] = 0.0F;
  float conf_sum = 0.0F;
  for (int ty = 0; ty < taps; ++ty) {
    for (int tx = 0; tx < taps; ++tx) {
      const int px = d * ox + base + tx;
      const int py = d * oy + base + ty;
      float dx = 0.0F;
      float dy = 0.0F;
      float conf = 1.0F;
      if (flow != nullptr) {
        const warp::Axis ax = warp::coarse_axis(px, inv, p.flow.width);
        const warp::Axis ay = warp::coarse_axis(py, inv, p.flow.height);
        dx = enlarged(flow, p.flow.height, p.flow.width, p.flow.pitch, p.flow.channel, ax, ay) * p.flow.gain;
        dy = enlarged(flow, p.flow.height, p.flow.width, p.flow.pitch, p.flow.channel + 1, ax, ay) *
             p.flow.gain;
        if (confidence != nullptr)
          conf = fminf(fmaxf(enlarged(confidence, p.flow.height, p.flow.width, 0, 0, ax, ay), 0.0F), 1.0F);
      } else {
        const float fx = static_cast<float>(px);
        const float fy = static_cast<float>(py);
        dx = p.flow.affine[0] + p.flow.affine[1] * fx + p.flow.affine[2] * fy;
        dy = p.flow.affine[3] + p.flow.affine[4] * fx + p.flow.affine[5] * fy;
      }
      const warp::Tap tap = warp::tap_at(px, py, dx, dy, window);
      const std::int64_t at = static_cast<std::int64_t>(py) * p.width + px;
#pragma unroll
      for (int c = 0; c < kMaxChannels; ++c) {
        if (c < p.channels) {
          const __half* src = image + c * plane;
          float v = Cubic ? warp::catmull_rom(src, tap, p.width, window) : warp::bilinear(src, tap, p.width);
          if (fallback != nullptr) v = conf * v + (1.0F - conf) * __half2float(fallback[c * plane + at]);
          acc[c] += v;
        }
      }
      conf_sum += conf;
    }
  }
  const float norm = 1.0F / static_cast<float>(taps * taps);
  const std::int64_t o = static_cast<std::int64_t>(oy) * out_width + ox;
  auto* out = static_cast<__half*>(p.out);
#pragma unroll
  for (int c = 0; c < kMaxChannels; ++c) {
    if (c < p.channels) {
      const __half h = __float2half_rn(acc[c] * norm);
      if (p.out_pitch == 0) {
        out[static_cast<std::int64_t>(c) * out_height * out_width + o] = h;
      } else {
        out[o * p.out_pitch + c] = h;
      }
    }
  }
  if (p.confidence_out != nullptr)
    static_cast<__half*>(p.confidence_out)[o] = __float2half_rn(conf_sum * norm);
}

}  // namespace

cudaError_t launch_warp_planes(const WarpPlanesParams& params, cudaStream_t stream) {
  const WarpPlanesParams& p = params;
  if (p.image == nullptr || p.out == nullptr || p.channels < 1 || p.channels > kMaxChannels ||
      p.height <= 0 || p.width <= 0 || p.reduce < 1 || (p.reduce > 1 && p.reduce % 2 != 0) ||
      p.flow.scale < 1 || (p.out_pitch != 0 && p.out_pitch < p.channels) ||
      (p.flow.planes != nullptr && (p.flow.height <= 0 || p.flow.width <= 0)))
    return cudaErrorInvalidValue;
  warp::Bounds window{0, 0, p.width, p.height};
  if (p.window_width > 0 && p.window_height > 0)
    window = {p.window_left, p.window_top, p.window_left + p.window_width, p.window_top + p.window_height};
  const int out_width = p.width / p.reduce;
  const int out_height = p.height / p.reduce;
  const dim3 block(32, 8);
  const dim3 grid((static_cast<unsigned>(out_width) + 31U) / 32U,
                  (static_cast<unsigned>(out_height) + 7U) / 8U);
  if (p.cubic != 0) {
    warp_planes<true><<<grid, block, 0, stream>>>(params, window, out_width, out_height);
  } else {
    warp_planes<false><<<grid, block, 0, stream>>>(params, window, out_width, out_height);
  }
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
