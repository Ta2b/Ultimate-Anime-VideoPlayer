#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "prepare/gmfss_pair_grid.h"

namespace kirari::gpu::cuda::gmfss_metric {

struct MotionPlanes {
  __half* metric[2][3];  // le troisième plan de motion{k}_{l}
};

// La métrique du côté side au pixel (y, x) : tanh arrondi en FP16, puis × 10 arrondi, comme Tanh et Mul de la
// paire en FP16.
__device__ __forceinline__ float metric_at(const __half* m, int w, int y, int x, int side) {
  const __half t = __float2half(tanhf(__half2float(m[(static_cast<long>(y) * w + x) * 8 + side])));
  return __half2float(__float2half(__half2float(t) * prepare::kGmfssMetricScale));
}

// Un fil possède la même abscisse dans quatre lignes. Chaque Tanh et ses deux arrondis
// sont faits une fois, et le voisin à droite fournit l'autre colonne via un échange de registre.
template <bool Complete>
__global__ void metric_motion(const __half* m, int h, int w, MotionPlanes out) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = 4 * static_cast<int>(blockIdx.y);
  unsigned active = 0xffffffffU;
  if constexpr (!Complete) active = __ballot_sync(0xffffffffU, x < w);
  if (x >= w) return;  // au chemin complet, la largeur est un multiple du warp
#pragma unroll
  for (int side = 0; side < 2; ++side) {
    float column[4];
    float right[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      if (Complete || y + row < h) {
        column[row] = metric_at(m, w, y + row, x, side);
        out.metric[side][0][static_cast<long>(y + row) * w + x] = __float2half(column[row]);
      } else {
        column[row] = 0.F;
      }
      right[row] = __shfl_down_sync(active, column[row], 1);
    }
    if ((x & 1) == 0 && (Complete || x + 1 < w)) {
      const long i = static_cast<long>(y / 2) * (w / 2) + x / 2;
      if (Complete || y + 1 < h) {
        const float top = column[0] + right[0] + column[1] + right[1];
        out.metric[side][1][i] = __float2half(0.25F * top);
      }
      if (Complete || y + 3 < h) {
        const float bottom = column[2] + right[2] + column[3] + right[3];
        out.metric[side][1][i + w / 2] = __float2half(0.25F * bottom);
      }
    }
    if ((x & 3) == 1 && (Complete || (x + 2 < w && y + 3 < h))) {
      const float middle = column[1] + right[1] + column[2] + right[2];
      out.metric[side][2][static_cast<long>(y / 4) * (w / 4) + x / 4] = __float2half(0.25F * middle);
    }
  }
}

inline cudaError_t launch_metric_motion(cudaStream_t stream, const __half* logits, int h, int w,
                                        MotionPlanes planes) {
  if (h <= 0 || w <= 0) return cudaErrorInvalidValue;
  constexpr unsigned threads = 128;
  const dim3 grid((static_cast<unsigned>(w) + threads - 1) / threads, (static_cast<unsigned>(h) + 3U) / 4U);
  if (h % 4 == 0 && w % 32 == 0)
    metric_motion<true><<<grid, threads, 0, stream>>>(logits, h, w, planes);
  else
    metric_motion<false><<<grid, threads, 0, stream>>>(logits, h, w, planes);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda::gmfss_metric
