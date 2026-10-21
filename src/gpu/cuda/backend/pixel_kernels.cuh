// Règles communes des noyaux pixel du backend TensorRT (étude 3) : blocs de 256 fils (16 × 16, ou
// 32 × 8 pour la mesure), au moins quatre blocs résidents par SM, soit 64 registres au plus par fil ;
// trois pour les noyaux lourds (warp, Area : 85 registres, la moitié des fils d'un SM d'Ada). ptxas
// refuse tout débordement de registres (--warn-on-spills en erreur, CMakeLists.txt).
// Réductions d'un bloc par mélange de registres, sans arbre en mémoire partagée.
#pragma once

#include <cuda_runtime.h>

namespace ajp::gpu::cuda {

inline constexpr int kPixelThreads = 256;
inline constexpr int kPixelBlocksPerSm = 4;
inline constexpr int kHeavyBlocksPerSm = 3;
inline constexpr int kWarpSize = 32;

// Combine value sur tout le bloc (op associative, identity neutre) ; le résultat vaut dans le fil 0.
// Chaque fil du bloc doit l'appeler, y compris hors de l'image (avec identity).
template <typename Op>
__device__ float block_reduce(float value, float identity, Op op) {
  __shared__ float warps[kPixelThreads / kWarpSize];
  const int t = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
    value = op(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
  if (t % kWarpSize == 0) warps[t / kWarpSize] = value;
  __syncthreads();
  const int count = static_cast<int>(blockDim.x * blockDim.y) / kWarpSize;
  value = t < count ? warps[t] : identity;
  if (t < kWarpSize) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
      value = op(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
  }
  return value;
}

struct Sum {
  __device__ float operator()(float a, float b) const { return a + b; }
};
struct Max {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

}  // namespace ajp::gpu::cuda
