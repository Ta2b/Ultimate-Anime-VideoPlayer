// Règles communes des noyaux pixel du backend TensorRT (étude 3) : blocs de 256 fils (16 × 16, ou
// 32 × 8 pour la mesure), au moins quatre blocs résidents par SM, soit 64 registres au plus par fil ;
// trois pour les noyaux lourds (warp, Area : 85 registres, la moitié des fils d'un SM d'Ada). ptxas
// refuse tout débordement de registres (--warn-on-spills en erreur, CMakeLists.txt).
// Réductions d'un bloc par mélange de registres, sans arbre en mémoire partagée.
#pragma once

#include <cuda_fp16.h>
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

// kernels/spec/color.h unorm16 : clamp, × 65535, + 0,5, troncature.
__device__ inline unsigned short unorm16(float v) {
  v = v != v ? 0.0F : fminf(fmaxf(v, 0.0F), 1.0F);  // NaN donne 0
  return static_cast<unsigned short>(v * 65535.0F + 0.5F);
}

// Un pixel RGB FP16 dans un slot RGBA16 UNORM du ring (ref::to_rgba16), alpha opaque : le code que
// write_rgba16 écrirait du tenseur FP16, au bit près.
__device__ inline void store_rgba16(cudaSurfaceObject_t surface, int x, int y, __half r, __half g, __half b) {
  const ushort4 pixel =
      make_ushort4(unorm16(__half2float(r)), unorm16(__half2float(g)), unorm16(__half2float(b)), 65535);
  surf2Dwrite(pixel, surface, x * static_cast<int>(sizeof(ushort4)), y);
}

struct Sum {
  __device__ float operator()(float a, float b) const { return a + b; }
};
struct Max {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

}  // namespace ajp::gpu::cuda
