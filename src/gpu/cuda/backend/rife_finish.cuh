// La fin de RIFE fusionné en un point pleine résolution (rife_fused_kernels.h) : le flot du dernier bloc
// ajouté au flot accumulé, puis l'image (warp des deux images, masque sigmoïde, mélange) ou le flot final et
// le masque. Partagée par la fin (fused_finish, rife_fused_kernels.cu) et par la dernière couche du dernier
// bloc du RIFE natif, qui la fait dans son épilogue sans écrire Y (rife_native_finish.cu) : les mêmes
// fonctions, donc les mêmes arrondis.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/rife_fused_kernels.h"
#include "gpu/cuda/backend/warp_lib.cuh"

namespace kirari::gpu::cuda::finish {

// Le flot accumulé au pixel i (canaux 24 à 27 de X du dernier bloc) : une lecture de 8 octets par pixel
// (pas 32), ou quatre plans de plane éléments.
__device__ __forceinline__ void accumulated(const FusedFinishParams& p, std::size_t i, std::size_t plane,
                                            float (&acc)[4]) {
  const auto* last_x = static_cast<const __half*>(p.x_last);
  if (p.x_pitch > 0) {
    const uint2 word = *reinterpret_cast<const uint2*>(last_x + i * static_cast<std::size_t>(p.x_pitch) + 24);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&word.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&word.y));
    acc[0] = a.x;
    acc[1] = a.y;
    acc[2] = b.x;
    acc[3] = b.y;
  } else {
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] = __half2float(last_x[static_cast<std::size_t>(24 + k) * plane + i]);
  }
}

// La fin au point (x, y), pixel i d'un plan de plane éléments : acc le flot accumulé, z les cinq canaux de Z
// du dernier bloc en ce point (son flot, puis le masque avant la sigmoïde).
__device__ __forceinline__ void at(const FusedFinishParams& p, int x, int y, std::size_t i, std::size_t plane,
                                   const float (&acc)[4], const float (&z)[5]) {
  using warp::rounded;
  float fl[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) fl[k] = rounded(acc[k] + z[k]);
  const float sig = rounded(1.0F / (1.0F + __expf(-z[4])));
  if (p.image == nullptr) {  // variante mouvement : le flot final et le masque
    auto* flow = static_cast<__half*>(p.flow);
    auto* out_mask = static_cast<__half*>(p.mask);
#pragma unroll
    for (int k = 0; k < 4; ++k) flow[static_cast<std::size_t>(k) * plane + i] = __float2half_rn(fl[k]);
    out_mask[i] = __float2half_rn(sig);
    return;
  }
  const auto* input = static_cast<const __half*>(p.input);
  float warped[2][3];
  const warp::Bounds window{p.window_left, p.window_top, p.window_left + p.window_width,
                            p.window_top + p.window_height};
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const warp::Tap tap = warp::tap_at(x, y, fl[2 * n], fl[2 * n + 1], window);
#pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
      const __half* source = input + static_cast<std::size_t>(3 * n + ch) * plane;
      warped[n][ch] = rounded(p.cubic != 0 ? warp::catmull_rom(source, tap, p.width, window)
                                           : warp::bilinear(source, tap, p.width));
    }
  }
  const float rest = rounded(1.0F - sig);
  auto* image = static_cast<__half*>(p.image);
#pragma unroll
  for (int ch = 0; ch < 3; ++ch)
    image[static_cast<std::size_t>(ch) * plane + i] =
        __float2half_rn(rounded(warped[0][ch] * sig) + rounded(warped[1][ch] * rest));
}

}  // namespace kirari::gpu::cuda::finish
