// Débandage de l'image RGB d'une source, à sa taille, avant SPAN (KernelOps::deband) : l'algorithme de
// libplacebo (pl_shader_deband, une itération, sans grain), fait une fois par source au lieu de chaque image
// présentée en 4K, et avant que le réseau n'accentue les paliers d'un dégradé 8 bits.
#pragma once

#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

// Rayon et seuil de libplacebo (pl_deband_default_params : 16 pixels, 3 / 1000 du code) : un pixel prend la
// moyenne de quatre points tirés dans son rayon tant qu'elle reste sous le seuil, canal par canal ; un vrai
// trait s'en écarte davantage et reste net.
inline constexpr float kDebandRadius = 16.0F;
inline constexpr float kDebandThreshold = 3.0F / 1000.0F;

struct DebandParams {
  const void* in = nullptr;  // 1 × 3 × height × width, FP16, valeurs dans [0, 1]
  void* out = nullptr;       // même forme, distinct de in
  int width = 0;
  int height = 0;
};

cudaError_t launch_deband(const DebandParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
