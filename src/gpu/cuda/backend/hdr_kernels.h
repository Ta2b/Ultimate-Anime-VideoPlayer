// Noyaux CUDA de RTX HDR (TrueHDR) dans la tranche verticale : l'image RGB FP16 de SPAN vers
// l'entrée 8 bits de Maxine, tramée avant la quantification (AH-392 : sans tramage, les quasi-noirs
// se crantent avant même TrueHDR), puis la sortie RGB10A2 (PQ BT.2020) vers le slot RGBA16 du ring.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

struct ToRgba8Params {
  const void* rgb = nullptr;  // 1 × 3 × height × width, FP16, valeurs dans [0, 1]
  int width = 0;
  int height = 0;
  std::uint32_t seed = 0;  // change à chaque image : le bruit de tramage ne reste pas figé
  void* out = nullptr;     // R, G, B, A sur 8 bits par pixel, lignes de pitch octets
  std::size_t pitch = 0;
};

// Tramage triangulaire de ±1 code 8 bits (somme de deux bruits uniformes), puis quantification.
cudaError_t launch_to_rgba8(const ToRgba8Params& params, cudaStream_t stream);

struct Rgb10a2ToSurfaceParams {
  const void* in = nullptr;  // R bits 0-9, G 10-19, B 20-29, A 30-31, lignes de pitch octets
  std::size_t pitch = 0;
  int width = 0;
  int height = 0;
  cudaSurfaceObject_t surface = 0;  // RGBA16 UNORM : code 10 bits × 65535 / 1023, arrondi
};

cudaError_t launch_rgb10a2_to_surface(const Rgb10a2ToSurfaceParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
