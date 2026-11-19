// Noyaux CUDA de la carte de la scène (kernels/spec/scene_flow.h) : la luma 8 bits du flux tirée du RGB de
// travail, écrite dans un tampon de NVOF (son pas, pas celui des surfaces du décodeur), puis les sommes par
// tuile du flux, de son coût et des deux lumas. Référence : kernels/ref/scene_flow.cpp (scene_luma,
// scene_tile_sums). Les sommes sont entières : le résultat est celui de la référence au bit près.
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "kernels/spec/scene_flow.h"

namespace kirari::gpu::cuda {

struct SceneLumaParams {
  const void* rgb = nullptr;  // 1 × 3 × height × width
  bool fp16 = true;           // sinon FP32
  int width = 0;              // image de travail
  int height = 0;
  int divisor = 1;  // boîte divisor × divisor par pixel du flux
  float kr = 0.0F;
  float kb = 0.0F;
  unsigned char* luma = nullptr;  // (width / divisor) × (height / divisor) codes
  std::size_t pitch = 0;          // octets
};

cudaError_t launch_scene_luma(const SceneLumaParams& params, cudaStream_t stream);

struct SceneTilesParams {
  const unsigned char* a = nullptr;  // luma du flux, image de départ
  std::size_t a_pitch = 0;
  const unsigned char* b = nullptr;  // et d'arrivée
  std::size_t b_pitch = 0;
  int width = 0;  // image du flux
  int height = 0;
  const short2* flow = nullptr;  // S10.5, scene_flow_cells(width) × scene_flow_cells(height)
  std::size_t flow_pitch = 0;
  const unsigned char* cost = nullptr;  // un octet par vecteur
  std::size_t cost_pitch = 0;
  int cols = 0;  // tuiles
  int rows = 0;
  kernels::SceneTileSums* out = nullptr;  // cols × rows
};

cudaError_t launch_scene_tiles(const SceneTilesParams& params, cudaStream_t stream);

}  // namespace kirari::gpu::cuda
