// Noyaux CUDA autour du flux optique matériel (kernels/spec/optical_flow.h, référence
// kernels/ref/optical_flow.cpp) : la luma 8 bits qu'il lit, tirée d'une image RGB FP16, et le
// mouvement au milieu de la paire tiré de ses deux flots S10.5, dans les tenseurs du warp.
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

// Luma BT.709 en 8 bits plein (round(255 Y), Y = 0,2126 R + 0,7152 G + 0,0722 B bornée à [0, 1])
// d'une image RGB FP16 1 × 3 × height × width, écrite dans deux tampons du flux matériel (un par
// session, avant et arrière), chacun avec son pas de ligne.
struct Luma8Params {
  const void* rgb = nullptr;
  int width = 0;
  int height = 0;
  void* first = nullptr;
  std::size_t first_pitch = 0;
  void* second = nullptr;
  std::size_t second_pitch = 0;
};

cudaError_t launch_luma8(const Luma8Params& params, cudaStream_t stream);

// Mouvement au milieu depuis les flots S10.5 avant (a → b) et arrière (b → a), grilles de
// cells_x × cells_y vecteurs (int16 x, int16 y entrelacés, pas de ligne en octets) : flot FP16
// 1 × 4 × cells_y × cells_x (f0x, f0y, f1x, f1y) et masque FP16 1 × 1, en pixels de la grille.
struct MidMotionParams {
  const void* forward = nullptr;
  std::size_t forward_pitch = 0;
  const void* backward = nullptr;
  std::size_t backward_pitch = 0;
  int cells_x = 0;
  int cells_y = 0;
  int grid = 0;  // 1, 2 ou 4 pixels de la luma par vecteur
  void* flow = nullptr;
  void* mask = nullptr;
};

cudaError_t launch_mid_motion(const MidMotionParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
