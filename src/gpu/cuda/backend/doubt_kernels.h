// Noyau CUDA du fondu local du mouvement de RIFE (kernels/spec/motion_doubt.h, référence
// kernels/ref/motion_doubt.cpp) : flot 1 × 4 et masque 1 × 1 FP16 lus, tirés vers le fondu selon une grille
// de doute passée par valeur dans les paramètres du noyau (au plus 48 × 48 octets), écrits ailleurs. Et le
// mouvement d'une caméra affine (kernels/spec/motion_camera.h, référence kernels/ref/motion_camera.cpp),
// écrit aux mêmes formats.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#include "kernels/spec/motion_doubt.h"

namespace kirari::gpu::cuda {

struct MotionDoubtParams {
  const void* flow = nullptr;  // 1 × 4 × plane_height × plane_width, FP16
  const void* mask = nullptr;  // 1 × 1, même taille
  void* out_flow = nullptr;    // mêmes formes
  void* out_mask = nullptr;
  int plane_width = 0;
  int plane_height = 0;
  int window_x = 0;  // l'image dans les plans
  int window_y = 0;
  int window_width = 0;
  int window_height = 0;
  int cols = 0;  // la grille, posée sur la fenêtre
  int rows = 0;
  float fade = 0.5F;
  // cols × rows poids, ligne par ligne ; un tableau C : nvcc ne lit pas std::array dans un noyau.
  std::uint8_t weights[kernels::kDoubtMaxCells * kernels::kDoubtMaxCells]{};  // NOLINT(*-avoid-c-arrays)
};

cudaError_t launch_motion_doubt(const MotionDoubtParams& params, cudaStream_t stream);

struct MotionCameraParams {
  void* out_flow = nullptr;  // 1 × 4 × plane_height × plane_width, FP16
  void* out_mask = nullptr;  // 1 × 1, même taille
  int plane_width = 0;
  int plane_height = 0;
  int window_x = 0;
  int window_y = 0;
  int window_width = 0;
  int window_height = 0;
  float scale = 1.0F;  // pixels de la fenêtre par pixel de travail
  float camera[6]{};   // NOLINT(*-avoid-c-arrays) : c0 à c5, pixels de travail
};

cudaError_t launch_motion_camera(const MotionCameraParams& params, cudaStream_t stream);

}  // namespace kirari::gpu::cuda
