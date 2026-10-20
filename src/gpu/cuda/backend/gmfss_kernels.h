// Noyaux CUDA du softmax splatting de GMFSS Fortuna (kernels/spec/gmfss.h), entre les deux
// graphes TensorRT : remise à zéro des sommes FP32, addition bilinéaire par atomicAdd, puis
// normalisation vers le FP16 que lit le graphe « fusion ». Le portage de softsplat.py (CuPy).
// Les six splats d'un instant passent en un seul lot : un memset, une addition et une
// normalisation pour tous, au lieu de trois lancements par splat qui s'attendent l'un l'autre.
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

struct SoftSplatParams {
  const void* source = nullptr;  // 1 × channels × height × width, FP16
  const void* motion = nullptr;  // 1 × 3 × height × width, FP16 : fx, fy, z pour t = 1
  int width = 0;
  int height = 0;
  int channels = 0;
  float s = 0.5F;         // instant du côté, dans [0, 1]
  float* sums = nullptr;  // (channels + 1) plans FP32, le dernier pour les poids
  void* out = nullptr;    // 1 × channels × height × width, FP16
};

inline constexpr int kMaxSplatJobs = 8;

// Un lot de splats : chacun ses tenseurs et ses sommes, toutes dans un même tampon que le lot
// remet à zéro d'un seul memset. Les tables de blocs (préfixes, l'entrée count donne le total)
// sont remplies par launch_soft_splat_all.
struct SoftSplatBatch {
  SoftSplatParams jobs[kMaxSplatJobs];
  int count = 0;
  unsigned splat_blocks[kMaxSplatJobs + 1];      // tuiles × groupes de plans, par travail
  unsigned normalize_blocks[kMaxSplatJobs + 1];  // tuiles, par travail
};

// Octets des sommes d'un splat de params.
std::size_t soft_splat_sums_bytes(int width, int height, int channels);

// Les trois étapes pour tous les travaux de batch (1 à kMaxSplatJobs, leurs sommes dans
// [sums, sums + sums_bytes[), enfilées sur stream. cudaErrorInvalidValue hors de ce nombre.
cudaError_t launch_soft_splat_all(SoftSplatBatch& batch, void* sums, std::size_t sums_bytes,
                                  cudaStream_t stream);

// count éléments FP16 à value (l'instant t de GMFSS Union), enfilé sur stream.
cudaError_t launch_fill(void* data, std::size_t count, float value, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
