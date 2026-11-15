// Noyaux CUDA du softmax splatting de GMFSS Fortuna (kernels/spec/gmfss.h), entre les deux graphes TensorRT :
// remise à zéro des sommes FP32, addition bilinéaire par atomicAdd, puis normalisation vers le FP16 que lit
// le graphe « fusion ». Le portage de softsplat.py (CuPy). Les splats d'un instant passent en un seul lot :
// un memset, une addition et une normalisation pour tous, au lieu de trois lancements par splat qui
// s'attendent l'un l'autre ; la normalisation remplit aussi l'instant de la fusion.
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
  void* out = nullptr;    // 1 × channels × height × width, FP16 (un plan de plus avec coverage)
  int pairs = 0;          // rempli par launch_soft_splat_all : normalisation en half2
  // Le mouvement sur sa grille (kernels::MotionGrid) : motion_width × motion_height, lu en bilinéaire ;
  // grid_scale 0 : la grille de la source, le mouvement du pixel lui-même.
  int motion_width = 0;
  int motion_height = 0;
  float grid_scale = 0.0F;
  float grid_x = 0.0F;
  float grid_y = 0.0F;
  int coverage = 0;  // 1 : le plan channels de out reçoit la couverture
};

inline constexpr int kMaxSplatJobs = 8;

// Un lot de splats : chacun ses tenseurs et ses sommes, toutes dans un même tampon que le lot remet à zéro
// d'un seul memset. fill, s'il n'est pas nul : fill_count éléments FP16 mis à fill_value par le lancement de
// la normalisation (l'instant t de la fusion).
struct SoftSplatBatch {
  SoftSplatParams jobs[kMaxSplatJobs];
  int count = 0;
  void* fill = nullptr;
  std::size_t fill_count = 0;
  float fill_value = 0.0F;
};

// Octets des sommes d'un splat.
std::size_t soft_splat_sums_bytes(int width, int height, int channels);

// Les trois étapes pour tous les travaux de batch (1 à kMaxSplatJobs, leurs sommes dans
// [sums, sums + sums_bytes[), enfilées sur stream. cudaErrorInvalidValue hors de ce nombre.
cudaError_t launch_soft_splat_all(SoftSplatBatch& batch, void* sums, std::size_t sums_bytes,
                                  cudaStream_t stream);

// La composition à la taille de travail (spec gmfss_compose) : sources de travail splattées high_a, high_b
// (4 plans out_width × out_height, la couverture en dernier), celles de RIFE low_a, low_b (4 plans
// width × height), la fusion (3 plans fusion_width × fusion_height, l'image de RIFE en (left, top)).
struct ComposeParams {
  const void* high_a = nullptr;
  const void* high_b = nullptr;
  const void* low_a = nullptr;
  const void* low_b = nullptr;
  const void* fusion = nullptr;
  int width = 0;  // RIFE
  int height = 0;
  int out_width = 0;
  int out_height = 0;
  int fusion_width = 0;
  int fusion_height = 0;
  int left = 0;
  int top = 0;
  float t = 0.5F;
  void* out = nullptr;  // 3 plans FP16 out_width × out_height
};

cudaError_t launch_gmfss_compose(const ComposeParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
