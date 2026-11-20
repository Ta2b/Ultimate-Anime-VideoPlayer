// Noyaux CUDA du softmax splatting de GMFSS Fortuna (kernels/spec/gmfss.h), entre les deux graphes TensorRT :
// remise à zéro des sommes FP32, addition bilinéaire par atomicAdd, puis normalisation vers le FP16 que lit
// le graphe « fusion ». Le portage de softsplat.py (CuPy). Les splats d'un instant passent en un seul lot :
// un memset, une addition et une normalisation pour tous, au lieu de trois lancements par splat qui
// s'attendent l'un l'autre ; la normalisation remplit aussi l'instant de la fusion.
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace kirari::gpu::cuda {

struct SoftSplatParams {
  const void* source = nullptr;  // 1 × channels × height × width, FP16
  const void* motion = nullptr;  // 1 × 3 × height × width, FP16 : fx, fy, z pour t = 1
  int width = 0;
  int height = 0;
  int channels = 0;
  float s = 0.5F;              // instant du côté, dans [0, 1]
  float* sums = nullptr;       // (channels + 1) plans FP32, le dernier pour les poids
  void* out = nullptr;         // 1 × channels × height × width, FP16 (un plan de plus avec coverage)
  int pairs = 0;               // rempli par launch_soft_splat_all : normalisation en half2
  int coverage = 0;            // 1 : le plan channels de out reçoit la couverture
  int weight_planes = 0;       // 0 : les poids suivent channels plans ; sinon après weight_planes
  bool reuse_weights = false;  // le premier lot du même travail les a déjà accumulés
  int out_pitch = 0;           // 0 : plans ; > 0 : HWC ; < 0 : groupes C8, out pointe le début du côté
  int out_first = 0;           // C8 : canal de la première somme de cette étape dans le côté
};

inline constexpr int kMaxSplatJobs = 8;
inline constexpr int kMaxIndexedSplatChannels = 192;

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

// Mémoire de l'index de contributions, par grille. Zéro pour les grilles hors du chemin indexé.
std::size_t indexed_splat_bytes(int width, int height);
// Chaque travail est terminé avant le suivant. Leurs sorties ne doivent recouvrir aucune de leurs entrées.
cudaError_t launch_indexed_splats(const SoftSplatBatch& batch, void* scratch, std::size_t bytes,
                                  cudaStream_t stream);

// Le mouvement d'une autre grille (kernels::MotionGrid) à la taille de la source, en FP16 : motion
// 3 × motion_height × motion_width lu en bilinéaire, out 3 × height × width, le flot mis à l'échelle.
struct MotionGridParams {
  const void* motion = nullptr;
  int motion_width = 0;
  int motion_height = 0;
  float scale = 1.0F;
  float x = 0.0F;
  float y = 0.0F;
  void* out = nullptr;
  int width = 0;
  int height = 0;
};

cudaError_t launch_motion_on_grid(const MotionGridParams& params, cudaStream_t stream);

// Le splat tiré (spec MotionGrid, splat tiré) : flow les trois plans FP16 width × height du flot poussé
// (fx, fy) et de sa couverture ; la source (channels plans) lue en Catmull-Rom à (X − s fx, Y − s fy) dans
// out, la couverture en plus si coverage.
struct PullParams {
  const void* source = nullptr;
  int channels = 0;
  const void* flow = nullptr;
  int width = 0;
  int height = 0;
  float s = 0.5F;
  void* out = nullptr;
  int coverage = 0;
};

cudaError_t launch_soft_pull(const PullParams& params, cudaStream_t stream);

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

// Le décor fixe repris des sources (spec gmfss_still), sur place dans key : trois images RGB FP16
// width × height.
struct StillParams {
  void* key = nullptr;
  const void* a = nullptr;
  const void* b = nullptr;
  int width = 0;
  int height = 0;
  float t = 0.5F;
};

cudaError_t launch_gmfss_still(const StillParams& params, cudaStream_t stream);

}  // namespace kirari::gpu::cuda
