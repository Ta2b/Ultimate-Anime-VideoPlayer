// Noyaux CUDA du softmax splatting de GMFSS Fortuna (kernels/spec/gmfss.h), entre les deux
// graphes TensorRT : remise à zéro des sommes FP32, addition bilinéaire par atomicAdd, puis
// normalisation vers le FP16 que lit le graphe « fusion ». Le portage de softsplat.py (CuPy).
// Les splats d'un instant passent en un seul lot : un memset, une addition et une normalisation
// pour tous, au lieu de trois lancements par splat qui s'attendent l'un l'autre. La composition
// après SPAN (spec gmfss_compose) termine l'image intermédiaire déformée après SPAN.
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

struct SoftSplatParams {
  const void* source = nullptr;  // 1 × channels × height × width, FP16
  const void* motion = nullptr;  // 1 × 3 × motion_height × motion_width, FP16 : fx, fy, z pour t = 1
  int width = 0;
  int height = 0;
  int channels = 0;
  float s = 0.5F;  // instant du côté, dans [0, 1]
  // Le mouvement sur sa grille (spec MotionGrid) : direct, même grille que la source et lu tel
  // quel ; sinon bilinéaire en ((x + ½) / scale − ½ + mx, …), flot × scale.
  int motion_width = 0;
  int motion_height = 0;
  float motion_scale = 1.0F;
  float motion_x = 0.0F;
  float motion_y = 0.0F;
  int direct = 1;
  int coverage = 0;       // out a un plan de plus : la couverture (spec)
  float* sums = nullptr;  // (channels + 1) plans FP32, le dernier pour les poids
  void* out = nullptr;    // 1 × channels (+ 1) × height × width, FP16
  int pairs = 0;          // rempli par launch_soft_splat_all : normalisation en half2
};

inline constexpr int kMaxSplatJobs = 12;

// Un lot de splats : chacun ses tenseurs et ses sommes, toutes dans un même tampon que le lot
// remet à zéro d'un seul memset. fill, s'il n'est pas nul : fill_count éléments FP16 mis à
// fill_value par le lancement de la normalisation (l'instant t de GMFSS Union).
struct SoftSplatBatch {
  SoftSplatParams jobs[kMaxSplatJobs];
  int count = 0;
  void* fill = nullptr;
  std::size_t fill_count = 0;
  float fill_value = 0.0F;
};

// Octets des sommes d'un splat de params.
std::size_t soft_splat_sums_bytes(int width, int height, int channels);

// Les trois étapes pour tous les travaux de batch (1 à kMaxSplatJobs, leurs sommes dans
// [sums, sums + sums_bytes[), enfilées sur stream. cudaErrorInvalidValue hors de ce nombre.
cudaError_t launch_soft_splat_all(SoftSplatBatch& batch, void* sums, std::size_t sums_bytes,
                                  cudaStream_t stream);

// count éléments FP16 à value (l'instant t de GMFSS Union), enfilé sur stream.
cudaError_t launch_fill(void* data, std::size_t count, float value, cudaStream_t stream);

// Composition après SPAN (spec gmfss_compose) : out = H + F − L borné à [0, 1].
struct GmfssComposeParams {
  const void* high_a = nullptr;  // 1 × 4 × height × width, FP16 : RGB splatté et couverture
  const void* high_b = nullptr;
  int width = 0;  // scale × low_width
  int height = 0;
  const void* low_a = nullptr;  // 1 × 4 × low_height × low_width
  const void* low_b = nullptr;
  int low_width = 0;
  int low_height = 0;
  int scale = 1;
  const void* fusion = nullptr;  // 1 × 3 × fusion_height × fusion_width, la fenêtre dedans
  int fusion_width = 0;
  int fusion_height = 0;
  int window_x = 0;
  int window_y = 0;
  float t = 0.5F;
  void* out = nullptr;  // 1 × 3 × height × width
  // Non nul : la sortie va dans ce slot RGBA16 UNORM du ring, comme WarpBlendParams::surface.
  cudaSurfaceObject_t surface = 0;
  // Objets texture des sources de travail et de la fusion (vues en largeur × plans · hauteur), ou
  // 0. Les trois présents, largeur paire et tenseurs pleine taille alignés sur 4 octets : lecture
  // par le filtre bilinéaire matériel, deux pixels par fil (half2) ; sinon, la lecture exacte.
  struct Textures {
    cudaTextureObject_t low_a = 0;
    cudaTextureObject_t low_b = 0;
    cudaTextureObject_t fusion = 0;
  } textures;
};

cudaError_t launch_gmfss_compose(const GmfssComposeParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
