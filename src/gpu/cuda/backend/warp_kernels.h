// Noyaux CUDA du warp arrière par flot et du mélange de deux images (kernels/spec/warp.h,
// référence kernels/ref/warp.cpp) : le mouvement sorti par la variante « mouvement » de RIFE
// (flot 1 × 4, masque 1 × 1, plans remplis de rife_geometry, fenêtre de l'image dedans) déforme
// deux images RGB FP16 de la taille de sortie, égale ou plus grande que la fenêtre (mouvement
// estimé à 720p, images à 1080p).
#pragma once

#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

struct WarpBlendParams {
  const void* a = nullptr;  // 1 × 3 × height × width, FP16
  const void* b = nullptr;
  int width = 0;
  int height = 0;
  const void* flow = nullptr;  // 1 × 4 × plane_height × plane_width, FP16 (f0x, f0y, f1x, f1y)
  const void* mask = nullptr;  // 1 × 1 × plane_height × plane_width, FP16
  int plane_width = 0;
  int plane_height = 0;
  int window_x = 0;  // l'image dans les plans du mouvement
  int window_y = 0;
  int window_width = 0;
  int window_height = 0;
  float time = 0.5F;         // instant de la sortie
  bool from_middle = false;  // mouvement estimé au milieu, sortie à time (loi de mélange ajustée)
  // Garder les poses (kernels/spec/poses.h, avec from_middle) : l'instant de chaque pixel réglé
  // d'après le mouvement de la paire précédente, mêmes plans et même fenêtre que flow ;
  // previous_flow nul : la paire précédente était tenue.
  bool keep_poses = false;
  const void* previous_flow = nullptr;
  void* out = nullptr;  // 1 × 3 × height × width, FP16
  // Objets texture des mêmes tenseurs (vus en largeur × plans · hauteur), ou 0. Tous ceux des
  // tenseurs lus présents, largeur paire et sortie alignée sur 4 octets : le noyau lit par le
  // filtre bilinéaire matériel et écrit deux pixels par fil (half2) ; sinon, la lecture exacte.
  struct Textures {
    cudaTextureObject_t a = 0;
    cudaTextureObject_t b = 0;
    cudaTextureObject_t flow = 0;
    cudaTextureObject_t mask = 0;
    cudaTextureObject_t previous_flow = 0;
  } textures;
};

cudaError_t launch_warp_blend(const WarpBlendParams& params, cudaStream_t stream);

// Plus grand |déplacement| des 4 plans de flot dans la fenêtre, en pixels de la fenêtre, écrit en
// bits de float dans *result (remis à zéro par l'appelant ; les valeurs positives se comparent
// comme des entiers).
struct FlowMaxParams {
  const void* flow = nullptr;
  int plane_width = 0;
  int plane_height = 0;
  int window_x = 0;
  int window_y = 0;
  int window_width = 0;
  int window_height = 0;
  unsigned int* result = nullptr;
};

cudaError_t launch_flow_max(const FlowMaxParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
