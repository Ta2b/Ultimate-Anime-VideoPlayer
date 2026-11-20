// Noyaux CUDA du warp arrière par flot et du mélange de deux images (kernels/spec/warp.h,
// référence kernels/ref/warp.cpp) : le mouvement sorti par la variante « mouvement » de RIFE
// (flot 1 × 4, masque 1 × 1, plans remplis de rife_geometry, fenêtre de l'image dedans) déforme
// deux images RGB FP16 de la taille de sortie, égale ou plus grande que la fenêtre (mouvement
// estimé à 720p, images à 1080p).
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "gpu/cuda/backend/ring_surface.h"

namespace kirari::gpu::cuda {

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
  bool cubic = false;        // images lues en Catmull-Rom (kernels::RifeWarp), sinon en bilinéaire
  bool from_middle = false;  // mouvement estimé au milieu, sortie à time (loi de mélange ajustée)
  // L'instant du flot lu, que suit le mouvement d'ensemble (kernels/spec/warp_pan.h) : 0,5 (le milieu), ou
  // time quand le flot est celui de time hors from_middle (DistilDRBA, RIFE direct). Calculé par l'hôte : le
  // noyau filtré en Catmull-Rom tient alors ses registres sans déborder (sm_120, --warn-on-spills).
  float pan_time = 0.5F;
  // Garder les poses (kernels/spec/poses.h, avec from_middle) : l'instant de chaque pixel réglé
  // d'après le mouvement de la paire précédente, mêmes plans et même fenêtre que flow ;
  // previous_flow nul : la paire précédente était tenue.
  bool keep_poses = false;
  // Mouvement d'ensemble (kernels/spec/warp_pan.h) : l'estimation de la paire (dx, dy, sûre), écrite par
  // launch_pan_global sur la même file ; nul : flot de RIFE tel quel. Ignoré quand keep_poses.
  const float* pan = nullptr;
  const void* previous_flow = nullptr;
  void* out = nullptr;  // 1 × 3 × height × width, FP16
  // Non nul : la sortie va dans ce slot RGBA16 UNORM du ring (width × height), au code que
  // write_rgba16 écrirait de out, et out n'est pas écrit (étude 1, P4).
  RingSurface surface{};  // emplacement de l'anneau (ring_surface.h) ; object 0 : aucun
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

struct LayerArgs;

// Essai isolé warp CUDA exact + calques de la paire (pas la plaque). Un indicateur par tampon, lu sur la
// même file et valable pour cette écriture du tampon seulement. Nul, le raccourci ne s'applique pas.
struct LayerWarpBounds {
  const unsigned* a_outside = nullptr;
  const unsigned* b_outside = nullptr;
  const unsigned* mask_outside = nullptr;
};
// Cherche une valeur FP16 hors de [-2,2] dans une source, ou hors de [0,1] dans un masque. NaN et Inf dehors.
// Réinitialise *outside sur stream. elements compte les valeurs FP16, tous les plans du tampon inclus.
cudaError_t launch_certify_layer_warp(const void* data, std::size_t elements, bool mask, unsigned* outside,
                                      cudaStream_t stream);
// shortcut_count, s'il est donné, est remis à zéro puis reçoit le nombre de pixels qui évitent le warp.
cudaError_t launch_warp_blend_layers(const WarpBlendParams& warp, const LayerArgs& layer,
                                     LayerWarpBounds bounds, unsigned* shortcut_count, cudaStream_t stream,
                                     bool compact_poses = false);

// Mouvement d'ensemble de la paire (a et b de params, kernels/spec/warp_pan.h) dans out[0..2] : dx, dy et
// 1 si l'estimation est sûre, 0 sinon. Un seul bloc ; pan de params n'est pas lu.
cudaError_t launch_pan_global(const WarpBlendParams& params, float* out, cudaStream_t stream);

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

}  // namespace kirari::gpu::cuda
