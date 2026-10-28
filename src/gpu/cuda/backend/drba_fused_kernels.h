// Noyaux CUDA de DistilDRBA fusionné (kernels/spec/drba_fused.h) : tout ce que le réseau fait hors
// convolution, entre ses quatre moteurs TensorRT.
//   pre   : réductions d'images 2 × 2 (Resize d'échelle 1/s, demi-pixel, linéaire = moyenne exacte de deux
//           lignes et deux colonnes) vers X_2, X_1, X_0, et plans du temps de X_0, X_1, X_2 ;
//   post  : réductions 2 × 2 des trois encodages (sortie de l'encodeur, dans X_2) vers X_1 et X_0 ;
//   glue  : X_1 puis X_2 : sortie du bloc précédent agrandie ×2 (DepthToSpace, Resize), flot, grilles,
//           GridSample des images et des encodages, masque et attributs ;
//   finish: sortie du bloc 2 agrandie ×t, flot, sigmoïde du masque, warps pleine résolution, mélange.
// Le côté choisi (image 0 ou 2 selon tau) est lu sur le périphérique (plan 6 de pair01) : il change à chaque
// image. Les arrondis FP16 du graphe (sorties des Resize, Mul, Add, Cast, Sigmoid) sont reproduits aux mêmes
// endroits ; les Resize et GridSample du graphe sont en FP32.
#pragma once

#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

// Pas de grille d'un axe de n pixels : 2/(n-1) en FP32 (calcul de la base) et arrondi FP16 (échelle du flot).
struct DrbaGridStep {
  float step32 = 0.0F;
  float step16 = 0.0F;
};

// Les entrées X_0, X_1, X_2 (leurs 0-8 images réduites et plans du temps) à partir des paires.
struct DrbaPreParams {
  int height = 0;  // pleine résolution
  int width = 0;
  int reduce0 = 0;  // s_0, s_1, s_2 : les réductions d'entrée des trois blocs
  int reduce1 = 0;
  int reduce2 = 0;
  const void* pair01 = nullptr;  // 7 plans FP16
  const void* pair12 = nullptr;
  void* x0 = nullptr;
  void* x1 = nullptr;
  void* x2 = nullptr;
};

// Les encodages réduits de X_1 (canaux 9-56, moyenne des pixels 2o, 2o+1) et de X_0 (4o+1, 4o+2) depuis la
// sortie de l'encodeur rangée dans X_2.
struct DrbaPostParams {
  int enc_height = 0;  // résolution de l'encodage (H / t)
  int enc_width = 0;
  const void* x2 = nullptr;
  void* x1 = nullptr;
  void* x0 = nullptr;
};

// L'entrée X_k (k = 1, 2) : canaux 57-94 (warps), 96-104 (masque, attributs), 105-108 (flot).
struct DrbaGlueParams {
  int height = 0;  // résolution de X_k
  int width = 0;
  const void* y = nullptr;  // sortie du bloc k-1 : 52 canaux à (height / 4) × (width / 4)
  const void* pair01 = nullptr;
  int full_height = 0;  // pleine résolution (le plan de tau de pair01)
  int full_width = 0;
  void* x = nullptr;
  DrbaGridStep grid_x;
  DrbaGridStep grid_y;
};

struct DrbaFinishParams {
  int height = 0;
  int width = 0;
  int upscale = 4;          // t : agrandissement de la sortie du bloc 2
  const void* y = nullptr;  // sortie du bloc 2 : 52 canaux à (height / 2t) × (width / 2t)
  const void* pair01 = nullptr;
  const void* pair12 = nullptr;
  void* out = nullptr;  // 3 plans
  DrbaGridStep grid_x;
  DrbaGridStep grid_y;
};

DrbaGridStep drba_grid_step(int n);

cudaError_t launch_drba_pre(const DrbaPreParams& params, cudaStream_t stream);
cudaError_t launch_drba_post(const DrbaPostParams& params, cudaStream_t stream);
cudaError_t launch_drba_glue(const DrbaGlueParams& params, cudaStream_t stream);
cudaError_t launch_drba_finish(const DrbaFinishParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
