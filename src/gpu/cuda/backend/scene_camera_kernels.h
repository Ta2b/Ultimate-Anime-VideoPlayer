// Noyaux CUDA de l'affinage de la caméra de la carte (kernels/spec/scene_camera.h) : les lumas du flux
// floutées, les pas de Gauss-Newton (histogramme des écarts, sommes pondérées, résolution par un fil) et la
// vérification par tuile, tout sur la file de NVOF sans retour à l'hôte entre les pas. Référence :
// kernels/ref/scene_camera.cpp ; les sommes sont en float par fil puis en double, le résultat suit la
// référence à l'arrondi près.
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "kernels/spec/scene_camera.h"

namespace kirari::gpu::cuda {

// L'état d'un affinage sur le GPU : le modèle, puis les sommes et l'histogramme du pas en cours (remis à zéro
// par la résolution). Mis à zéro, modèle de départ copié, avant le premier pas ; relu tel quel par l'hôte.
struct SceneRefineState {
  double model[6];
  double sums[kernels::scene::kRefineSums];
  unsigned histogram[kernels::scene::kRefineBins];
  unsigned count;
  int steps;    // pas faits
  int stopped;  // H mal conditionné : plus de pas
};

// Une luma du flux floutée par la boîte (2 radius + 1)² : somme des lignes dans rows, puis des colonnes.
struct SceneBoxParams {
  const unsigned char* luma = nullptr;
  std::size_t pitch = 0;  // octets
  float* rows = nullptr;  // width × height
  float* out = nullptr;   // width × height
  int width = 0;
  int height = 0;
  int radius = 1;
};
cudaError_t launch_scene_box(const SceneBoxParams& params, cudaStream_t stream);

struct SceneRefineParams {
  const float* a = nullptr;  // lumas floutées, width × height
  const float* b = nullptr;
  int width = 0;
  int height = 0;
  int cols = 0;                          // tuiles
  const unsigned char* decor = nullptr;  // cols × rows
  SceneRefineState* state = nullptr;
};
// Un pas : l'histogramme des |e|, les sommes pondérées, puis la résolution.
cudaError_t launch_scene_refine_step(const SceneRefineParams& params, cudaStream_t stream);

struct SceneCheckParams {
  const float* a = nullptr;  // lumas du dernier flou
  const float* b = nullptr;
  int width = 0;
  int height = 0;
  int cols = 0;
  int rows = 0;
  const unsigned char* decor = nullptr;
  const SceneRefineState* state = nullptr;  // le modèle affiné
  kernels::SceneTileCheck* out = nullptr;   // cols × rows
};
cudaError_t launch_scene_check(const SceneCheckParams& params, cudaStream_t stream);

}  // namespace kirari::gpu::cuda
