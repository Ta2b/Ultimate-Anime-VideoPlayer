// La composition par calques (kernels/spec/layers.h, ref::compose_layers) : le décor repris des sources
// déplacées par le mouvement de caméra, sur place dans l'image de RIFE (ou dans SPAN de b, pour la plaque).
// Lit les briques de la bibliothèque de warp (warp_lib.cuh).
#pragma once

#include <cuda_runtime_api.h>

#include "kernels/spec/layers.h"

namespace kirari::gpu::cuda {

// Tout passe par valeur au noyau (environ 3,2 Ko, sous la limite de 4 Ko des paramètres).
struct LayerGridArgs {
  int cols = 0;  // 0 : pas de grille (masque de la source seulement)
  int rows = 0;
  float tile_w = 1.0F;
  float tile_h = 1.0F;
  unsigned bits[kernels::kLayerGridWords] = {};  // NOLINT(modernize-avoid-c-arrays) : copié tel quel au noyau
};

struct LayerArgs {
  void* out = nullptr;      // 3 plans FP16 width × height
  const void* a = nullptr;  // source 0, même forme
  const void* b = nullptr;  // source 1 si count vaut 2
  int width = 0;
  int height = 0;
  int count = 1;
  float shift[2][6] = {};    // NOLINT(modernize-avoid-c-arrays)
  float weight[2] = {1.0F};  // NOLINT(modernize-avoid-c-arrays)
  float guard_low = 0.0F;
  float guard_high = 1.0F;
  int cubic = 0;
  LayerGridArgs grid;
  LayerGridArgs source_grid;
};

LayerArgs layer_args(const kernels::LayerParams& params);

cudaError_t launch_compose_layers(const LayerArgs& args, cudaStream_t stream);

}  // namespace kirari::gpu::cuda
