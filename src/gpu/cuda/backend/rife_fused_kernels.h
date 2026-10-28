// Noyaux CUDA de RIFE fusionné (kernels/spec/rife_fused.h) : tout ce que le réseau fait en pleine résolution
// entre ses moteurs TensorRT. Un noyau par bloc remplace DepthToSpace, Resize ×d, mise à l'échelle du flot,
// accumulation, grille, GridSample, Cast, Split/Reshape/Concat et le Resize d'entrée du bloc suivant. Ce
// Resize (échelle 1/d, demi-pixel, linéaire) est une moyenne exacte de 2 × 2 pixels pleine résolution (d ≥ 2)
// ou l'identité (d = 1) : les noyaux n'évaluent le flot et le warp que sur les pixels que le bloc suivant
// lira (4/64 des pixels pour d = 8, 4/16 pour d = 4), puis moyennent. Rien de pleine résolution n'est écrit
// hors de l'entrée du dernier bloc. Les arrondis FP16 du graphe (sortie du Resize, Mul, Add, Cast, Sigmoid…)
// sont reproduits aux mêmes endroits : l'écart au graphe ne vient que de l'ordre des sommes.
#pragma once

#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

// Une source de flot : la sortie Y d'un bloc (52 canaux, yh × yw), dont le DepthToSpace Z (13 canaux,
// 2 yh × 2 yw) est agrandi ×scale vers la pleine résolution.
struct FusedFlowSource {
  const void* y = nullptr;  // FP16
  int height = 0;           // yh
  int width = 0;            // yw
  int scale = 1;            // facteur du Resize de sortie du bloc (8, 4, 2 ou 1)
};

// L'entrée X_0 (15 canaux) : moyenne 2 × 2 de l'image A, de l'image B, de leurs encodages et de l'instant.
struct FusedEntryParams {
  int height = 0;  // pleine résolution
  int width = 0;
  int reduce = 8;  // d_0
  int out_height = 0;
  int out_width = 0;
  const void* input = nullptr;   // 11 plans height × width, FP16
  const void* encode = nullptr;  // 8 plans
  void* x = nullptr;             // 15 × out_height × out_width
};

// L'entrée X_k d'un bloc k ≥ 1 (28 canaux) : flot accumulé des blocs 0 à k-1, warp des images et de leurs
// encodages, masque et attributs du bloc k-1.
struct FusedGlueParams {
  int height = 0;
  int width = 0;
  int reduce = 1;  // d_k
  int out_height = 0;
  int out_width = 0;
  int flows = 0;  // blocs 0 à flows-1 : 1 à 4
  FusedFlowSource
      source[4]{};  // NOLINT(modernize-avoid-c-arrays) : std::array n'est pas appelable depuis un noyau
  const void* input = nullptr;
  const void* encode = nullptr;
  void* x = nullptr;
};

// La fin : flot du dernier bloc ajouté, puis soit l'image (warp des deux images, masque sigmoïde, mélange),
// soit le flot final et le masque (la variante mouvement de RIFE, dont le warp est refait par la chaîne).
struct FusedFinishParams {
  int height = 0;
  int width = 0;
  const void* input = nullptr;
  const void* x_last = nullptr;  // entrée du dernier bloc : canaux 24-27 = flot accumulé
  const void* y_last = nullptr;  // sortie du dernier bloc : 52 canaux à (height / 2) × (width / 2)
  int y_height = 0;
  int y_width = 0;
  void* image = nullptr;  // 3 plans, ou nul
  void* flow = nullptr;   // 4 plans (variante mouvement)
  void* mask = nullptr;   // 1 plan (variante mouvement)
};

cudaError_t launch_fused_entry(const FusedEntryParams& params, cudaStream_t stream);
cudaError_t launch_fused_glue(const FusedGlueParams& params, cudaStream_t stream);
cudaError_t launch_fused_finish(const FusedFinishParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
