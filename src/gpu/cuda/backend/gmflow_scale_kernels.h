// Entrées des deux échelles du transformeur de GMFlow, depuis les caractéristiques du squelette.
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace kirari::gpu::cuda {

// Table FP16 des positions d'une fenêtre : hauteur / splits lignes de 64 canaux de y,
// suivies de largeur / splits lignes de 64 canaux de x. Taille nulle si la grille ne se découpe pas.
std::size_t gmflow_position_bytes(int height, int width, int splits);
cudaError_t launch_gmflow_position(cudaStream_t stream, int height, int width, int splits, void* table);

// Caractéristiques et jetons [2][height × width][128] FP16. La table correspond à cette grille.
// Avec decimate, les caractéristiques au quart sont deux fois plus larges et hautes que les jetons.
cudaError_t launch_gmflow_eighth_input(cudaStream_t stream, const void* features, const void* positions,
                                       int height, int width, int splits, void* tokens,
                                       bool decimate = false, bool swapped = false);

// height × width : grille au quart (double de celle du flot grossier). Les deux flots xy FP32
// agrandis ×2 avec align_corners sont écrits dans quarter_flow, dans l'ordre 0→1, 1→0.
// tokens porte 4 images [source 0, source 1, cible 1 déformée, cible 0 déformée] FP16,
// positions comprises. Les régions sont distinctes et les tenseurs FP16 alignés sur 16 octets.
// Si swapped est vrai, les deux images de features sont stockées dans l'ordre inverse.
cudaError_t launch_gmflow_quarter_input(cudaStream_t stream, const void* features, const void* coarse_flow,
                                        const void* positions, int height, int width, int splits,
                                        void* tokens, void* quarter_flow, bool swapped = false);

}  // namespace kirari::gpu::cuda
