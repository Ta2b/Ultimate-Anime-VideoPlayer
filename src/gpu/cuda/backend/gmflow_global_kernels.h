// Appariement et propagation globaux au huitième pour GMFlow natif (docs/gmflow-natif.md).
#pragma once

#include <cuda_runtime_api.h>

namespace kirari::gpu::cuda {

// Jetons FP16 [2][height × width][128] alignés sur 16 octets, en sortie du transformeur.
// Rend les deux sens : flux FP32 [2][height × width][2], x puis y, en pixels de cette grille.
cudaError_t launch_gmflow_global_match(cudaStream_t stream, const void* tokens, int height, int width,
                                       void* flow);

// query et keys sont les projections FP16 des jetons du même sens, dans la même disposition.
// flow et out sont des flux FP32 dans la disposition de l'appariement, dans deux régions distinctes.
cudaError_t launch_gmflow_global_propagate(cudaStream_t stream, const void* query, const void* keys,
                                           const void* flow, int height, int width, void* out);

}  // namespace kirari::gpu::cuda
