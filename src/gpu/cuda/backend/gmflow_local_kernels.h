// Appariement local r=4 et propagation 3×3, au quart de GMFlow.
#pragma once

#include <cuda_runtime_api.h>

namespace kirari::gpu::cuda {

// Jetons [4][height × width][128] FP16 : les deux sources, puis les deux cibles déformées.
// Flot initial et sortie [2][height × width][2] FP32 xy, en pixels du quart. out peut être initial.
cudaError_t launch_gmflow_local_match(cudaStream_t stream, const void* tokens, const void* initial,
                                      int height, int width, void* out);

// Projections query et keys [2][height × width][128] FP16. Les voisins hors image valent zéro
// mais participent au softmax. Les flots ont la même disposition que ci-dessus, régions distinctes.
cudaError_t launch_gmflow_local_propagate(cudaStream_t stream, const void* query, const void* keys,
                                          const void* flow, int height, int width, void* out);

}  // namespace kirari::gpu::cuda
