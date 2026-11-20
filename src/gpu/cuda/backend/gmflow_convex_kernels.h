// Fin de GMFlow au quart : entrée de la tête de masque et agrandissement convexe ×4.
#pragma once

#include <cuda_runtime_api.h>

namespace kirari::gpu::cuda {

// features : [2][height*width][128] FP16. flow : [2][height*width][2] FP32, x puis y.
// packed : [2][height*width][144] FP16, les deux plans de flot puis les 128 traits et le rembourrage à zéro.
cudaError_t launch_gmflow_convex_input(cudaStream_t stream, const void* features, const float* flow,
                                       int height, int width, void* packed);

// mask : [2][height*width][9][4][4] FP16, 3×3 par lignes puis sous-pixel (y,x). flow : comme ci-dessus.
// output : [2][2][4*height][4*width] FP32 ; les sens 0→1 puis 1→0, et les plans x puis y.
// Le softmax est en FP32. Hors image, le flot d'un voisin vaut zéro mais son poids compte toujours.
cudaError_t launch_gmflow_convex(cudaStream_t stream, const void* mask, const float* flow, int height,
                                 int width, float* output);

}  // namespace kirari::gpu::cuda
