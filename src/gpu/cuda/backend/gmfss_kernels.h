// Noyaux CUDA du softmax splatting de GMFSS Fortuna (kernels/spec/gmfss.h), entre les deux
// graphes TensorRT : remise à zéro des sommes FP32, addition bilinéaire par atomicAdd, puis
// normalisation vers le FP16 que lit le graphe « fusion ». Le portage de softsplat.py (CuPy).
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

struct SoftSplatParams {
  const void* source = nullptr;  // 1 × channels × height × width, FP16
  const void* motion = nullptr;  // 1 × 3 × height × width, FP16 : fx, fy, z pour t = 1
  int width = 0;
  int height = 0;
  int channels = 0;
  float s = 0.5F;         // instant du côté, dans [0, 1]
  float* sums = nullptr;  // (channels + 1) plans FP32, le dernier pour les poids
  void* out = nullptr;    // 1 × channels × height × width, FP16
};

// Octets des sommes d'un splat de params.
std::size_t soft_splat_sums_bytes(int width, int height, int channels);

// Les trois étapes, enfilées sur stream.
cudaError_t launch_soft_splat(const SoftSplatParams& params, cudaStream_t stream);

}  // namespace ajp::gpu::cuda
