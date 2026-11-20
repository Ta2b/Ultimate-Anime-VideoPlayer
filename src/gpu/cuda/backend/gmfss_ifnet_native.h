#pragma once

#include "gpu/cuda/backend/rife_native_net.h"

namespace kirari::gpu::cuda {

// IFNet de la fusion GMFSS. Entrées A/B et sortie en trois plans FP16, instant scalaire FP16.
class GmfssIfnetNative {
 public:
  virtual ~GmfssIfnetNative() = default;
  static std::unique_ptr<GmfssIfnetNative> create(const std::array<RifeNativePiece, 4>& pieces, int height,
                                                  int width, cudaStream_t stream, std::string& error);
  virtual cudaError_t run(cudaStream_t stream, const void* a, const void* b, const void* time,
                          void* out) const = 0;
};

}  // namespace kirari::gpu::cuda
