// Dernier bloc du DistilDRBA v2 lite, du X2 HWC8 au flot et au masque HWC8.
#pragma once

#include <memory>
#include <span>

#include "gpu/cuda/backend/rife_native_net.h"

namespace kirari::gpu::cuda {

class DrbaTailNative {
 public:
  virtual ~DrbaTailNative() = default;

  static std::unique_ptr<DrbaTailNative> create(std::span<const RifeNativeConv> layers, cudaStream_t stream,
                                                cudaError_t& error);

  static constexpr int kOutputPitch = 24;

  // reuse désigne les mêmes images, comme D6.
  virtual cudaError_t enqueue(cudaStream_t stream, const void* input, void* output, bool reuse) const = 0;
};

}  // namespace kirari::gpu::cuda
