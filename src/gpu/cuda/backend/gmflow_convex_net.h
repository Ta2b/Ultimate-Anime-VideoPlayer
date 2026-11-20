// Masque appris (3 × 3, Relu, 1 × 1) et agrandissement convexe ×4 de GMFlow natif.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "prepare/gmflow_upsampler.h"

namespace kirari::gpu::cuda {

class GmflowConvexNet {
 public:
  virtual ~GmflowConvexNet() = default;
  static std::unique_ptr<GmflowConvexNet> create(const prepare::GmflowUpsampler& weights, int height,
                                                 int width, std::string& error);
  virtual std::int64_t workspace_bytes() const = 0;
  // Au quart : features [2][H*W][128] FP16, flow [2][H*W][2] FP32 (xy). Résultat [2][2][4H][4W] FP32
  // planaire. Les quatre tampons sont distincts, workspace et features alignés à 16 octets.
  // Capture de graphe CUDA possible.
  virtual cudaError_t run(cudaStream_t stream, const void* features, const float* flow, void* workspace,
                          float* output) const = 0;
};

}  // namespace kirari::gpu::cuda
