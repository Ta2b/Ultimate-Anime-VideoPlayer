// Instanciation du corps CUTLASS à 64 canaux. Celui à 48 canaux est dans span_fp8_net48.cu.
#include "gpu/cuda/backend/span_fp8_body.cuh"

namespace kirari::gpu::cuda {

std::unique_ptr<SpanFp8Body> make_span_fp8_body64(int height, int width) {
  return std::make_unique<Body<64, Alg::kOptimized>>(height, width);
}

}  // namespace kirari::gpu::cuda
