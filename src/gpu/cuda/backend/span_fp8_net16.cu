// Le SPAN FP16 en noyaux CUTLASS (lot 2 du fil SPAN) : le graphe du SPAN FP8, blocs en FP16 sans
// quantification, donc sans porte de qualité. Il sert là où le FP8 ne passe pas : b5f48 refusé par sa porte,
// et plus tard les cartes sans mma FP8. À part pour que ses instanciations CUTLASS se compilent en parallèle.
#include "gpu/cuda/backend/span_fp8_body.cuh"

namespace ajp::gpu::cuda {

std::unique_ptr<SpanFp8Body> make_span_fp8_body16(int height, int width, int channels) {
  if (channels == 48) return std::make_unique<Body<48, Alg::kFewChannels, true>>(height, width);
  return std::make_unique<Body<64, Alg::kOptimized, true>>(height, width);
}

}  // namespace ajp::gpu::cuda
