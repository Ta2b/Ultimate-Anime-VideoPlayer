// Le corps du SPAN FP8 à 48 canaux sans rembourrage (kKernelsVersion 4, chantier SPAN 1) : un b5f48 rangé en
// 64 dans son paquet faisait 44 % de son travail sur des zéros. À part de span_fp8_net.cu pour que chaque
// unité de compilation garde ses instanciations CUTLASS et se compile en parallèle.
#include "gpu/cuda/backend/span_fp8_body.cuh"

namespace ajp::gpu::cuda {

std::unique_ptr<SpanFp8Body> make_span_fp8_body48(int height, int width, bool flat) {
  if (flat) return std::make_unique<Body<48, Alg::kFewChannels, false>>(height, width);
  return std::make_unique<Body<48, Alg::kOptimized, false>>(height, width);
}

}  // namespace ajp::gpu::cuda
