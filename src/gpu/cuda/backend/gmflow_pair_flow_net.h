// GMFlow complet à la frontière du moteur « flot » de la paire native de GMFSS.
#pragma once

#include <array>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <expected>
#include <memory>
#include <span>

#include "core/types.h"
#include "prepare/gmfss_pair_split.h"

namespace kirari::gpu::cuda {

struct GmflowPairOutput {
  std::array<void*, 2> half;                         // deux images [3][h][w] FP16
  std::array<void*, prepare::kGmfssPairFlows> flow;  // niveau 1/2/3, puis sens inverse, deux plans FP16
  void* metric = nullptr;                            // 14 plans FP16 à la taille de la fusion
};

class GmflowPairFlowNet {
 public:
  virtual ~GmflowPairFlowNet() = default;
  static std::expected<std::unique_ptr<GmflowPairFlowNet>, Refusal> create(std::span<const std::byte> pair,
                                                                           int height, int width);
  // Les deux images pleines [2][3][2h][2w] FP16. Une instance par voie.
  // follows : A est exactement la même source que B du dernier appel.
  virtual cudaError_t run(cudaStream_t stream, const void* frames, const GmflowPairOutput& out,
                          bool follows = false) const = 0;
};

}  // namespace kirari::gpu::cuda
