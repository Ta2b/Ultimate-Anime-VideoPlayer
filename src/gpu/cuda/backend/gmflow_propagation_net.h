// Les projections de la propagation de GMFlow, exécutées par la GEMM du transformeur.
#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "prepare/gmflow_propagation_plan.h"

namespace kirari::gpu::cuda {

class GmflowPropagationNet {
 public:
  virtual ~GmflowPropagationNet() = default;
  static std::unique_ptr<GmflowPropagationNet> create(const prepare::GmflowPropagation& weights,
                                                      std::string& error);

  // [2][pixels][128] FP16, régions distinctes et alignées sur 16 octets. Sommes et biais en FP32.
  // En global les clés lisent les requêtes projetées. En local elles lisent les mêmes caractéristiques.
  virtual cudaError_t project(cudaStream_t stream, const void* features, long pixels, bool global,
                              void* query, void* keys) const = 0;
};

}  // namespace kirari::gpu::cuda
