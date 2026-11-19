// La tête de confiance de TemporalFix natif fondue (tfix_head.cu, docs/tfix-natif.md) : ses trois
// convolutions (25 → 32 et 32 → 32 avec activation, puis 32 → 1) par tuiles, sans écrire ni relire ses deux
// activations du milieu. En-tête sans std::expected : il est lu par nvcc en C++20.
#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "gpu/cuda/backend/rife_native_net.h"

namespace kirari::gpu::cuda {

class TfixHead {
 public:
  virtual ~TfixHead() = default;
  // head : les trois convolutions de la tête (prepare::rife_native_layers). Rien, error rempli, si elles ne
  // sont pas de cette forme (une autre tête passe alors couche par couche) ou si les poids ne s'envoient pas.
  static std::unique_ptr<TfixHead> create(const RifeNativePiece& head, RifeNativeSums sums,
                                          std::string& error);
  // in : l'entrée de la tête des six appuis (h × w au pas 32) ; out : sa sortie (h × w au pas 8, le canal
  // puis sept zéros).
  virtual cudaError_t run(cudaStream_t stream, const void* in, void* out, int h, int w, int batch) const = 0;
};

}  // namespace kirari::gpu::cuda
