// Le MLP fondu des couches de cross-attention du transformeur de GMFlow natif (docs/gmflow-natif.md, « MLP
// natif (PR 3) ») : les couches de prepare/gmflow_mlp_plan.h exécutées par le noyau de
// gmflow_mlp_kernels.cuh. Pas encore branché : le transformeur reste dans la paire TensorRT ;
// tools/gmfss/gmflow_mlp/gmflow_mlp_banc.cpp le chronomètre. Sans std::expected, car nvcc lit cet en-tête en
// C++20.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "prepare/gmflow_mlp_plan.h"

namespace kirari::gpu::cuda {

// Le bloc du noyau : 64 jetons (4 warps, deux blocs par SM, 45 Ko de mémoire partagée, toute carte) ou 128 (8
// warps, un bloc par SM, 78 Ko : pas sur Turing). Le banc dira lequel garder.
enum class GmflowMlpBlock : int { k64 = 4, k128 = 8 };

class GmflowMlpNet {
 public:
  virtual ~GmflowMlpNet() = default;
  // max_tokens : le plus de jetons d'un lancement (lot × hauteur × largeur de l'échelle). Sommes en f32. Rien
  // et error rempli si le MLP sort de ce que le noyau fait, si la carte n'a pas la mémoire partagée du bloc,
  // ou si la mémoire manque.
  static std::unique_ptr<GmflowMlpNet> create(const prepare::GmflowMlp& mlp, long max_tokens,
                                              GmflowMlpBlock block, std::string& error);
  virtual std::int64_t workspace_bytes() const = 0;
  // La couche layer sur tokens jetons, x, message et out en [tokens][128] FP16 :
  // out = x + LayerNorm(W2 · GELU(W1 · [x, message])). out peut être x. Capturable dans un graphe CUDA.
  virtual cudaError_t run(cudaStream_t stream, int layer, const void* x, const void* message, void* out,
                          long tokens) const = 0;
};

}  // namespace kirari::gpu::cuda
