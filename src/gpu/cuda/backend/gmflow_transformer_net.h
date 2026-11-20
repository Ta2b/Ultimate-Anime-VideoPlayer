// Les six blocs du transformeur de GMFlow à une échelle (docs/gmflow-natif.md).
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "gpu/cuda/backend/gmflow_attention_net.h"
#include "gpu/cuda/backend/gmflow_mlp_net.h"

namespace kirari::gpu::cuda {

class GmflowTransformerNet {
 public:
  // 0 : huitième (lot 2), 1 : quart (lot 4). Les deux plans proviennent de la
  // même paire ONNX.
  static std::unique_ptr<GmflowTransformerNet> create(const prepare::GmflowAttention& attention,
                                                      const prepare::GmflowMlp& mlp, int scale, int height,
                                                      int width, GmflowMlpBlock block, std::string& error);

  // Deux tenseurs de jetons FP16, en plus des allocations propres à l'attention
  // et au MLP.
  std::size_t scratch_bytes() const { return 2 * token_bytes_; }

  // in et out : [lot][hauteur][largeur][128] FP16. L'entrée contient déjà les
  // positions et, au quart, les caractéristiques déformées. in, out et scratch
  // sont distincts, alignés sur 16 octets. Une instance par voie. Les
  // lancements sont capturables dans un graphe CUDA.
  cudaError_t run(cudaStream_t stream, const void* in, void* out, void* scratch) const;

 private:
  GmflowTransformerNet(std::unique_ptr<GmflowAttentionNet> attention, std::unique_ptr<GmflowMlpNet> mlp,
                       long tokens, std::size_t token_bytes);

  std::unique_ptr<GmflowAttentionNet> attention_;
  std::unique_ptr<GmflowMlpNet> mlp_;
  long tokens_;
  std::size_t token_bytes_;
};

}  // namespace kirari::gpu::cuda
