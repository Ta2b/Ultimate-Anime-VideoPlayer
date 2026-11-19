// Les couches d'attention du transformeur de GMFlow natif (docs/gmflow-natif.md, PR 2) : une couche du plan
// de prepare/gmflow_attention.h exécutée par les noyaux de gmflow_attention_kernels.cuh (la GEMM QKV, puis
// l'attention fondue avec la fusion, la LayerNorm et le résiduel). Pas encore branché : le MLP (PR 3) et le
// reste de GMFlow restent dans la paire TensorRT. Sans std::expected, car nvcc lit cet en-tête en C++20.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "prepare/gmflow_attention.h"

namespace kirari::gpu::cuda {

class GmflowAttentionNet {
 public:
  virtual ~GmflowAttentionNet() = default;
  // Une échelle du transformeur : scale 0 (le huitième) ou 1 (le quart), batch images de height × width
  // jetons (le lot de GMFlow : 2 au huitième, 4 au quart), coupées en fenêtres de splits[scale] par côté.
  // Sommes des GEMM (QKV, fusion) et de l'attention en f32. Rien et error rempli si la grille ne se coupe pas
  // en fenêtres égales, si le lot est impair ou si la mémoire manque.
  static std::unique_ptr<GmflowAttentionNet> create(const prepare::GmflowAttention& attention, int scale,
                                                    int batch, int height, int width, std::string& error);
  // Le tampon de q, k, v : [batch][height · width][384] FP16.
  virtual std::int64_t workspace_bytes() const = 0;
  // La couche layer (0 à 11) : query et keys [batch][height · width][128] FP16, out de même (distinct des
  // deux). Self : keys = query, out = query + LN. Cross : keys est l'entrée du bloc (k et v lus dans l'autre
  // moitié du lot), out = LN, le message du MLP. Capturable dans un graphe CUDA.
  virtual cudaError_t run_layer(cudaStream_t stream, int layer, const void* query, const void* keys,
                                void* out) const = 0;
};

}  // namespace kirari::gpu::cuda
