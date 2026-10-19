// Tables de poids du rééchantillonnage séparable (kernels/spec/resample.h), calculées sur l'hôte
// à la préparation d'un plan, puis rangées sur l'appareil.
// Repris de animejanai-inference-rtx@118c272 src/resample.h:53-84 (compute_filter de zimg) et
// src/pipeline_kernels.cu:34-50 (empreinte de Area), dans l'écriture de kernels/ref/resample.cpp :
// mêmes opérations en double, dans le même ordre, donc les mêmes float.
// Tests : tests/unit/cuda_compute/resize_test.cpp (égalité avec kernels::ref::compute_weights).
#pragma once

#include <cuda_runtime_api.h>
#include <vector>

#include "gpu/cuda/compute/kernels/views.h"
#include "gpu/cuda/compute/memory.h"
#include "gpu/cuda/compute/refusal.h"
#include "kernels/spec/resample.h"

namespace ajp::gpu::cuda {

struct HostWeights {
  int taps = 0;
  int src = 0;
  int dst = 0;
  std::vector<int> start;  // [dst] : premier indice source, avant le miroir
  std::vector<float> w;    // [dst × taps] : poids normalisés
};

// Table de dst sorties sur src entrées. Refus : dimensions hors de [1, kMaxDimension], décalage
// non fini ou hors de [−1, 1], Area qui agrandirait ou décalerait.
Result<HostWeights> compute_weights(int src, int dst, double shift, kernels::Filter filter);

// Une table rangée sur l'appareil, prête pour les passes.
class DeviceWeights {
 public:
  DeviceWeights() = default;
  // Copie synchrone sur stream : à la préparation seulement.
  static Result<DeviceWeights> upload(DevicePool& pool, const HostWeights& weights, cudaStream_t stream);
  // Calcul et copie en un appel.
  static Result<DeviceWeights> create(DevicePool& pool, int src, int dst, double shift,
                                      kernels::Filter filter, cudaStream_t stream);

  const dev::Weights& view() const { return view_; }

 private:
  DeviceBuffer start_;
  DeviceBuffer w_;
  dev::Weights view_;
};

}  // namespace ajp::gpu::cuda
