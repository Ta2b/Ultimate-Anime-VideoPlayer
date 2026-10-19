// Rééchantillonnage séparable de kernels/spec/resample.h (Spline36, bilinéaire, Area) sur des
// tenseurs planaires : passe horizontale vers un intermédiaire float, puis verticale, rien n'est
// borné entre les deux (kernels::ref::resize). Le plan range ses tables et son intermédiaire
// dans le pool une fois ; run() n'alloue rien.
#pragma once

#include <cuda_runtime_api.h>

#include "gpu/cuda/compute/image.h"
#include "gpu/cuda/compute/memory.h"
#include "gpu/cuda/compute/weights.h"

namespace ajp::gpu::cuda {

struct ResizeSpec {
  int src_width = 0;
  int src_height = 0;
  int dst_width = 0;
  int dst_height = 0;
  kernels::Filter filter = kernels::Filter::Spline36;
  int channels = 3;
};

class ResizePlan {
 public:
  // Refus : dimensions qui ne sont pas celles d'une image de la chaîne (paires, dans
  // [2, kMaxDimension]), Area qui agrandirait, canaux hors de [1, 16], pool insuffisant.
  static Result<ResizePlan> create(DevicePool& pool, const ResizeSpec& spec, cudaStream_t stream);

  // in : src_width × src_height ; out : dst_width × dst_height ; F16 ou F32 chacun.
  Result<> run(const Tensor& in, const Tensor& out, cudaStream_t stream) const;

 private:
  ResizeSpec spec_;
  DeviceWeights wx_;
  DeviceWeights wy_;
  DeviceBuffer tmp_;
};

}  // namespace ajp::gpu::cuda
