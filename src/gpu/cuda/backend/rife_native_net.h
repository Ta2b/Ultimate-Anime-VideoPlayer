// Le RIFE natif (--rife-native, docs/rife-natif.md) : l'encodeur et les convolutions des blocs de RIFE
// fusionné par les noyaux de rife_native_kernels.cuh, à la place des six moteurs TensorRT. Les formats sont
// ceux des moteurs en E/S HWC8 : l'encodeur lit les trois plans d'une image et écrit ses quatre plans
// d'encodage, un bloc lit X_k rangé par pixel (pas 16 ou 32) et écrit Y_k par pixel au pas 56. La colle, les
// warps et la fin restent ceux de rife_fused_kernels.h : RifeFusedEngine appelle ce réseau là où il appelait
// TensorRT. En-tête sans std::expected : il est lu par nvcc en C++20.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace kirari::gpu::cuda {

// Une couche, telle que prepare::RifeNativeLayer la lit (poids [cout][kernel²][cin] en flottants).
struct RifeNativeConv {
  bool transposed = false;
  int cin = 0;
  int cout = 0;
  int stride = 1;
  std::vector<float> weights;
  std::vector<float> bias;
  std::vector<float> slope;  // vide : pas d'activation
  std::vector<float> scale;  // vide : pas de résiduel
};
using RifeNativePiece = std::vector<RifeNativeConv>;

class RifeNativeNet {
 public:
  virtual ~RifeNativeNet() = default;
  // pieces : l'encodeur puis les blocs ; reduce : d_k de chaque bloc ; height × width : la taille remplie.
  // acc32 : sommes en f32. Rien et error rempli si une couche sort de ce que les noyaux font, ou si les
  // poids ne s'envoient pas.
  static std::unique_ptr<RifeNativeNet> create(const std::vector<RifeNativePiece>& pieces,
                                               const std::vector<int>& reduce, int height, int width,
                                               bool acc32, std::string& error);
  // La zone de travail (activations entre les couches) : octets, et son adresse pour les appels suivants.
  virtual std::int64_t workspace_bytes() const = 0;
  // L'encodeur sur une image (3 plans height × width) vers encoded (4 plans).
  virtual cudaError_t encode(cudaStream_t stream, const void* image, void* encoded,
                             void* workspace) const = 0;
  // Le bloc k : x (X_k par pixel), y (Y_k par pixel, pas 56).
  virtual cudaError_t block(cudaStream_t stream, int k, const void* x, void* y, void* workspace) const = 0;
  virtual bool acc32() const = 0;
};

}  // namespace kirari::gpu::cuda
