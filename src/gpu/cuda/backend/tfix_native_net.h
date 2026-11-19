// TemporalFix natif (--tfix-native, docs/tfix-natif.md) : les cinq morceaux de convolutions de TemporalFix
// fusionné (prepare/tfix_split.h : l'encodeur, les blocs 0 à 2, la tête de confiance) par les noyaux de
// rife_native_kernels.cuh, à la place des cinq moteurs TensorRT. La colle, les warps et le mélange restent
// ceux de tfix_fused_kernels.h, qui écrivent et lisent alors X, Y et la tête par pixel (TfixShape::native).
//
// - L'encodeur lit les trois plans d'une image du réseau, et écrit ses quatre plans de traits à la même
//   taille (sa dernière convolution, à demi-résolution, agrandie ×2 en bilinéaire demi-pixel : le Resize du
//   graphe) ;
// - un bloc ou la tête lit les six appuis en un lot (blockIdx.z), au pas kernels::tfix_in_ld, et écrit au pas
//   kernels::tfix_out_ld.
// En-tête sans std::expected : il est lu par nvcc en C++20.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

#include "gpu/cuda/backend/rife_native_net.h"

namespace kirari::gpu::cuda {

class TfixNativeNet {
 public:
  virtual ~TfixNativeNet() = default;
  // pieces : l'encodeur, les blocs 0 à 2 et la tête (couches de prepare::rife_native_layers, sans
  // transposée) ; height × width : le réseau (pair, kernels::tfix_size_valid). acc32 : sommes en f32. Rien et
  // error rempli si une couche sort de ce que les noyaux font, ou si les poids ne s'envoient pas.
  static std::unique_ptr<TfixNativeNet> create(const std::vector<RifeNativePiece>& pieces, int height,
                                               int width, bool acc32, std::string& error);
  // La zone de travail (activations entre les couches, sommes partielles) : octets, de l'encodeur seul, et de
  // tous les morceaux (le plus grand).
  virtual std::int64_t encoder_workspace_bytes() const = 0;
  virtual std::int64_t workspace_bytes() const = 0;
  // L'encodeur sur une image (3 plans height × width) vers traits (4 plans height × width).
  virtual cudaError_t encode(cudaStream_t stream, const void* image, void* traits, void* workspace) const = 0;
  // Le morceau k de la fenêtre (0 à 2 : les blocs ; 3 : la tête), sur les six appuis : x vers y.
  virtual cudaError_t window(cudaStream_t stream, int k, const void* x, void* y, void* workspace) const = 0;
  virtual bool acc32() const = 0;
};

}  // namespace kirari::gpu::cuda
