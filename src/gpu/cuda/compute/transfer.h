// Fonctions de transfert (PQ, HLG, sRGB, BT.1886) et conversion de primaires de
// kernels/spec/transfer.h, pixel par pixel sur trois plans : décodage vers la lumière linéaire,
// OOTF HLG éventuelle, gain, matrice de primaires éventuelle, puis encodage. Chaque étape suit
// la fonction de kernels/ref/transfer.h du même nom.
#pragma once

#include <cuda_runtime_api.h>
#include <optional>

#include "gpu/cuda/compute/image.h"
#include "kernels/spec/transfer.h"

namespace ajp::gpu::cuda {

using dev::Curve;

struct TransferSpec {
  Curve decode = Curve::Linear;  // Pq : cd/m² ; Hlg : lumière de scène (inverse de l'OETF)
  Curve encode = Curve::Linear;
  std::optional<float> hlg_ootf_peak;  // OOTF HLG pour un écran de ce pic, dans [400, 2000] cd/m²
  float bt1886_white = 1.0F;           // écran BT.1886 : blanc > noir ≥ 0
  float bt1886_black = 0.0F;
  float gain = 1.0F;                       // fini
  std::optional<kernels::Mat3> primaries;  // en lumière linéaire, rangée en float
};

// Les constantes de l'écran, évaluées comme dans la référence. Refus : pic HLG ou écran BT.1886
// invalides, gain non fini.
Result<dev::TransferParams> make_transfer(const TransferSpec& spec);

// in et out : 3 plans de même taille, F16 ou F32 chacun.
Result<> transfer(const Tensor& in, const dev::TransferParams& params, const Tensor& out,
                  cudaStream_t stream);

}  // namespace ajp::gpu::cuda
