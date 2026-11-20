// Le GridNet de la fusion de GMFSS Fortuna Union par les noyaux de rife_native_kernels.cuh (--gmfss-native,
// docs/gmfss-natif.md), à la place de sa part du moteur TensorRT de la fusion ; l'IFNet (RIFE 4.26) reste un
// moteur TensorRT et lui donne son image. Les entrées sont celles de la fusion, en plans FP16 ; la sortie,
// les trois plans de l'image au double de la taille, bornés à [0, 1]. En-tête sans std::expected : il est lu
// par nvcc en C++20.
#pragma once

#include <array>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "gpu/cuda/backend/gmfss_native_parts.h"
#include "prepare/gmfss_grid.h"

namespace kirari::gpu::cuda {

// Les plans des entrées : warped0_1..3 et warped1_1..3 (3 + 64, 128 et 192 plans à la taille de la fusion,
// sa moitié et son quart), puis l'image de l'IFNet (3 plans).
struct GmfssNativeInputs {
  std::array<const void*, 3> warped0{};
  std::array<const void*, 3> warped1{};
  const void* rife = nullptr;
  bool packed_features = false;  // les splats ont écrit les six tranches de x1 à x3
};

// Les morceaux (GmfssNativeParts) : ligne 0, ligne 1 et ligne 2 (les convolutions qui écrivent un tenseur de
// ce niveau : 64 canaux à la taille de la fusion, 128 à la moitié, 192 au quart), puis la queue (la
// convolution d'avant le PixelShuffle, celle du PixelShuffle et conv_last, la tête).
class GmfssNativeNet : public GmfssNativeParts {
 public:
  // height × width : la taille de la fusion (multiples de 32). Les activations sont rangées une fois (deux
  // tenseurs qui vivent ensemble ne se recouvrent pas) dans une zone allouée ici ; chaque convolution est
  // réglée au banc sur stream pour les trois sommes, celles posées étant f32 partout. Rien et error rempli si
  // une convolution sort de ce que les noyaux font, ou sur une erreur CUDA (mémoire, banc).
  static std::unique_ptr<GmfssNativeNet> create(const prepare::GmfssGrid& grid, int height, int width,
                                                cudaStream_t stream, std::string& error,
                                                bool features_c8 = false);
  // Le réseau sur stream : out reçoit 3 plans FP16 de 2·height × 2·width.
  virtual cudaError_t run(cudaStream_t stream, const GmfssNativeInputs& in, void* out) const = 0;
  // x1 à x3, avec les canaux de A puis B dans chaque pixel.
  virtual std::array<void*, 3> feature_inputs() const = 0;
  // Le placement par anneaux est essayé après le choix des précisions et de l'IFNet.
  virtual bool join_ready() const = 0;
  virtual void use_join(bool enabled) = 0;
  // Les convolutions d'un morceau seul, sur ce que la zone garde : pour son temps.
  virtual cudaError_t run_piece(cudaStream_t stream, int piece) const = 0;
};

}  // namespace kirari::gpu::cuda
