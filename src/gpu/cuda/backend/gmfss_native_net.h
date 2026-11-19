// Le GridNet de la fusion de GMFSS Fortuna Union par les noyaux de rife_native_kernels.cuh (--gmfss-native,
// docs/gmfss-natif.md), à la place de sa part du moteur TensorRT de la fusion ; l'IFNet (RIFE 4.26) reste un
// moteur TensorRT et lui donne l'image « rife ». Les entrées sont celles de la fusion, en plans FP16 ; la
// sortie, les trois plans de l'image au double de la taille, bornés à [0, 1]. En-tête sans std::expected : il
// est lu par nvcc en C++20.
#pragma once

#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "prepare/gmfss_grid.h"

namespace kirari::gpu::cuda {

// Les plans des entrées : warped0_1..3 et warped1_1..3 (3 + 64, 128 et 192 plans à la taille de la fusion,
// sa moitié et son quart), puis l'image de RIFE (3 plans).
struct GmfssNativeInputs {
  std::array<const void*, 3> warped0{};
  std::array<const void*, 3> warped1{};
  const void* rife = nullptr;
};

class GmfssNativeNet {
 public:
  virtual ~GmfssNativeNet() = default;
  // height × width : la taille de la fusion (multiples de 32). acc32 : sommes en f32. Les activations sont
  // rangées une fois (deux tenseurs qui vivent ensemble ne se recouvrent pas) dans une zone allouée ici. Rien
  // et error rempli si une convolution sort de ce que les noyaux font, ou si la mémoire manque.
  static std::unique_ptr<GmfssNativeNet> create(const prepare::GmfssGrid& grid, int height, int width,
                                                bool acc32, std::string& error);
  virtual std::int64_t workspace_bytes() const = 0;
  // Le réseau sur stream : out reçoit 3 plans FP16 de 2·height × 2·width.
  virtual cudaError_t run(cudaStream_t stream, const GmfssNativeInputs& in, void* out) const = 0;
  virtual bool acc32() const = 0;
};

}  // namespace kirari::gpu::cuda
