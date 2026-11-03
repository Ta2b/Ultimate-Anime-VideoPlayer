// Un emplacement de l'anneau vu des noyaux CUDA (étape 5.5b) : la surface et son empaquetage. Les noyaux
// l'écrivent par store_ring (pixel_kernels.cuh), seule fonction d'écriture de l'anneau.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace ajp::gpu::cuda {

enum class RingPacking : std::uint32_t {
  Rgba16,   // RGBA16 UNORM, 8 octets par pixel (kernels/spec/color.h, unorm16)
  Rgb10a2,  // RGB10A2 UNORM, 4 octets par pixel (kernels/spec/ring_pack.h)
};

struct RingSurface {
  cudaSurfaceObject_t object = 0;  // 0 : pas d'emplacement
  RingPacking packing = RingPacking::Rgba16;
  bool dither = false;  // RGB10A2 seulement : tramage TPDF fixe par pixel
};

}  // namespace ajp::gpu::cuda
