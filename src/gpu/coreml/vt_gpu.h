// Les conversions de l'interpolation d'Apple (coreml_vt.h) sur le GPU de la puce, par Metal : les
// plans RGB FP16 de la chaîne, dans leurs IOSurface (coreml_surfaces.h), écrits dans les tampons de
// VideoToolbox au format retenu et relus, comme vt::pack et vt::unpack (vt_pixels.h) le font sur le
// CPU ; et la reconnaissance d'une paire déjà vue, par comparaison et copie des plans sur le GPU.
// Les images ne passent plus par les cœurs du CPU.
//
// Objective-C++, sur un Mac seulement.
#pragma once

#include <CoreVideo/CoreVideo.h>
#include <IOSurface/IOSurfaceRef.h>
#include <array>
#include <expected>
#include <memory>
#include <span>

#include "core/types.h"
#include "gpu/coreml/vt_pixels.h"

namespace ajp::gpu::coreml_rt {

// Chaque appel encode son travail, le soumet et attend sa fin. Les IOSurface de la chaîne sont
// des tenseurs FP16 NCHW (une rangée de width éléments par ligne d'image) : l'entrée 1 × 7 × H × W
// (image a aux rangées 0 à 3H, image b de 3H à 6H), la sortie 1 × 3 × H × W ; prêtées au GPU par
// l'appelant. Les tampons de VideoToolbox sont des IOSurface au format d'open, width × height.
class VtGpu {
 public:
  // Le GPU du système et les noyaux de ce format, pour des images width × height. Refus
  // FeatureUnavailable : pas de GPU Metal, noyaux non compilés, images trop grandes pour une
  // texture ou rangées que Metal ne prend pas.
  static std::expected<std::unique_ptr<VtGpu>, Refusal> open(vt::Pixels format, int width, int height);

  VtGpu() = default;
  VtGpu(const VtGpu&) = delete;
  VtGpu& operator=(const VtGpu&) = delete;
  VtGpu(VtGpu&&) = delete;
  VtGpu& operator=(VtGpu&&) = delete;
  virtual ~VtGpu() = default;

  // Une image de l'entrée (0 : a, 1 : b) écrite dans target, et gardée dans la place keep (0 ou 1).
  struct Pack {
    int image = 0;
    CVPixelBufferRef target = nullptr;
    int keep = 0;
  };
  virtual std::expected<void, Refusal> pack(IOSurfaceRef input, std::span<const Pack> packs) = 0;
  // Les images de l'entrée comparées bit à bit aux gardées : a ≠ gardée 0, b ≠ gardée 1, a ≠ gardée 1.
  virtual std::expected<std::array<bool, 3>, Refusal> differs(IOSurfaceRef input) = 0;
  // Échange les deux gardées (la paire suivante commence par l'image b de celle-ci).
  virtual void swap_kept() = 0;
  // source, un tampon de VideoToolbox, relu dans les 3 plans de output.
  virtual std::expected<void, Refusal> unpack(CVPixelBufferRef source, IOSurfaceRef output) = 0;
  // Une image de l'entrée copiée telle quelle dans output.
  virtual std::expected<void, Refusal> copy(IOSurfaceRef input, int image, IOSurfaceRef output) = 0;
};

}  // namespace ajp::gpu::coreml_rt
