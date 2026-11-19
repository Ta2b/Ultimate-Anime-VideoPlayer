// Les slots de l'export par puits CUDA et le côté de l'échelle par paire, sortis d'upscale_stage.h, qui
// touchait la limite de taille des fichiers.
#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

#include "gpu/kernel_ops.h"

namespace kirari::app::play {

// Les slots de l'export par puits CUDA (export_sink.h) : chaque slot est une image de l'encodeur,
// que write_slot remplit en YUV (KernelOps::write_yuv) au lieu du ring ; done_value est la valeur
// qu'aurait signalée le slot du ring (RingFrame::ready_value). Appelé depuis le thread gpu.
class YuvSlots {
 public:
  virtual ~YuvSlots() = default;
  virtual gpu::YuvTarget target(std::uint32_t slot, std::uint64_t done_value) const = 0;
  // Le bloc qui loge les signaux done des cibles, aligné sur une page et en multiple de pages, pour que le
  // backend y écrive depuis le GPU (KernelOps::map_fences) ; vide : rappels hôtes.
  virtual std::span<std::byte> fence_block() const { return {}; }
};

// Échelle par paire (PlaySettings::pair_scale) : le côté où le second RIFE estime les paires à mouvement
// fort, la moitié de celui de RIFE, pair (1080 → 540, 1280 → 640).
constexpr int pair_scale_side(int rife_side) {
  return std::max(2, rife_side / 4 * 2);
}

}  // namespace kirari::app::play
