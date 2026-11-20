// Taille d'un lot de plans de splat FP32, poids de normalisation inclus.
#pragma once

#include <cstddef>

namespace kirari::gpu::cuda::gmfss_splat {

inline constexpr std::size_t kAlignment = 256;
inline constexpr std::size_t kChannelsPerPass = 4;  // les quatre plans en mémoire partagée de splat_all

constexpr std::size_t aligned_bytes(std::size_t bytes) {
  return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

// Nombre de canaux dans une étape, ou zéro si le cache ne peut même pas accueillir un passage complet.
// La dernière étape peut être incomplète, comme dans le splat ordinaire.
constexpr int stage_channels(std::size_t plane, int channels, std::size_t budget) {
  if (plane == 0 || channels <= 0) return 0;
  const std::size_t slots = (budget / kAlignment * kAlignment) / sizeof(float) / plane;
  if (slots <= 1) return 0;
  const std::size_t usable = slots - 1;
  if (static_cast<std::size_t>(channels) <= usable) return channels;
  return static_cast<int>(usable / kChannelsPerPass * kChannelsPerPass);
}

}  // namespace kirari::gpu::cuda::gmfss_splat
