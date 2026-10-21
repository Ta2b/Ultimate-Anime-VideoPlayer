// Tables pré-réglées du moteur Vulkan (net_autotune.h : PresetTable), générées par
// tools/vk/presets_from_cache.py depuis le cache de l'autotuner d'un PC de référence : ne pas
// modifier à la main. Version des candidats : c5.
#include <array>
#include <span>

#include "gpu/vulkan/backend/net_autotune.h"

namespace ajp::gpu::vk::net {

std::span<const PresetTable> preset_tables() {
  static constexpr std::array<PresetTable, 0> kTables{};
  return kTables;
}

}  // namespace ajp::gpu::vk::net
