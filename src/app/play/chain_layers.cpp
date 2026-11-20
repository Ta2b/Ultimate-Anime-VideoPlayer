// La plaque de décor dans la chaîne (chain_setup.h, plan_plate ; docs/calques.md) : la sortie de SPAN sur une
// image clé par plan, reprise par les sources suivantes du plan (layer_stage.h).
#include "app/play/chain_setup.h"
#include "core/log.h"

namespace kirari::app::play {

void plan_plate(ChainEngines& c, const PlaySettings& settings) {
  const auto log = log_channel("app");
  if ((settings.background_plate || settings.layers) && !c.scene_map) {
    log->info("calques et plaque de décor : sans --scene-map, pas de carte de la scène, éteints");
    return;
  }
  if (!settings.background_plate || c.engine == nullptr) return;
  c.plate = true;
  log->info(
      "plaque de décor : la sortie de SPAN d'une image clé par plan, reprise par les sources suivantes");
}

LayerSetup layer_setup(const ChainEngines& c, const MasterClock* clock) {
  if (clock == nullptr || !c.scene_map) return {};
  return {.layers = c.layers && c.rife_motion != nullptr,
          .plate = c.plate,
          .fused = c.fused_layers,
          .nvof_guard = c.nvof_layer_guard};
}

}  // namespace kirari::app::play
