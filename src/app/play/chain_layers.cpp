// La plaque de décor dans la chaîne (chain_setup.h, load_plate ; docs/calques.md) : un second moteur SPAN,
// le réseau tel que livré en FP16, que l'étage gpu passe sur une image clé par plan (layer_stage.h).
#include <format>

#include "app/play/chain_setup.h"
#include "app/play/i18n.h"
#include "app/play/player_engines.h"
#include "core/log.h"
#include "core/utf8_path.h"

namespace kirari::app::play {

std::optional<bool> load_plate(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                               const PlaySettings& settings) {
  const auto log = log_channel("app");
  if ((settings.background_plate || settings.layers) && !c.scene_map) {
    log->info("calques et plaque de décor : sans --scene-map, pas de carte de la scène, éteints");
    return true;
  }
  if (!settings.background_plate || c.engine == nullptr || !c.model) return true;
  if (!settings.layers)
    log->info(
        "plaque de décor : elle suit la carte de la scène, comme les calques ; les images intermédiaires "
        "gardent RIFE seul");
  const gpu::Shape shape{.dims = {1, 3, c.work.height, c.work.width}};
  auto engine = prepare_engine(p, init, *c.backend, c.model->path, shape,
                               EngineLabel{.name = i18n::tr("Background plate"),
                                           .model = core::to_utf8(c.model->path.stem()),
                                           .width = c.work.width,
                                           .height = c.work.height},
                               gpu::Lane::Upscale);
  if (!engine) {
    log->warn("plaque de décor refusée ({}) : lecture sans elle", engine.error().detail);
    return true;
  }
  if (!*engine) return std::nullopt;  // fenêtre fermée pendant la construction
  c.plate = std::move(**engine);
  log->info("plaque de décor : SPAN {} FP16 sur une image clé par plan", core::to_utf8(c.model->path.stem()));
  return true;
}

LayerSetup layer_setup(const ChainEngines& c, const MasterClock* clock) {
  if (clock == nullptr || !c.scene_map) return {};
  return {.layers = c.layers && c.rife_motion != nullptr, .plate = c.plate.get()};
}

}  // namespace kirari::app::play
