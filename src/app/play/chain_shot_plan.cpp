// L'Auto plan par plan dans la chaîne (shot_plan.h, --shot-plan) : DistilDRBA chargé à côté de RIFE
// mouvement, quand les deux peuvent servir la même forme de chaîne (variantes mouvement, warp après SPAN, à
// la même taille d'estimation). L'étage gpu (upscale_shot.cpp) choisit ensuite plan par plan.
#include "app/play/chain_setup.h"
#include "app/play/settings.h"
#include "core/log.h"

namespace kirari::app::play {

bool drba_for_shots(const ChainEngines& c, const PlaySettings& settings) {
  return c.shot_plan && settings.rife_factor >= 2 && settings.motion_engine == MotionEngine::Rife &&
         c.drba_found && c.rife_model.has_value() && settings.upscale && !c.smooth_drawings &&
         warp_after_span(settings.span_order, c.work.height * kSpanFactor, c.order_factor);
}

void settle_shot_plan(ChainEngines& c) {
  if (!c.drba_by_shot) {
    if (c.shot_plan)
      log_channel("app")->info("plan par plan : un seul moteur d'interpolation, TemporalFix par plan");
    return;
  }
  const char* why = c.rife_motion == nullptr       ? "RIFE mouvement non chargé"
                    : c.rife != nullptr            ? "RIFE image chargé"
                    : !c.drba_motion               ? "DistilDRBA en variante image"
                    : c.engine == nullptr          ? "sans SPAN"
                    : c.drba_size != c.rife_size   ? "DistilDRBA et RIFE n'estiment pas à la même taille"
                    : c.gmfss_use != GmfssUse::Off ? "GMFSS union chargé"
                    : c.smooth_drawings            ? "dessins tenus lissés"
                                                   : nullptr;
  if (why == nullptr) {
    log_channel("app")->info(
        "plan par plan : DistilDRBA et RIFE mouvement chargés ensemble, estimés en {}x{}", c.rife_size.width,
        c.rife_size.height);
    return;
  }
  log_channel("app")->warn("plan par plan : {}, DistilDRBA rendu", why);
  c.drba_by_shot = false;
  // Sans aucun RIFE, DistilDRBA reste le seul moteur plutôt que rien.
  if (c.rife != nullptr || c.rife_motion != nullptr) {
    c.drba.reset();
    c.drba_motion = false;
    c.drba_fused = false;
  }
}

}  // namespace kirari::app::play
