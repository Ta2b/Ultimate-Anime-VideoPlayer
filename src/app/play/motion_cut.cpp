#include "app/play/motion_cut.h"

#include "app/play/chain.h"
#include "app/play/chain_stats.h"
#include "app/play/i18n.h"
#include "app/play/session.h"
#include "app/play/ui.h"
#include "core/clock.h"
#include "core/log.h"

namespace ajp::app::play {

bool MotionGuard::held(Chain& chain) {
  if (cut_ && *cut_ != interpolation_choice(chain.running)) {
    log_channel("app")->info("interpolation rendue : son choix a changé dans les réglages");
    cut_.reset();
    watch_.reset();
    chain.live.set_rife_factor(factor_of(chain.running));
    chain.ui.set_motion_note({});
  }
  return cut_.has_value();
}

bool MotionGuard::watch(Chain& chain, double room_hz, double source_fps, std::int64_t now_us) {
  if (cut_) return false;
  const bool counting = chain.running.auto_off && chain.live.free_active() && !chain.session.view().paused &&
                        !chain.live.hidden() && room_hz >= 2.0 * source_fps;
  if (!watch_.note(now_us, chain.live.motion_tally(), counting)) return false;
  log_channel("app")->warn("interpolation coupée : moins d'une image interpolée par paire depuis {} s ({})",
                           watch_.behind(), free_status_text(chain.live.free_info(now_us)));
  cut_ = interpolation_choice(chain.running);
  chain.live.set_rife_factor(Factor{.num = 1, .den = 1});  // la carte rendue à SPAN ; le moteur reste chargé
  chain.ui.controls().warn(i18n::ui_tr("Interpolation off: the graphics card cannot keep up"), now_us);
  chain.ui.set_motion_note(
      i18n::tr("Interpolation is off for this video: the graphics card could not keep up. It comes back with "
               "the next video, with Retry, or when an interpolation setting changes."));
  return true;
}

}  // namespace ajp::app::play
