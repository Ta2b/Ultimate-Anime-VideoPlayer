#include "app/play/auto_watch.h"

namespace ajp::app::play {

AutoActions AutoWatch::turn(const AutoTurn& t) {
  AutoActions out;
  if (t.free && t.at_us - free_logged_us_ >= kFreeLogEveryUs) {
    free_logged_us_ = t.at_us;
    out.log_free = true;
  }
  // Un changement de palier : sa ligne de journal ; l'annonce et la mémoire attendent qu'il tienne.
  if (t.centi != shown_) {
    out.log_change = t.centi > 0;
    shown_ = t.centi;
    changed_us_ = t.at_us;
    return out;
  }
  if (t.centi <= 0) return out;
  if (t.at_floor) {
    if (floor_since_us_ == 0) floor_since_us_ = t.at_us;
    if (t.at_us - floor_since_us_ >= kFloorLogAfterUs &&
        (!floor_logged_us_ || t.at_us - *floor_logged_us_ >= kFloorLogEveryUs)) {
      floor_logged_us_ = t.at_us;
      out.floor_us = t.at_us - floor_since_us_;
    }
  } else {
    floor_since_us_ = 0;
  }
  if (t.centi != notified_ && t.at_us - changed_us_ >= kAutoNotifyAfterUs) {
    if (notified_ > 0) out.notify = t.centi;  // le premier palier ne s'annonce pas : le départ
    notified_ = t.centi;
  }
  if (t.centi != stored_ && t.at_us - changed_us_ >= kAutoStoreAfterUs && t.can_store) {
    out.store = t.centi;
    stored_ = t.centi;  // même en cas d'échec de l'écriture : pas de réécriture à chaque tour
  }
  return out;
}

}  // namespace ajp::app::play
