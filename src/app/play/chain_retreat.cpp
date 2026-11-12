#include "app/play/chain_retreat.h"

#include "app/play/links.h"
#include "app/play/stage_fallback.h"

namespace ajp::app::play {

bool retreats(const Refusal& failure, const ChainEngines& c) {
  return (failure.key == kFatalStageKey || display_retreats(failure)) &&
         failure.code != RefusalCode::DeviceLost && c.backend && c.backend->kernels()->health();
}

}  // namespace ajp::app::play
