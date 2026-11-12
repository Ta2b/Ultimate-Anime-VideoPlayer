#include "app/play/chain_retreat.h"

#include "app/play/links.h"
#include "app/play/stage_fallback.h"

namespace ajp::app::play {

bool retreats(const Refusal& failure, const ChainEngines& c) {
  return (is_fatal_stage_key(failure.key) || display_retreats(failure)) &&
         failure.code != RefusalCode::DeviceLost && c.backend && c.backend->kernels()->health();
}

}  // namespace ajp::app::play
