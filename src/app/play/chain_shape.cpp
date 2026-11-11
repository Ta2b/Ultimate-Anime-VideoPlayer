#include "app/play/chain_shape.h"

#include <algorithm>

#include "app/play/links.h"
#include "app/play/visible_error.h"
#include "core/auto_rate.h"

namespace ajp::app::play {

int top_factor(const PlaySettings& settings, double display_hz, double source_fps) {
  if (settings.rife_factor < 2) return 1;
  if (!settings.rife_auto) return settings.rife_factor / std::max(settings.rife_factor_den, 1);
  // Auto : jusqu'à rife_factor, et pas au-delà de ce que l'écran (ou le plafond choisi) affiche.
  const int ceiling = core::auto_ceiling(core::auto_limit_hz(display_hz, settings.auto_cap), source_fps);
  // Rapide : le warp et l'anneau suivent l'écran, pas le facteur demandé (6 à 165 Hz sur 23,976 i/s).
  if (settings.auto_free) return std::max(ceiling, 1);
  return std::clamp(ceiling, 1, settings.rife_factor);
}

static_assert(kMaxRingSlots >= 8);

std::uint32_t ring_depth(const PlaySettings& settings, Factor factor) {
  if (settings.auto_free && settings.rife_auto) return 8U;
  const int den = std::max(factor.den, 1);
  return factor.num >= 5 * den ? 8U : factor.num >= 3 * den ? 6U : 4U;
}

ChainShape chain_shape(const PlaySettings& settings, const OpenedChain& at) {
  return {.ring = ring_depth(settings, at.rife_loaded ? factor_of(settings) : Factor{}),
          .warp_after_span = at.span && warp_after_span(settings.span_order, at.output_height,
                                                        top_factor(settings, at.display_hz, at.source_fps))};
}

bool fits(const OpenedChain& opened, const PlaySettings& wanted) {
  if (wanted.rife_factor < 2) return true;
  const ChainShape shape = chain_shape(wanted, opened);
  return shape.ring <= opened.shape.ring && shape.warp_after_span == opened.shape.warp_after_span;
}

WorkSize drba_motion_size(const WorkSize& work, int source_width, int source_height, bool image) {
  return !image && work.height > kDrbaMotionHeight ? work_size(source_width, source_height, kDrbaMotionHeight)
                                                   : work;
}

i18n::Translated fused_refusal_notice(const OpenedChain& opened) {
  if (opened.drba_fused_refused)
    return i18n::ui_tr_format("Fused DistilDRBA unavailable ({}): the whole network runs, slower",
                              visible_error(Refusal{*opened.drba_fused_refused, ""}));
  if (opened.rife_fused_refused)
    return i18n::ui_tr_format("Fused RIFE unavailable ({}): the whole network runs, slower",
                              visible_error(Refusal{*opened.rife_fused_refused, ""}));
  return {};
}

}  // namespace ajp::app::play
