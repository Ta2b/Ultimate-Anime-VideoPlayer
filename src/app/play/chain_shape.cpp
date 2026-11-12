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

int order_factor(const PlaySettings& settings, double display_hz, double source_fps) {
  if (settings.rife_auto && settings.rife_factor >= 2) return 0;
  return top_factor(settings, display_hz, source_fps);
}

static_assert(kMaxRingSlots >= 8);

std::uint32_t ring_depth(const PlaySettings& settings, Factor factor, double display_hz, double source_fps) {
  if (settings.auto_free && settings.rife_auto && factor.num > factor.den) {
    if (display_hz <= 0.0 || source_fps <= 0.0) return 8U;
    factor = Factor{.num = top_factor(settings, display_hz, source_fps), .den = 1};
  }
  const int den = std::max(factor.den, 1);
  return factor.num >= 5 * den ? 8U : factor.num >= 3 * den ? 6U : 4U;
}

ChainShape chain_shape(const PlaySettings& settings, const OpenedChain& at) {
  return {
      .ring =
          ring_depth(settings, at.rife_loaded ? factor_of(settings) : Factor{}, at.display_hz, at.source_fps),
      .warp_after_span = at.span && warp_after_span(settings.span_order, at.output_height,
                                                    order_factor(settings, at.display_hz, at.source_fps))};
}

OpenedChain::Loaded loaded_by(const PlaySettings& settings) {
  return {.motion_engine = settings.motion_engine,
          .rife_model = settings.rife_model,
          .motion_precision = settings.motion_precision,
          .span_order = settings.span_order,
          .rife_fused = settings.rife_fused,
          .drba_fused = settings.drba_fused,
          .smooth_drawings = settings.smooth_drawings,
          .cubic_warp = settings.cubic_warp,
          .pair_scale = settings.pair_scale};
}

bool fits(const OpenedChain& opened, const PlaySettings& wanted) {
  if (wanted.rife_factor < 2) return true;
  const ChainShape shape = chain_shape(wanted, opened);
  return shape.ring <= opened.shape.ring && shape.warp_after_span == opened.shape.warp_after_span;
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
