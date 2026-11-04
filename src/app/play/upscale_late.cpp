// Rapide à pas constant, producteur « une image à la fois » (chantier 7.6, UpscaleSetup::late_fate) : le pts
// de chaque instant est comparé à l'horloge prédite juste avant son calcul, plus seulement à l'arrivée de la
// paire. Un instant passé n'est ni calculé ni écrit (Fate::Late) ; la source en retard garde son SPAN quand
// la paire suivante le lit (warp après SPAN), sans être écrite : fini, la source calculée puis jetée.
#include <algorithm>
#include <cstdint>
#include <expected>

#include "app/play/frame_latency.h"
#include "app/play/upscale_stage.h"

namespace ajp::app::play {

bool UpscaleStage::instant_late(const DecodedFrame& frame, std::uint32_t output) const {
  if (setup_.clock == nullptr) return false;
  const Instant& instant = instants_[std::min<std::size_t>(output, instants_.size() - 1)];
  const std::int64_t pts = instant.t >= 1.0F ? frame.pts_us : instant.pts_us;
  if (pts == kNoPts) return false;
  const ClockSnapshot now = setup_.clock->snapshot();
  if (!now.valid || now.epoch != frame.epoch || now.rate <= 0.0) return false;
  return pts < ready_media_us(now, host_now_us(), 0, latency_us());
}

std::expected<void, Refusal> UpscaleStage::late(const DecodedFrame& frame, std::uint32_t output) {
  ++late_dropped_;
  if (setup_.live) setup_.live->auto_unexpect(frame.id.value, 1);
  const Instant& instant = instants_[std::min<std::size_t>(output, instants_.size() - 1)];
  if (instant.t >= 1.0F) {
    // Le warp de la paire suivante lit les deux sources agrandies : le SPAN de celle-ci, sans écriture.
    if (warps_spanned()) {
      if (auto ok = span_current(); !ok) return ok;
    }
    output_has_source_ = false;  // la sortie agrandie n'est plus celle de cette source
    shown_source_ = false;
  }
  // La dernière sortie de la paire : la source reste l'image d'avant de la suivante, écrite ou non.
  if (output + 1 >= outputs_)
    previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
  return {};
}

}  // namespace ajp::app::play
