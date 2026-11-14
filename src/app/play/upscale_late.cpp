// Rapide à pas constant, producteur « une image à la fois » (chantier 7.6, UpscaleSetup::late_fate) : le pts
// de chaque instant est comparé à l'horloge prédite juste avant son calcul, plus seulement à l'arrivée de la
// paire. Un instant passé n'est ni calculé ni écrit (Fate::Late) ; la source en retard garde son SPAN quand
// la paire suivante le lit (warp après SPAN), sans être écrite : fini, la source calculée puis jetée.
#include <algorithm>
#include <cstdint>
#include <expected>
#include <utility>

#include "app/play/frame_latency.h"
#include "app/play/upscale_stage.h"

namespace ajp::app::play {
namespace {

// L'écart de pts planifié entre deux instants qui se suivent ; 0 : inconnu (pas d'instant d'avant, pts absent
// ou qui recule).
std::int64_t planned_gap(std::int64_t before_us, std::int64_t pts_us) {
  return before_us != kNoPts && pts_us != kNoPts && pts_us > before_us ? pts_us - before_us : 0;
}

}  // namespace

void UpscaleStage::resync_grids(bool unrun, bool free_mode, bool steady) {
  if (!unrun) return;
  if (cadence_) cadence_->resync();
  if (free_mode) free_grid_.resync();
  if (steady) steady_grid_.resync();
}

// Rapide libre : main compte un instant sauté ou jeté comme manqué, jamais l'écart voulu par la grille (un
// pas lourd de deux vsyncs à 165 Hz, une paire affamée) : chaque sortie porte l'écart depuis l'instant
// planifié d'avant, celui-là sauté en retard compris. Le même parcours que outputs() pour les instants
// gardés.
void UpscaleStage::plan_gaps(const DecodedFrame& frame, const core::CadenceStep* step, std::size_t late,
                             bool on) {
  if (!on) {
    planned_pts_ = kNoPts;  // les instants de la paire, neufs, valent 0 : inconnu
    return;
  }
  const auto next = [this](std::int64_t pts) { return planned_gap(std::exchange(planned_pts_, pts), pts); };
  if (step == nullptr) {
    instants_[0].gap_us = next(frame.pts_us);
    return;
  }
  std::uint32_t kept = 0;
  std::size_t skipped = 0;
  for (const core::OutputInstant& at : step->instants()) {
    const bool source = at.is_source() && !smooth_first_;
    const std::int64_t pts = at.pts_us.value_or(frame.pts_us);  // celui que outputs() donne à l'image
    const std::int64_t gap = next(pts);
    if (!source && skipped < late) {
      ++skipped;
      continue;
    }
    if (kept < outputs_) instants_[kept++].gap_us = gap;
  }
}

std::int64_t UpscaleStage::planned_gap_us(std::uint32_t output) const {
  return instants_[std::min<std::size_t>(output, instants_.size() - 1)].gap_us;
}

bool UpscaleStage::instant_late(const DecodedFrame& frame, std::uint32_t output) const {
  if (setup_.clock == nullptr) return false;
  const Instant& instant = instants_[std::min<std::size_t>(output, instants_.size() - 1)];
  const std::int64_t pts = instant.pts_us != kNoPts ? instant.pts_us : frame.pts_us;
  if (pts == kNoPts) return false;
  const ClockSnapshot now = setup_.clock->snapshot();
  if (!now.valid || now.epoch != frame.epoch || now.rate <= 0.0) return false;
  return pts < ready_media_us(now, host_now_us(), 0, latency_us());
}

// Une paire du Rapide à une image. Réduite à sa source par le pas seul (FreeGrid::starved), elle fait
// redescendre l'estimation du temps de service, que plus rien ne mesure. Une image interpolée et calculée se
// mesure. Toute autre image seule (coupe, dessin tenu, copie d'une pose) ne dit rien.
void UpscaleStage::expect_pair(std::uint32_t id, int written, std::int64_t lead_us, bool free, bool steady,
                               bool runs) {
  LonePair lone = LonePair::Source;
  if (free && !steady && free_grid_.starved()) {
    lone = LonePair::Starved;
  } else if (runs && outputs_ == 1 && intermediates_ == 1 && !copy_pair_) {
    lone = LonePair::Interpolated;
  }
  setup_.live->auto_expect(id, written, lead_us, free && heavy_pair_, lone);
}

std::expected<void, Refusal> UpscaleStage::late(const DecodedFrame& frame, std::uint32_t output) {
  ++late_dropped_;
  if (setup_.live) setup_.live->auto_unexpect(frame.id.value, 1);
  // DistilDRBA, côté b tenu : la sortie chargée d'écrire b a été jugée Write, puis est passée en retard en
  // attendant un slot. b n'est pas à l'écran : la sortie suivante (la source) l'écrit (B09-03). La copie
  // d'une pose (copy_written_) n'a pas ce cas : late_fate_now_ est éteint pour elle.
  if (b_shown_ && output == b_output_) b_shown_ = false;
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
