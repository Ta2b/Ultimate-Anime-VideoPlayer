// Le sort de chaque sortie de l'étage gpu (upscale_stage.h, RingStage) : écrite, laissée (déjà à l'écran),
// ou en retard. Les côtés tenus de DistilDRBA et la copie d'une pose après un dessin tenu y décident quelle
// sortie écrit b.
#include <algorithm>
#include <cstdint>

#include "app/play/upscale_stage.h"
#include "kernels/spec/distil_drba.h"

namespace ajp::app::play {

RingStage::Fate UpscaleStage::fate(const DecodedFrame& frame, std::uint32_t output) {
  if (held_same_) return Fate::Same;
  // A09-07 : DistilDRBA aussi laisse un instant déjà passé, avant de juger son côté tenu ; une passe jetée
  // ne s'amortit sur aucun autre instant.
  if (late_fate_now_ && instant_late(frame, output)) return Fate::Late;
  if (setup_.skip_held && drba()) return drba_fate(output);
  if (!copy_pair_) return Fate::Write;
  const Instant& instant = instants_[std::min<std::size_t>(output, instants_.size() - 1)];
  if (poses_copy(true, true, instant.t) == PoseCopy::Before) return Fate::Same;
  if (copy_written_) return output == b_output_ ? Fate::Write : Fate::Same;
  copy_written_ = true;
  b_output_ = output;  // le premier instant après 1/2 : b, à son pts ; les suivants sont déjà à l'écran
  return Fate::Write;
}

void UpscaleStage::skip(const DecodedFrame& frame, std::uint32_t output) {
  if (setup_.skip_held && drba()) {
    if (setup_.live) setup_.live->auto_unexpect(frame.id.value, 1);
    // La source, ou la dernière sortie d'une paire sans source (A09-12, comme late()) : cette image reste
    // l'image d'avant de la suivante.
    if (instants_[std::min<std::size_t>(output, instants_.size() - 1)].t >= 1.0F || output + 1 >= outputs_) {
      previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
    }
    return;
  }
  // La source tenue n'est pas écrite : l'étage la tient pour la dernière vue, comme après une écriture.
  if (!held_same_) return;
  // reuse_span_ : l'agrandissement de la précédente sert à celle-ci, sans calcul (ne peut pas échouer).
  if (auto ok = span_current(); !ok) return;
  previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
}

RingStage::Output UpscaleStage::held_b(Output made, const DecodedFrame& frame, std::uint32_t output) {
  const bool on_b = instants_[std::min<std::size_t>(output, instants_.size() - 1)].t >= 0.5F;
  if (setup_.skip_held && drba() && drba_held_ && on_b) {
    made.left = frame.id;
    made.right = frame.id;
    shown_source_ = true;
  }
  return made;
}

// DistilDRBA : un côté tenu rend l'image du côté sans passe. Avant ½ (côté a tenu), l'écran montre déjà a ;
// dès ½ (côté b tenu), le premier instant écrit b, les suivants et la source sont déjà à l'écran.
RingStage::Fate UpscaleStage::drba_fate(std::uint32_t output) {
  const Instant& instant = instants_[std::min<std::size_t>(output, instants_.size() - 1)];
  if (instant.t >= 1.0F) return b_shown_ && output != b_output_ ? Fate::Same : Fate::Write;
  const kernels::DrbaCall call = kernels::drba_call(instant.t);
  if (call.on_b && !settle_after_scores()) return Fate::Write;  // l'erreur sort à produce
  if (!drba_side_held(call)) return Fate::Write;
  if (!call.on_b) return shown_source_ ? Fate::Same : Fate::Write;
  if (b_shown_) return output == b_output_ ? Fate::Write : Fate::Same;
  b_shown_ = true;
  b_output_ = output;
  return Fate::Write;
}

}  // namespace ajp::app::play
