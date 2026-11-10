// Smooth (--smooth-drawings, chantier 7.9, essai) : un dessin tenu deux sources (a, a') suivi d'un dessin
// nouveau b se fond en un seul mouvement de a vers b, au lieu de rester figé une période puis de sauter. La
// paire (a, a') interpole a → b à u = t / 2 (la source tenue a' comprise, à u = 1/2) ; la paire (a', b)
// reprend à u = 1/2 + t / 2 (rife_plan.h, smooth_hold_u). b est l'image suivante, déjà là (wants_next) : elle
// est ingérée et mesurée en avance, ses mesures servent à la paire d'après. Sans elle, ou pour une tenue plus
// longue, le rythme des sources reste.
#include <array>
#include <cstdint>
#include <expected>
#include <utility>

#include "app/play/upscale_stage.h"
#include "kernels/spec/detect.h"

namespace ajp::app::play {

std::expected<void, Refusal> UpscaleStage::smooth_decide(const DecodedFrame& frame,
                                                         const gpu::IngestSpec& spec, const PairFacts& facts,
                                                         bool on_grid) {
  if (!smooth_possible() || !on_grid || !facts.same_epoch || !facts.scores || smooth_second_) return {};
  if (pair_cut(facts) || !kernels::is_held(facts.scores->worst_block)) return {};
  if (setup_.live && setup_.live->keep_poses()) return {};  // les poses ont leur propre traitement du tenu
  // a doit commencer le dessin : une tenue plus longue (la paire d'avant tenait déjà) garde le rythme source.
  if (now_ == PairMotion::Still) return {};
  if (!looks_ahead() || !ahead_.rgb || !ahead_scores_ || next() == nullptr) return {};
  // Warp après SPAN : a' reprend le SPAN de a (dessin tenu), le tampon libre reçoit celui de b.
  if (warps_spanned() && !(setup_.engine != nullptr && spans_reused_in_row_ < kMaxSpanReuse)) return {};
  if (auto ok = look_ahead(frame, spec); !ok) return ok;
  if (ahead_.ticket == 0 || !ahead_.id) return {};
  auto scores = setup_.backend->kernels()->scores_take(std::exchange(ahead_.ticket, 0));
  if (!scores) return std::unexpected(std::move(scores.error()));
  ahead_taken_ = std::pair{frame.id, *scores};
  // b doit être un dessin nouveau, sans coupe : une tenue plus longue ou une coupe gardent le rythme source.
  if (cut_measured(*scores, frame.pts_us, next()->pts_us) || kernels::is_held(scores->worst_block)) return {};
  smooth_first_ = true;
  smooth_b_id_ = *ahead_.id;
  ++smoothed_holds_;
  return {};
}

std::expected<void, Refusal> UpscaleStage::span_smooth_b() {
  if (smooth_spanned_) return {};
  smooth_spanned_ = true;
  // Le tampon que a' ne prend pas (a' reprend celui de a) : celui où SPAN de b irait de toute façon.
  span_ahead_buf_ = span_of_[current_ ^ 1U] ^ 1U;
  span_ahead_ = smooth_b_id_;
  const gpu::Lane back = lane_;
  use_lane(gpu::Lane::Upscale);
  const std::array inputs{smooth_b_rgb()};
  std::array results{spanned_[span_ahead_buf_]->tensor()};
  auto ok = setup_.backend->enqueue(*setup_.engine, inputs, results);
  use_lane(back);
  return ok;
}

}  // namespace ajp::app::play
