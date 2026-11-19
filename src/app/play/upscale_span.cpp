// L'étage gpu (upscale_stage.h) : SPAN, son tampon repris sur un dessin tenu et l'écriture de
// l'emplacement de l'anneau. Sorti d'upscale_stage.cpp (C09-09).
#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <utility>

#include "app/play/links.h"
#include "app/play/upscale_stage.h"
#include "core/log.h"
#include "core/trace.h"
#include "kernels/spec/resample.h"

namespace kirari::app::play {

namespace {

// L'écart entre l'ancre et l'image en cours ; nullopt si le backend ne sait pas le mesurer (la reprise
// retombe alors à deux de suite). waited_us : l'attente hôte de la relecture.
std::optional<AnchorGap> measure_anchor(gpu::KernelOps& ops, const gpu::Tensor& anchor,
                                        const gpu::Tensor& now, SpanReuse& reuse, std::int64_t& waited_us) {
  KIRARI_TRACE("span.anchor");
  const auto asked = std::chrono::steady_clock::now();
  auto scores = ops.scores_now(anchor, now);
  waited_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - asked).count();
  if (scores)
    return AnchorGap{.worst_block = scores->worst_block, .mean_levels = scores->scene_score * 255.0};
  log_channel("gpu")->info("SPAN repris : pas de mesure contre l'ancre ({}) ; deux de suite au plus",
                           scores.error().detail);
  reuse.anchor_measure = false;
  return std::nullopt;
}

// Début d'une série : l'ancre (la précédente, telle que SPAN l'a lue) est copiée avant que son tampon ne
// reçoive une autre image ; la copie suit sur la voie en cours, sans attente.
std::expected<void, Refusal> keep_anchor(gpu::Backend& backend, const gpu::Tensor& previous,
                                         SpanReuse& reuse) {
  if (!reuse.anchor) {
    auto tensor = backend.alloc_tensor(previous.type, previous.shape);
    if (!tensor) {
      log_channel("gpu")->info("SPAN repris : ancre refusée ({}) ; deux de suite au plus",
                               tensor.error().detail);
      reuse.anchor_measure = false;
      return {};
    }
    reuse.anchor = std::move(*tensor);
  }
  return backend.kernels()->resample(previous, kernels::Filter::Area, reuse.anchor->tensor());
}

}  // namespace

std::expected<bool, Refusal> UpscaleStage::reuses_span(const DecodedFrame& frame, const PairFacts& facts,
                                                       bool held) {
  if (setup_.engine == nullptr) return false;
  // Smooth, warp après SPAN : chaque copie de la tenue reprend le SPAN de a, quelle que soit sa longueur ;
  // aucune n'est montrée telle quelle, et l'autre tampon tient celui de b (span_smooth_b).
  if (smooth_first_ && warps_spanned()) {
    reuse_.force();
    return true;
  }
  const bool pair_measured = facts.scores.has_value();
  ReuseFacts rf{.same_epoch = facts.same_epoch,
                .corrupt = frame.corrupt,
                .has_output = warps_spanned() || output_has_source_ || (setup_.skip_held && shown_source_),
                .pair_held = !pair_measured || held};
  const bool candidate = rf.same_epoch && !rf.corrupt && rf.has_output && rf.pair_held;
  const int in_row = reuse_.in_row();
  const gpu::Tensor& previous = rgb_[current_ ^ 1U]->tensor();
  // L'ancre est la précédente au début d'une série, sa copie ensuite. Sans mesure de la paire (SPAN seul),
  // l'ancre se mesure dès la première image.
  const bool anchored = in_row == 0 || reuse_.anchor != nullptr;
  if (candidate && reuse_.anchor_measure && reuse_.wants_anchor(pair_measured) && anchored) {
    std::int64_t waited = 0;
    rf.anchor = measure_anchor(*setup_.backend->kernels(), in_row == 0 ? previous : reuse_.anchor->tensor(),
                               rgb_[current_]->tensor(), reuse_, waited);
    keep_scores_wait(waited);
  }
  // Sans mesure du tout (SPAN seul, mesure refusée) : rien ne dit que l'image est tenue.
  if (!pair_measured && !rf.anchor) rf.pair_held = false;
  const bool reused = reuse_.decide(rf) == ReuseKind::Whole;
  if (reused && in_row == 0 && reuse_.anchor_measure) {
    if (auto ok = keep_anchor(*setup_.backend, previous, reuse_); !ok)
      return std::unexpected(std::move(ok.error()));
  }
  return reused;
}

void UpscaleStage::pick_span_buffer() {
  // Choisi au premier SPAN, le tampon restait celui d'il y a deux images, qui après un dessin tenu est
  // celui de a : le premier instant de la paire mélangeait a à elle-même, warpée par le mouvement de la
  // paire (contours doublés, masque de RIFE en trame).
  span_of_[current_] = reuse_span_ ? span_of_[current_ ^ 1U] : span_of_[current_ ^ 1U] ^ 1U;
}

std::expected<void, Refusal> UpscaleStage::span_current() {
  KIRARI_TRACE("span");
  if (spanned_now_) return {};
  spanned_now_ = true;
  // Le tampon est choisi à l'arrivée de l'image : celui de la précédente pour un dessin tenu, sinon
  // l'autre, car celui de la précédente reste le bord a des warps de la paire.
  if (reuse_span_) return {};
  // Smooth : le SPAN de cette image a été fait en avance, dans ce même tampon (span_smooth_b).
  if (std::exchange(span_ahead_, std::nullopt) == frame_id_ && span_ahead_buf_ == span_of_[current_])
    return {};
  // Sur la voie de SPAN, au milieu du travail de la paire : la voie de l'appelant reprend après.
  const gpu::Lane back = lane_;
  use_lane(gpu::Lane::Upscale);
  const std::array inputs{rgb_[current_]->tensor()};
  std::array results{spanned(current_)};
  auto ok = blame_stage(setup_.backend->enqueue(*setup_.engine, inputs, results), kSpanRefusalKey);
  if (ok) ok = grade_spanned(span_of_[current_]);
  use_lane(back);
  return ok;
}

std::expected<void, Refusal> UpscaleStage::grade_spanned(std::uint32_t buf) {
  span_graded_[buf] = false;
  if (!setup_.grader || setup_.yuv_slots) return {};
  KIRARI_TRACE("grade avant le warp");
  auto graded = setup_.grader->grade(*setup_.backend->kernels(), spanned_[buf]->tensor(), debanded_);
  if (!graded) {
    log_channel("gpu")->warn("grade avant le warp : {} {} ; l'écran reprend le grade",
                             name(graded.error().code), graded.error().detail);
    setup_.grader.reset();
    return {};
  }
  span_graded_[buf] = *graded;
  return {};
}

bool UpscaleStage::source_graded() const {
  return warps_spanned() && span_graded_[span_of_[current_]];
}

RingStage::Output UpscaleStage::source_output(std::int64_t pts_us, SourceId id) const {
  return Output{.pts_us = pts_us,
                .left = id,
                .right = id,
                .uninterpolated = uninterpolated_,
                .debanded = debanded_,
                .graded = source_graded()};
}

bool UpscaleStage::pair_graded() const {
  return span_graded_[span_of_[current_ ^ 1U]] &&
         span_graded_[smooth_first_ ? span_ahead_buf_ : span_of_[current_]];
}

std::expected<void, Refusal> UpscaleStage::finish(const gpu::Tensor& rgb, std::uint32_t slot,
                                                  const gpu::RingWrite& sync) {
  if (setup_.engine == nullptr) return write_slot(rgb, slot, sync);
  const std::array inputs{rgb};
  std::array outputs{output_->tensor()};
  // Sans export YUV : SPAN et l'écriture de l'emplacement en un appel (étape 3.4b), que le backend sait
  // fondre (le SPAN FP8 écrit l'anneau depuis son épilogue) ; la sortie est écrite aussi (reuse_span_).
  if (!setup_.yuv_slots) {
    return blame_stage(
        setup_.backend->enqueue_ring(*setup_.engine, inputs, outputs,
                                     gpu::RingTarget{.slot = &setup_.slots[slot], .sync = sync}),
        kSpanRefusalKey);
  }
  if (auto ok = setup_.backend->enqueue(*setup_.engine, inputs, outputs); !ok)
    return blame_stage(std::move(ok), kSpanRefusalKey);
  return write_slot(output_->tensor(), slot, sync);
}

std::expected<void, Refusal> UpscaleStage::write_slot(const gpu::Tensor& image, std::uint32_t slot,
                                                      const gpu::RingWrite& sync) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (setup_.yuv_slots) return ops.write_yuv(image, setup_.yuv_slots->target(slot, sync.signal_value));
  return ops.write_ring(image, setup_.slots[slot], sync);
}

}  // namespace kirari::app::play
