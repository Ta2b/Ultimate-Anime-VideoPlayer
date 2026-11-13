// L'étage gpu (upscale_stage.h) : SPAN, son tampon repris sur un dessin tenu et l'écriture de
// l'emplacement de l'anneau. Sorti d'upscale_stage.cpp (C09-09).
#include <array>
#include <optional>
#include <utility>

#include "app/play/links.h"
#include "app/play/upscale_stage.h"
#include "core/log.h"
#include "core/trace.h"

namespace ajp::app::play {

bool UpscaleStage::reuses_span(bool held) const {
  if (setup_.engine == nullptr) return false;
  // Smooth, warp après SPAN : chaque copie de la tenue reprend le SPAN de a, quelle que soit sa longueur ;
  // aucune n'est montrée telle quelle, et l'autre tampon tient celui de b (span_smooth_b).
  if (smooth_first_ && warps_spanned()) return true;
  return held && spans_reused_in_row_ < kMaxSpanReuse &&
         (warps_spanned() || output_has_source_ || (setup_.skip_held && shown_source_));
}

void UpscaleStage::pick_span_buffer() {
  if (reuse_span_) ++spans_reused_;
  // Choisi au premier SPAN, le tampon restait celui d'il y a deux images, qui après un dessin tenu est
  // celui de a : le premier instant de la paire mélangeait a à elle-même, warpée par le mouvement de la
  // paire (contours doublés, masque de RIFE en trame).
  span_of_[current_] = reuse_span_ ? span_of_[current_ ^ 1U] : span_of_[current_ ^ 1U] ^ 1U;
}

std::expected<void, Refusal> UpscaleStage::span_current() {
  AJP_TRACE("span");
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
  AJP_TRACE("grade avant le warp");
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

}  // namespace ajp::app::play
