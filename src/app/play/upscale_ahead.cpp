// Économies de l'étage gpu (upscale_stage.h) d'une paire à l'autre.
//
// Mesures en avance : trois images de travail tournent sans copie, la précédente et l'image en
// cours (rgb_), et la suivante (ahead_) ingérée dès qu'elle attend derrière l'image en cours
// (RingStage::next). Ses mesures contre l'image en cours sont enfilées avant RIFE et SPAN de la
// paire en cours : à son tour, scores_take n'attend qu'elles.
//
// Entrée de RIFE : les images de la paire une fois, puis l'instant seul pour les autres instants.
#include <utility>

#include "app/play/upscale_stage.h"

namespace ajp::app::play {

std::expected<void, Refusal> UpscaleStage::allocate_ahead() {
  gpu::Backend& backend = *setup_.backend;
  const auto alloc = [&](std::unique_ptr<gpu::TensorBuffer>& into, int width,
                         int height) -> std::expected<void, Refusal> {
    auto tensor = backend.alloc_tensor(gpu::TensorType::F16, gpu::Shape{.dims = {1, 3, height, width}});
    if (!tensor) return std::unexpected(std::move(tensor.error()));
    into = std::move(*tensor);
    return {};
  };
  if (auto ok = alloc(ahead_.rgb, setup_.work_width, setup_.work_height); !ok) return ok;
  if (small_[0]) return alloc(ahead_.small, setup_.rife_width, setup_.rife_height);
  return {};
}

bool UpscaleStage::take_ahead(const DecodedFrame& frame, const gpu::IngestSpec& spec) {
  const bool ready = ahead_.id == frame.id && ahead_.epoch == frame.epoch && ahead_.reduce == spec.reduce;
  ahead_.id.reset();
  if (!ready) {
    ahead_.ticket = 0;
    return false;
  }
  // L'ancien tampon de rgb_[current_] (l'image d'avant la précédente) recevra la suivante.
  std::swap(rgb_[current_], ahead_.rgb);
  if (ahead_.small) std::swap(small_[current_], ahead_.small);
  return true;
}

std::expected<gpu::PairScores, Refusal> UpscaleStage::pair_scores() {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const std::uint64_t ticket = std::exchange(ahead_.ticket, 0);
  if (ticket != 0 && previous_ && ahead_.left == previous_->id) {
    if (auto taken = ops.scores_take(ticket)) return taken;
  }
  return ops.scores_now(rife_source(current_ ^ 1U), rife_source(current_));
}

std::expected<void, Refusal> UpscaleStage::look_ahead(const DecodedFrame& frame,
                                                      const gpu::IngestSpec& spec) {
  const DecodedFrame* after = next();
  if (after == nullptr || after->width != setup_.source_width || after->height != setup_.source_height)
    return {};
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (auto ok = ops.ingest(*after, spec, ahead_.rgb->tensor()); !ok) return ok;
  if (ahead_.small) {
    if (auto ok = ops.ingest(*after, gpu::IngestSpec{}, ahead_.small->tensor()); !ok) return ok;
  }
  ahead_.id = after->id;
  ahead_.epoch = after->epoch;
  ahead_.reduce = spec.reduce;
  ahead_.ticket = 0;
  if (!ahead_scores_) return {};
  const gpu::Tensor& next_source = ahead_.small ? ahead_.small->tensor() : ahead_.rgb->tensor();
  auto ticket = ops.scores_start(rife_source(current_), next_source);
  if (!ticket) {
    if (ticket.error().code != RefusalCode::FormatUnsupported)
      return std::unexpected(std::move(ticket.error()));
    ahead_scores_ = false;  // l'ingestion en avance sert encore
    return {};
  }
  ahead_.ticket = *ticket;
  ahead_.left = frame.id;
  return {};
}

std::expected<void, Refusal> UpscaleStage::fill_rife_input(float t) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (rife_in_pair_ && rife_time_) {
    auto ok = ops.rife_time(rife_in_->tensor(), t);
    if (ok || ok.error().code != RefusalCode::FormatUnsupported) return ok;
    rife_time_ = false;
  }
  auto ok = ops.rife_input(rife_source(current_ ^ 1U), rife_source(current_), t, rife_in_->tensor());
  rife_in_pair_ = ok.has_value();
  return ok;
}

}  // namespace ajp::app::play
