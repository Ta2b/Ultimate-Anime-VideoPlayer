// Économies de l'étage gpu (upscale_stage.h) d'une paire à l'autre.
//
// Mesures en avance : trois images de travail tournent sans copie, la précédente et l'image en
// cours (rgb_), et la suivante (ahead_) ingérée dès qu'elle attend derrière l'image en cours
// (RingStage::next). Ses mesures contre l'image en cours sont enfilées avant RIFE et SPAN de la
// paire en cours : à son tour, scores_take n'attend qu'elles.
//
// Entrée de RIFE : les images de la paire une fois, sur la voie des mesures et dans l'autre des deux
// entrées, puis l'instant seul pour les autres instants.
#include <utility>

#include "app/play/upscale_stage.h"
#include "core/trace.h"

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
  debanded_ = ahead_.debanded;
  if (ahead_.small) std::swap(small_[current_], ahead_.small);
  return true;
}

std::expected<bool, Refusal> UpscaleStage::ingest_source(const DecodedFrame& frame, gpu::IngestSpec spec,
                                                         const gpu::Tensor& rgb) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  spec.deband = wants_deband() && ops.debands_on_ingest();
  if (auto ok = ops.ingest(frame, spec, rgb); !ok) return std::unexpected(std::move(ok.error()));
  return spec.deband ? std::expected<bool, Refusal>{true} : deband_source(rgb);
}

std::expected<bool, Refusal> UpscaleStage::deband_source(const gpu::Tensor& rgb) {
  if (!wants_deband()) return false;
  auto ok = setup_.backend->kernels()->deband(rgb);
  if (ok) return true;
  if (ok.error().code != RefusalCode::FormatUnsupported) return std::unexpected(std::move(ok.error()));
  deband_op_ = false;  // l'écran débande à la place
  return false;
}

std::expected<gpu::PairScores, Refusal> UpscaleStage::pair_scores() {
  AJP_TRACE("rife.scores");
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (auto taken = std::exchange(ahead_taken_, std::nullopt);
      taken && previous_ && taken->first == previous_->id)
    return taken->second;  // Smooth : relues en avance
  const std::uint64_t ticket = std::exchange(ahead_.ticket, 0);
  if (ticket != 0 && previous_ && ahead_.left == previous_->id) {
    if (auto taken = ops.scores_take(ticket)) return taken;
  }
  return ops.scores_now(rife_source(current_ ^ 1U), rife_source(current_));
}

bool UpscaleStage::next_pair_measured() const {
  if (setup_.live == nullptr) return true;
  if (setup_.live->hidden()) return false;
  if (setup_.live->free_active()) return true;
  const Factor factor = setup_.live->factor_now();
  return factor.num > factor.den;
}

bool UpscaleStage::motion_direct(float t) const {
  if (!by_motion() || !flow_max_async_ || (setup_.live && setup_.live->keep_poses())) return false;
  return (strong_motion_ && intermediates_ > 1) || (intermediates_ == 1 && t != 0.5F);
}

std::expected<void, Refusal> UpscaleStage::look_ahead(const DecodedFrame& frame, const gpu::IngestSpec& spec,
                                                      bool always_scores) {
  const DecodedFrame* after = next();
  if (after == nullptr || after->width != setup_.source_width || after->height != setup_.source_height)
    return {};
  // Déjà ingérée en avance plus tôt dans cette paire (Smooth) : rien à refaire.
  if (ahead_.id == after->id && ahead_.epoch == after->epoch && ahead_.reduce == spec.reduce) return {};
  gpu::KernelOps& ops = *setup_.backend->kernels();
  auto debanded = ingest_source(*after, spec, ahead_.rgb->tensor());
  if (!debanded) return std::unexpected(std::move(debanded.error()));
  ahead_.debanded = *debanded;
  if (ahead_.small) {
    if (auto ok = ops.ingest(*after, gpu::IngestSpec{}, ahead_.small->tensor()); !ok) return ok;
  }
  ahead_.id = after->id;
  ahead_.epoch = after->epoch;
  ahead_.reduce = spec.reduce;
  ahead_.ticket = 0;
  if (!ahead_scores_ || !(always_scores || next_pair_measured())) return {};
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

std::expected<void, Refusal> UpscaleStage::prepare_rife_input() {
  // Le premier instant que RIFE estimera : le milieu pour la variante mouvement, sauf si le
  // mouvement fort de la paire précédente le fait sauter (direct_run_, décidé au premier instant ;
  // rife_time corrige une prévision fausse).
  const bool direct = motion_direct(instants_[0].t);
  const float t = by_motion() && !direct ? 0.5F : instants_[0].t;
  rife_pair_ = (rife_pair_ + 1) % copies_;
  return write_rife_pair(t);
}

std::expected<void, Refusal> UpscaleStage::write_rife_pair(float t) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const gpu::Tensor& a = rife_source(current_ ^ 1U);
  const gpu::Tensor& b = smooth_first_ ? smooth_b_rife() : rife_source(current_);
  std::expected<void, Refusal> ok;
  bool written = false;
  if (rife_mesh_[rife_pair_] && rife_pair_op_) {
    ok = ops.rife_pair(a, b, t, rife_in());
    written = ok || ok.error().code != RefusalCode::FormatUnsupported;
    if (!written) rife_pair_op_ = false;
  }
  if (!written) {
    ok = ops.rife_input(a, b, t, rife_in());
    rife_mesh_[rife_pair_] = ok.has_value();
  }
  // B lissée n'est pas la source current_ : son encodage ne servira pas de A à la paire suivante.
  rife_in_images_[rife_pair_] = {ok ? source_serial_[current_ ^ 1U] : 0U,
                                 ok && !smooth_first_ ? source_serial_[current_] : 0U};
  rife_in_pair_ = ok.has_value();
  rife_in_t_ = t;
  return ok;
}

std::expected<void, Refusal> UpscaleStage::fill_rife_input(float t) {
  AJP_TRACE("rife.pair");
  if (rife_in_pair_ && rife_in_t_ == t) return {};
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (rife_in_pair_ && rife_time_) {
    auto ok = ops.rife_time(rife_in(), t);
    if (ok) rife_in_t_ = t;
    if (ok || ok.error().code != RefusalCode::FormatUnsupported) return ok;
    rife_time_ = false;
  }
  return write_rife_pair(t);
}

void UpscaleStage::count_measured(const PairFacts& facts) {
  if (facts.factor > 1) {  // les compteurs ne bougent que pour une paire qui interpole
    ++pairs_;
    if (setup_.live) setup_.live->count_pair();
  }
  if (pair_cut(facts)) {
    forget_motion();
    ++cuts_;
    if (setup_.live) {
      setup_.live->count_cut();
      setup_.live->auto_cut(host_now_us());
      setup_.live->free_cut(host_now_us());
    }
  } else if (facts.scores && kernels::is_held(facts.scores->worst_block)) {
    ++held_;
    if (setup_.live) setup_.live->count_held();
  }
}

}  // namespace ajp::app::play
