// DistilDRBA dans l'étage gpu (upscale_stage.h, kernels/spec/distil_drba.h) : quatre images de
// travail, l'image d'avant la paire (drba_.before), a et b (rgb_), et celle d'après
// (drba_.after), ingérée dès qu'elle arrive derrière b (RingStage::next) et mesurée contre b ; ses
// mesures resservent quand elle devient b. Les tampons tournent sans copie : l'ancienne image
// d'avant reçoit la suivante.
//
// Une image intermédiaire : deux rife_input en 7 plans, (img0, img1) et (img1, img2) à l'instant
// u de drba_call, une passe du graphe, puis rife_crop. Sans l'image d'un côté (début d'époque,
// coupe, fin du flux), le côté manquant répète l'image du centre comme la tête et la fin
// d'infer.py.
#include <array>
#include <span>
#include <string>
#include <utility>

#include "app/play/upscale_stage.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/distil_drba.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

}  // namespace

std::expected<void, Refusal> UpscaleStage::allocate_drba() {
  gpu::Backend& backend = *setup_.backend;
  const kernels::RifeGeometry g =
      kernels::rife_geometry(setup_.work_width, setup_.work_height, setup_.drba_align);
  const auto alloc = [&](std::unique_ptr<gpu::TensorBuffer>& into, int channels, int width,
                         int height) -> std::expected<void, Refusal> {
    auto tensor =
        backend.alloc_tensor(gpu::TensorType::F16, gpu::Shape{.dims = {1, channels, height, width}});
    if (!tensor) return std::unexpected(std::move(tensor.error()));
    into = std::move(*tensor);
    return {};
  };
  for (auto* frame : {&drba_.before, &drba_.after}) {
    if (auto ok = alloc(*frame, 3, setup_.work_width, setup_.work_height); !ok) return ok;
  }
  for (auto* input : {&drba_.pair01, &drba_.pair12}) {
    if (auto ok = alloc(*input, kernels::kDistilDrbaInputPlanes, g.padded_width, g.padded_height); !ok)
      return ok;
  }
  const std::span<const std::string> names = setup_.distil_drba->input_names();
  if (names.size() != 2 || names[0] != "pair01" || names[1] != "pair12") {
    return refuse(RefusalCode::InputMismatch, "DistilDRBA : entrées « pair01 » et « pair12 » attendues");
  }
  return alloc(rife_out_, 3, g.padded_width, g.padded_height);
}

std::expected<void, Refusal> UpscaleStage::drba_arrive(const DecodedFrame& frame,
                                                       const gpu::IngestSpec& spec) {
  // rgb_[current_] tient l'image d'avant a : elle passe dans before, qui rend son tampon.
  std::swap(drba_.before, rgb_[current_]);
  drba_.filled01 = {};
  drba_.filled12 = {};
  drba_.before_ok = drba_.continuous && previous_ && previous_->epoch == frame.epoch;
  drba_.continuous = false;
  // b est déjà ingérée si elle est arrivée derrière l'image précédente.
  if (drba_.after_id == frame.id && drba_.after_epoch == frame.epoch) {
    std::swap(rgb_[current_], drba_.after);
    drba_.after_id.reset();
    return {};
  }
  drba_.after_id.reset();
  drba_.after_scores.reset();
  drba_.after_ticket.reset();
  return setup_.backend->kernels()->ingest(frame, spec, rgb_[current_]->tensor());
}

std::expected<void, Refusal> UpscaleStage::settle_after_scores() {
  if (!drba_.after_ticket) return {};
  const std::uint64_t ticket = *drba_.after_ticket;
  drba_.after_ticket.reset();
  auto scores = setup_.backend->kernels()->scores_take(ticket);
  if (!scores) return std::unexpected(std::move(scores.error()));
  scores->left = drba_.ticket_left;
  scores->right = drba_.ticket_right;
  drba_.after_scores = *scores;
  drba_.after_ok = !kernels::is_scene_cut(scores->scene_score);
  return {};
}

std::expected<gpu::PairScores, Refusal> UpscaleStage::drba_scores(const DecodedFrame& frame) {
  if (auto ok = settle_after_scores(); !ok) return std::unexpected(std::move(ok.error()));
  if (drba_.after_scores && drba_.after_scores->right == frame.id) {
    const gpu::PairScores kept = *drba_.after_scores;
    drba_.after_scores.reset();
    return kept;
  }
  drba_.after_scores.reset();
  return setup_.backend->kernels()->scores_now(rgb_[current_ ^ 1U]->tensor(), rgb_[current_]->tensor());
}

std::expected<void, Refusal> UpscaleStage::drba_next(const DecodedFrame& frame, const gpu::IngestSpec& spec,
                                                     bool measure) {
  drba_.after_ok = false;
  drba_.after_ticket.reset();
  const DecodedFrame* after = next();
  if (after == nullptr) return {};
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (auto ok = ops.ingest(*after, spec, drba_.after->tensor()); !ok) return ok;
  drba_.after_id = after->id;
  drba_.after_epoch = after->epoch;
  if (!measure) return {};
  if (auto ticket = ops.scores_start(rgb_[current_]->tensor(), drba_.after->tensor())) {
    drba_.after_ticket = *ticket;
    drba_.ticket_left = frame.id;
    drba_.ticket_right = after->id;
    return {};
  }
  auto scores = ops.scores_now(rgb_[current_]->tensor(), drba_.after->tensor());
  if (!scores) return std::unexpected(std::move(scores.error()));
  scores->left = frame.id;
  scores->right = after->id;
  drba_.after_scores = *scores;
  drba_.after_ok = !kernels::is_scene_cut(scores->scene_score);
  return {};
}

std::expected<void, Refusal> UpscaleStage::interpolate_drba(std::uint32_t output) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const kernels::DrbaCall call = kernels::drba_call(instants_[output].t);
  if (call.on_b) {
    if (auto ok = settle_after_scores(); !ok) return ok;
  }
  const gpu::Tensor& a = rgb_[current_ ^ 1U]->tensor();
  const gpu::Tensor& b = rgb_[current_]->tensor();
  const bool whole = call.on_b ? drba_.after_ok : drba_.before_ok;
  const gpu::Tensor& img0 = call.on_b ? a : (whole ? drba_.before->tensor() : a);
  const gpu::Tensor& img1 = call.on_b ? b : a;
  const gpu::Tensor& img2 = call.on_b ? (whole ? drba_.after->tensor() : b) : b;
  const auto fill = [&](gpu::TensorBuffer& input, DrbaBuffers::Filled& filled, const gpu::Tensor& first,
                        const gpu::Tensor& second) -> std::expected<void, Refusal> {
    if (drba_.time_only && filled.first == &first && filled.second == &second) {
      if (ops.rife_time(input.tensor(), call.u)) return {};
      drba_.time_only = false;
    }
    filled = {};
    if (auto ok = ops.rife_input(first, second, call.u, input.tensor()); !ok) return ok;
    filled = {.first = &first, .second = &second};
    return {};
  };
  if (auto ok = fill(*drba_.pair01, drba_.filled01, img0, img1); !ok) return ok;
  if (auto ok = fill(*drba_.pair12, drba_.filled12, img1, img2); !ok) return ok;
  const std::array inputs{drba_.pair01->tensor(), drba_.pair12->tensor()};
  std::array results{rife_out_->tensor()};
  if (auto ok = setup_.backend->enqueue(*setup_.distil_drba, inputs, results); !ok) return ok;
  ++drba_.runs;
  if (!whole) ++drba_.edges;
  return ops.rife_crop(rife_out_->tensor(), middle());
}

}  // namespace ajp::app::play
