// DistilDRBA dans l'étage gpu (upscale_stage.h, kernels/spec/distil_drba.h) : quatre images de
// travail, l'image d'avant la paire (drba_.before), a et b (rgb_), et celle d'après
// (drba_.after), ingérée dès qu'elle arrive derrière b (RingStage::next) et mesurée contre b ; ses
// mesures resservent quand elle devient b. Les tampons tournent sans copie : l'ancienne image
// d'avant reçoit la suivante.
//
// Une image intermédiaire : deux rife_input en 7 plans, (img0, img1) et (img1, img2) à l'instant
// u de drba_call, une passe du graphe, puis rife_crop. Sans l'image d'un côté (début d'époque,
// coupe, fin du flux), le côté manquant répète l'image du centre comme la tête et la fin
// d'infer.py. Un côté tenu (dessin identique) rend l'image de ce côté sans passe du réseau.
#include <array>
#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <utility>

#include "app/play/upscale_stage.h"
#include "core/trace.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/distil_drba.h"
#include "kernels/spec/resample.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {
std::expected<void, Refusal> UpscaleStage::allocate_drba() {
  gpu::Backend& backend = *setup_.backend;
  // Variante mouvement : le mouvement s'estime à la taille de RIFE (D4 : 720p pour un travail plus grand),
  // sur des copies réduites des quatre images, et le warp déforme les images de travail.
  const kernels::RifeGeometry g = drba_geometry();
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
  if (small_[0]) {
    for (auto* frame : {&drba_.small_before, &drba_.small_after}) {
      if (auto ok = alloc(*frame, 3, setup_.rife_width, setup_.rife_height); !ok) return ok;
    }
  }
  for (auto* input : {&drba_.pair01, &drba_.pair12}) {
    if (auto ok = alloc(*input, kernels::kDistilDrbaInputPlanes, g.padded_width, g.padded_height); !ok)
      return ok;
  }
  const std::span<const std::string> names = setup_.distil_drba->input_names();
  if (names.size() != 2 || names[0] != "pair01" || names[1] != "pair12") {
    return refuse(RefusalCode::InputMismatch, "DistilDRBA : entrées « pair01 » et « pair12 » attendues");
  }
  if (setup_.drba_motion) {
    if (auto ok = alloc(drba_.flow, 4, g.padded_width, g.padded_height); !ok) return ok;
    return alloc(drba_.mask, 1, g.padded_width, g.padded_height);
  }
  return alloc(rife_out_, 3, g.padded_width, g.padded_height);
}

std::expected<void, Refusal> UpscaleStage::drba_arrive(const DecodedFrame& frame,
                                                       const gpu::IngestSpec& spec) {
  // rgb_[current_] tient l'image d'avant a : elle passe dans before, qui rend son tampon.
  std::swap(drba_.before, rgb_[current_]);
  if (small_[current_]) std::swap(drba_.small_before, small_[current_]);
  drba_.small_taken = false;
  drba_.filled01 = {};
  drba_.filled12 = {};
  drba_.before_ok = drba_.continuous && previous_ && previous_->epoch == frame.epoch;
  drba_.continuous = false;
  if (drba_.bridge_ticket) {
    // L'image d'avant est l'image sautée : gardée si rien ne la coupe de l'image de reprise.
    const auto ticket = drba_.bridge_ticket.value();
    drba_.bridge_ticket.reset();
    auto bridged = setup_.backend->kernels()->scores_take(ticket);
    if (!bridged) return std::unexpected(std::move(bridged.error()));
    drba_.before_ok = previous_ && previous_->epoch == frame.epoch &&
                      !cut_measured(*bridged, drba_.bridge_left_pts_us, drba_.bridge_right_pts_us);
  }
  // b est déjà ingérée si elle est arrivée derrière l'image précédente, avec la réduction d'aujourd'hui : une
  // réduction changée au menu entre-temps la fait ingérer de nouveau, comme take_ahead (C09-07).
  if (drba_.after_id == frame.id && drba_.after_epoch == frame.epoch && drba_.after_reduce == spec.reduce) {
    std::swap(rgb_[current_], drba_.after);
    if (small_[current_]) {
      std::swap(small_[current_], drba_.small_after);
      drba_.small_taken = true;
    }
    drba_.after_id.reset();
    debanded_ = drba_.after_debanded;
    return {};
  }
  drba_.after_id.reset();
  drba_.after_scores.reset();
  drba_.after_ticket.reset();
  auto debanded = ingest_source(frame, spec, rgb_[current_]->tensor());
  if (!debanded) return std::unexpected(std::move(debanded.error()));
  debanded_ = *debanded;
  return {};
}

void UpscaleStage::keep_scores_wait(std::int64_t waited_us) {
  scores_waits_.keep(waited_us);
}

std::expected<void, Refusal> UpscaleStage::settle_after_scores(bool timed) {
  if (!drba_.after_ticket) return {};
  const std::uint64_t ticket = *drba_.after_ticket;
  drba_.after_ticket.reset();
  // A09-19 : dès ×2 le ticket est relu ici, au premier instant du côté b, et non dans outputs() : sans ce
  // chronomètre le profil de fin donnait une attente nulle à DistilDRBA.
  const auto asked = std::chrono::steady_clock::now();
  auto scores = setup_.backend->kernels()->scores_take(ticket);
  if (timed)
    keep_scores_wait(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - asked)
            .count());
  if (!scores) return std::unexpected(std::move(scores.error()));
  scores->left = drba_.ticket_left;
  scores->right = drba_.ticket_right;
  drba_.after_scores = *scores;
  drba_.after_ok = !cut_measured(*scores, drba_.left_pts_us, drba_.right_pts_us);
  drba_.after_held = drba_.after_ok && kernels::is_held(scores->worst_block);
  return {};
}

bool UpscaleStage::cut_measured(const gpu::PairScores& scores, std::int64_t left_pts_us,
                                std::int64_t right_pts_us) const {
  PairFacts facts{.scores = scores, .useful_area = useful_area_};
  if (setup_.live && left_pts_us != kNoPts && right_pts_us != kNoPts) {
    facts.known_cut = setup_.live->cuts().cut_between(left_pts_us, right_pts_us);
    facts.cut_burst = cut_burst_.after_cut(left_pts_us);
  }
  return pair_cut(facts);
}

std::expected<gpu::PairScores, Refusal> UpscaleStage::drba_scores(const DecodedFrame& frame) {
  // Sans chronomètre : outputs() mesure déjà tout cet appel.
  if (auto ok = settle_after_scores(false); !ok) return std::unexpected(std::move(ok.error()));
  if (drba_.after_scores && drba_.after_scores->right == frame.id) {
    const gpu::PairScores kept = *drba_.after_scores;
    drba_.after_scores.reset();
    return kept;
  }
  drba_.after_scores.reset();
  return setup_.backend->kernels()->scores_now(rife_source(current_ ^ 1U), rife_source(current_));
}

kernels::RifeGeometry UpscaleStage::drba_geometry() const {
  return kernels::rife_geometry(setup_.rife_width, setup_.rife_height, setup_.drba_align);
}

std::expected<void, Refusal> UpscaleStage::drba_next(const DecodedFrame& frame, const gpu::IngestSpec& spec,
                                                     bool measure) {
  AJP_TRACE("drba.ingest");
  drba_.after_ok = false;
  drba_.after_held = false;
  drba_.after_ticket.reset();
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (std::exchange(drba_.bridge, false)) {
    // Reprise après une image sautée : elle (rgb_ de l'autre côté) contre celle-ci, relue à la suivante.
    if (auto ticket = ops.scores_start(rife_source(current_ ^ 1U), rife_source(current_))) {
      drba_.bridge_ticket = *ticket;
      drba_.bridge_right_pts_us = frame.pts_us;
    }
  }
  const DecodedFrame* after = next();
  // Une image d'une autre taille n'est pas ingérée (A09-04, comme look_ahead) : elle arrive ensuite comme
  // image courante, et son refus SourceResized rouvre la chaîne.
  if (after == nullptr || after->width != setup_.source_width || after->height != setup_.source_height)
    return {};
  auto debanded = ingest_source(*after, spec, drba_.after->tensor());
  if (!debanded) return std::unexpected(std::move(debanded.error()));
  drba_.after_debanded = *debanded;
  if (drba_.small_after) {
    if (auto ok = ops.ingest(*after, gpu::IngestSpec{}, drba_.small_after->tensor()); !ok) return ok;
  }
  drba_.after_id = after->id;
  drba_.after_epoch = after->epoch;
  drba_.after_reduce = spec.reduce;
  drba_.left_pts_us = frame.pts_us;
  drba_.right_pts_us = after->pts_us;
  if (!measure) return {};
  // A09-18 : la mesure se fait sur les copies réduites quand elles existent, comme sous RIFE : le seuil
  // des dessins tenus est réglé à cette taille, et la paire y coûte moins de pixels lus.
  const gpu::Tensor& measured_after = drba_.small_after ? drba_.small_after->tensor() : drba_.after->tensor();
  if (auto ticket = ops.scores_start(rife_source(current_), measured_after)) {
    drba_.after_ticket = *ticket;
    drba_.ticket_left = frame.id;
    drba_.ticket_right = after->id;
    return {};
  }
  auto scores = ops.scores_now(rife_source(current_), measured_after);
  if (!scores) return std::unexpected(std::move(scores.error()));
  scores->left = frame.id;
  scores->right = after->id;
  drba_.after_scores = *scores;
  drba_.after_ok = !cut_measured(*scores, drba_.left_pts_us, drba_.right_pts_us);
  drba_.after_held = drba_.after_ok && kernels::is_held(scores->worst_block);
  return {};
}

void UpscaleStage::keep_skipped(const DecodedFrame& frame) {
  drba_.bridge = false;
  drba_.bridge_ticket.reset();
  if (!drba() || drba_.after_id != frame.id || drba_.after_epoch != frame.epoch) return;
  // Sa mesure contre l'image d'avant ne sert plus : relue (le GPU l'a finie) pour ne pas la laisser en vol.
  if (drba_.after_ticket) {
    const auto ticket = drba_.after_ticket.value();
    drba_.after_ticket.reset();
    [[maybe_unused]] const auto dropped = setup_.backend->kernels()->scores_take(ticket);
  }
  drba_.after_scores.reset();
  drba_.after_id.reset();
  // L'image sautée prend la place de la dernière calculée, qui ne sert plus : à la reprise, elle passe
  // dans before comme l'image d'avant la paire.
  std::swap(rgb_[current_], drba_.after);
  if (small_[current_]) std::swap(small_[current_], drba_.small_after);
  source_serial_[current_] = ++serials_;
  drba_.bridge = true;
  drba_.bridge_left_pts_us = frame.pts_us;
}

std::expected<void, Refusal> UpscaleStage::run_drba(const kernels::DrbaCall& call,
                                                    std::span<gpu::Tensor> results) {
  AJP_TRACE("drba.pair");
  gpu::KernelOps& ops = *setup_.backend->kernels();
  // Les copies réduites quand le mouvement s'estime plus petit que le travail (variante mouvement).
  const bool small = small_[0] != nullptr;
  const gpu::Tensor& a = rife_source(current_ ^ 1U);
  const gpu::Tensor& b = rife_source(current_);
  const gpu::Tensor& before = small ? drba_.small_before->tensor() : drba_.before->tensor();
  const gpu::Tensor& after = small ? drba_.small_after->tensor() : drba_.after->tensor();
  const bool whole = call.on_b ? drba_.after_ok : drba_.before_ok;
  const gpu::Tensor& img0 = call.on_b ? a : (whole ? before : a);
  const gpu::Tensor& img1 = call.on_b ? b : a;
  const gpu::Tensor& img2 = call.on_b ? (whole ? after : b) : b;
  // Rend vrai quand seul le plan du temps a été réécrit.
  const auto fill = [&](gpu::TensorBuffer& input, DrbaBuffers::Filled& filled, const gpu::Tensor& first,
                        const gpu::Tensor& second) -> std::expected<bool, Refusal> {
    if (drba_.time_only && filled.first == &first && filled.second == &second) {
      if (ops.rife_time(input.tensor(), call.u)) return true;
      drba_.time_only = false;
    }
    filled = {};
    if (auto ok = ops.rife_input(first, second, call.u, input.tensor()); !ok)
      return std::unexpected(ok.error());
    filled = {.first = &first, .second = &second};
    return false;
  };
  auto times01 = fill(*drba_.pair01, drba_.filled01, img0, img1);
  if (!times01) return std::unexpected(std::move(times01.error()));
  auto times12 = fill(*drba_.pair12, drba_.filled12, img1, img2);
  if (!times12) return std::unexpected(std::move(times12.error()));
  // D6 : les mêmes images, un autre instant : le moteur fusionné garde leurs réductions et leurs encodages.
  if (*times01 && *times12) setup_.distil_drba->next_times_only();
  const std::array inputs{drba_.pair01->tensor(), drba_.pair12->tensor()};
  if (auto ok = setup_.backend->enqueue(*setup_.distil_drba, inputs, results); !ok)
    return blame_stage(std::move(ok), kMotionRefusalKey);
  ++drba_.runs;
  if (!whole) ++drba_.edges;
  return {};
}

// Côté tenu : (avant, a) sans mouvement, DistilDRBA garde a jusqu'à ½ ; (b, après) sans mouvement, il rend b
// dès ½. C'est ce que la passe calculerait : l'image se copie, sans passe (anime en deux, en trois).
bool UpscaleStage::drba_side_held(const kernels::DrbaCall& call) const {
  return call.on_b ? drba_.after_held : drba_.before_ok && before_ == PairMotion::Still;
}

std::expected<void, Refusal> UpscaleStage::interpolate_drba(std::uint32_t output) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const kernels::DrbaCall call = kernels::drba_call(instants_[output].t);
  if (call.on_b) {
    if (auto ok = settle_after_scores(); !ok) return ok;
  }
  if (drba_side_held(call)) {
    ++drba_.held;
    drba_held_ = true;
    return ops.resample(rgb_[call.on_b ? current_ : current_ ^ 1U]->tensor(), kernels::Filter::Area,
                        middle());
  }
  std::array results{rife_out_->tensor()};
  if (auto ok = run_drba(call, results); !ok) return ok;
  return ops.rife_crop(rife_out_->tensor(), middle());
}

std::expected<void, Refusal> UpscaleStage::copy_out(const gpu::Tensor& image, const Target& out) const {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (out.ring != nullptr) return ops.write_ring(image, *out.ring->slot, out.ring->sync);
  return ops.resample(image, kernels::Filter::Area, *out.tensor);
}

std::expected<void, Refusal> UpscaleStage::interpolate_drba_motion(std::uint32_t output, bool spanned_images,
                                                                   const Target& out) {
  const kernels::DrbaCall call = kernels::drba_call(instants_[output].t);
  if (call.on_b) {
    if (auto ok = settle_after_scores(); !ok) return ok;
  }
  const bool held = drba_side_held(call);
  if (!held) {
    std::array results{drba_.flow->tensor(), drba_.mask->tensor()};
    if (auto ok = run_drba(call, results); !ok) return ok;
  }
  // SPAN de b après la passe : elle ne l'attend pas sur le GPU. Les images se lisent ensuite (span_current
  // choisit le tampon de b).
  if (spanned_images) {
    if (auto ok = span_current(); !ok) return ok;
  }
  const gpu::Tensor& a = spanned_images ? spanned(current_ ^ 1U) : rgb_[current_ ^ 1U]->tensor();
  const gpu::Tensor& b = spanned_images ? spanned(current_) : rgb_[current_]->tensor();
  if (held) {
    ++drba_.held;
    drba_held_ = true;
    return copy_out(call.on_b ? b : a, out);
  }
  // Le warp final du graphe lit l'image du centre (flot 0-1, poids du masque) et celle du côté choisi
  // (flot 2-3) : a puis b avant ½, b puis a à partir de ½. Le mouvement est celui de l'instant, tel quel.
  // Estimé à la taille de RIFE dans ses plans remplis : le warp met le mouvement à l'échelle du travail.
  const kernels::RifeGeometry g = drba_geometry();
  const gpu::Motion motion{
      .flow = drba_.flow->tensor(),
      .mask = drba_.mask->tensor(),
      .area =
          kernels::Window{.x = g.left, .y = g.top, .width = setup_.rife_width, .height = setup_.rife_height}};
  return call.on_b ? blend(b, a, motion, 0.5F, out) : blend(a, b, motion, 0.5F, out);
}

}  // namespace ajp::app::play
