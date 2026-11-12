// Smooth (« Lisser les dessins tenus », --smooth-drawings, chantier 7.9, essai) : un dessin tenu de deux à
// six sources (a, puis ses copies) suivi d'un dessin nouveau b se fond en un seul mouvement de a vers b, au
// lieu de rester figé puis de sauter. Chaque paire de tête de la tenue (une copie, la suivante) interpole a →
// b à u = (k + t) / n, la copie tenue elle-même comprise, à u = (k + 1) / n ; la dernière paire (dernière
// copie, b) reprend à u = (n - 1 + t) / n (rife_plan.h, smooth_hold_u). b est lue en avance : la suivante de
// la première copie pour une tenue de deux sources (ahead_), plus loin sinon (smooth_far_rgb_), l'étage
// attendant kSmoothMaxHold - 1 images derrière la tête (frames_ahead). Une coupe, une image absente ou une
// tenue plus longue gardent le rythme des sources. L'idée vient du --smooth_dedup de TheAnimeScripter ; rien
// de son code.
#include <array>
#include <cstdint>
#include <expected>
#include <utility>

#include "app/play/upscale_stage.h"
#include "kernels/spec/detect.h"

namespace ajp::app::play {

std::expected<void, Refusal> UpscaleStage::allocate_smooth() {
  gpu::Backend& backend = *setup_.backend;
  const auto alloc = [&](std::unique_ptr<gpu::TensorBuffer>& into, int width,
                         int height) -> std::expected<void, Refusal> {
    auto tensor = backend.alloc_tensor(gpu::TensorType::F16, gpu::Shape{.dims = {1, 3, height, width}});
    if (!tensor) return std::unexpected(std::move(tensor.error()));
    into = std::move(*tensor);
    return {};
  };
  if (auto ok = alloc(smooth_far_rgb_, setup_.work_width, setup_.work_height); !ok) return ok;
  if (small_[0]) return alloc(smooth_far_small_, setup_.rife_width, setup_.rife_height);
  return {};
}

void UpscaleStage::smooth_stop() {
  smooth_first_ = false;
  smooth_second_ = false;
  smooth_hold_ = 0;
  smooth_far_ = false;
  // Le tampon du SPAN de b peut recevoir celui d'une copie qui ne reprend plus celui de a : b refait le sien.
  span_ahead_.reset();
}

void UpscaleStage::smooth_advance(const DecodedFrame& frame, const PairFacts& facts, bool on_grid) {
  if (!smooth_first_) {
    if (smooth_second_) smooth_stop();  // la tenue est finie
    return;
  }
  const std::uint32_t rank = smooth_rank_ + 1;
  const bool last = rank + 1 == smooth_hold_;
  // Une paire de tête se fait sur la grille, sans coupe ; la dernière arrive avec b. Sinon (seek, image
  // sautée en retard, interpolation coupée en cours de tenue) la tenue s'arrête : la paire reprend son
  // rythme.
  const bool kept = facts.same_epoch && (last ? frame.id == smooth_b_id_
                                              : frame.id != smooth_b_id_ && on_grid && !pair_cut(facts));
  if (!kept) {
    smooth_stop();
    return;
  }
  smooth_rank_ = rank;
  smooth_first_ = !last;
  smooth_second_ = last;
}

std::expected<void, Refusal> UpscaleStage::smooth_decide(const DecodedFrame& frame,
                                                         const gpu::IngestSpec& spec, const PairFacts& facts,
                                                         bool on_grid) {
  if (!smooth_possible() || !on_grid || !facts.same_epoch || !facts.scores || smooth_first_ || smooth_second_)
    return {};
  if (pair_cut(facts) || !kernels::is_held(facts.scores->worst_block)) return {};
  if (setup_.live && setup_.live->keep_poses()) return {};  // les poses ont leur propre traitement du tenu
  // a doit commencer le dessin : la paire d'avant tenait déjà (tenue trop longue, ou déjà jugée) : le rythme
  // source reste.
  if (now_ == PairMotion::Still) return {};
  if (!looks_ahead() || !ahead_.rgb || !ahead_scores_ || next() == nullptr) return {};
  if (auto ok = look_ahead(frame, spec, true); !ok) return ok;
  if (ahead_.ticket == 0 || !ahead_.id) return {};
  auto scores = setup_.backend->kernels()->scores_take(std::exchange(ahead_.ticket, 0));
  if (!scores) return std::unexpected(std::move(scores.error()));
  ahead_taken_ = std::pair{frame.id, *scores};
  // Une coupe derrière la copie garde le rythme source.
  if (cut_measured(*scores, frame.pts_us, next()->pts_us)) return {};
  std::uint32_t hold = 2;
  if (kernels::is_held(scores->worst_block)) {  // la suivante tient encore : b est plus loin
    auto found = smooth_find_b(frame, spec);
    if (!found) return std::unexpected(std::move(found.error()));
    if (*found == 0) return {};
    hold = *found;
  }
  smooth_first_ = true;
  smooth_rank_ = 0;
  smooth_hold_ = hold;
  smooth_far_ = hold > 2;
  smooth_spanned_ = false;
  smooth_b_id_ = smooth_far_ ? next(hold - 1)->id : *ahead_.id;
  ++smoothed_holds_[hold];
  return {};
}

std::expected<std::uint32_t, Refusal> UpscaleStage::smooth_find_b(const DecodedFrame& frame,
                                                                  const gpu::IngestSpec& spec) {
  if (!smooth_far_rgb_) return 0U;
  gpu::KernelOps& ops = *setup_.backend->kernels();
  // La copie en cours (frame) est a à la mesure près : chaque image de la tenue se compare à elle, et non à
  // la précédente, pour qu'un fondu lent, tenu paire par paire, ne passe pas pour un seul dessin.
  for (std::uint32_t hold = 3; hold <= kSmoothMaxHold; ++hold) {
    const DecodedFrame* candidate = next(hold - 1);
    if (candidate == nullptr || candidate->width != setup_.source_width ||
        candidate->height != setup_.source_height)
      return 0U;
    // La mesure se fait à la taille de RIFE quand elle existe : la taille de travail n'est ingérée que pour
    // le b trouvé, pas pour chaque copie encore tenue (une ingestion pleine taille de moins par candidat).
    if (smooth_far_small_) {
      if (auto ok = ops.ingest(*candidate, gpu::IngestSpec{}, smooth_far_small_->tensor()); !ok)
        return std::unexpected(std::move(ok.error()));
    } else if (auto debanded = ingest_source(*candidate, spec, smooth_far_rgb_->tensor()); !debanded) {
      return std::unexpected(std::move(debanded.error()));
    }
    const gpu::Tensor& far = smooth_far_small_ ? smooth_far_small_->tensor() : smooth_far_rgb_->tensor();
    auto scores = ops.scores_now(rife_source(current_), far);
    if (!scores) return std::unexpected(std::move(scores.error()));
    if (cut_measured(*scores, frame.pts_us, candidate->pts_us)) return 0U;
    if (!kernels::is_held(scores->worst_block)) {  // b : le dessin nouveau
      if (smooth_far_small_) {
        if (auto debanded = ingest_source(*candidate, spec, smooth_far_rgb_->tensor()); !debanded)
          return std::unexpected(std::move(debanded.error()));
      }
      return hold;
    }
  }
  return 0U;  // tenue de plus de kSmoothMaxHold sources : une pose, le rythme source reste
}

std::expected<void, Refusal> UpscaleStage::span_smooth_b() {
  if (smooth_spanned_) return {};
  smooth_spanned_ = true;
  // Le tampon que les copies ne prennent pas (elles reprennent celui de a) : celui où SPAN de b irait de
  // toute façon.
  span_ahead_buf_ = span_of_[current_ ^ 1U] ^ 1U;
  span_ahead_ = smooth_b_id_;
  const gpu::Lane back = lane_;
  use_lane(gpu::Lane::Upscale);
  const std::array inputs{smooth_b_rgb()};
  std::array results{spanned_[span_ahead_buf_]->tensor()};
  auto ok = blame_stage(setup_.backend->enqueue(*setup_.engine, inputs, results), kSpanRefusalKey);
  use_lane(back);
  return ok;
}

}  // namespace ajp::app::play
