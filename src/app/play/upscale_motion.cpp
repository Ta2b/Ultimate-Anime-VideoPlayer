// RIFE dans l'étage gpu (upscale_stage.h) : l'image de RIFE à chaque instant, ou sa variante
// mouvement, warpée à la sortie (kernels/spec/warp.h). Variante mouvement : une passe au milieu de
// la paire, dont le mouvement maximal dit si elle suffit à toute la paire ; sinon une passe à
// chaque instant. Quand la paire précédente bougeait fort, la passe au milieu est sautée : RIFE
// tourne d'emblée à chaque instant (×3 : deux passes au lieu de trois), et le mouvement maximal du
// premier instant, mis à l'échelle du milieu et relu sans attente à la paire suivante, dit si elle
// reprend la passe au milieu.
#include <algorithm>
#include <array>
#include <utility>

#include "app/play/upscale_stage.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/warp.h"

namespace ajp::app::play {

std::expected<void, Refusal> UpscaleStage::interpolate_image(float t) {
  if (auto ok = fill_rife_input(t); !ok) return ok;
  const std::array inputs{rife_in_->tensor()};
  std::array results{rife_out_->tensor()};
  if (auto ok = setup_.backend->enqueue(*setup_.rife, inputs, results); !ok) return ok;
  return setup_.backend->kernels()->rife_crop(rife_out_->tensor(), middle_->tensor());
}

std::expected<void, Refusal> UpscaleStage::run_motion(float t, const gpu::Tensor& flow,
                                                      const gpu::Tensor& mask) {
  if (auto ok = fill_rife_input(t); !ok) return ok;
  const std::array inputs{rife_in_->tensor()};
  std::array results{flow, mask};
  ++rife_runs_;
  return setup_.backend->enqueue(*setup_.rife_motion, inputs, results);
}

gpu::Motion UpscaleStage::mid_motion(std::uint32_t index) const {
  const MotionBuffers& mid = mids_[index];
  // RIFE estime à sa taille, dans ses plans remplis ; le warp met le mouvement à l'échelle.
  const kernels::RifeGeometry g =
      kernels::rife_geometry(setup_.rife_width, setup_.rife_height, setup_.rife_align);
  return {.flow = mid.flow->tensor(),
          .mask = mid.mask->tensor(),
          .area = kernels::Window{
              .x = g.left, .y = g.top, .width = setup_.rife_width, .height = setup_.rife_height}};
}

void UpscaleStage::forget_motion() {
  strong_motion_ = false;
  motion_.known = false;  // une mesure encore en vol ne vaut plus
}

std::expected<void, Refusal> UpscaleStage::settle_motion() {
  if (motion_.ticket == 0) return {};
  const std::uint64_t ticket = std::exchange(motion_.ticket, 0);
  auto biggest = setup_.backend->kernels()->flow_max_take(ticket);
  if (!biggest) return std::unexpected(std::move(biggest.error()));
  if (motion_.known)
    strong_motion_ = *biggest * motion_.scale >= kernels::single_run_motion(setup_.rife_height);
  return {};
}

std::expected<void, Refusal> UpscaleStage::interpolate_by_motion(std::uint32_t output, const gpu::Tensor& a,
                                                                 const gpu::Tensor& b,
                                                                 const gpu::Tensor& out) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const float t = instants_[output].t;
  const gpu::Motion middle = mid_motion(mid_);
  // Mouvement d'un autre instant que le milieu, tel quel (t = 1/2 au warp : « estimé »).
  const gpu::Motion other{.flow = flow_->tensor(), .mask = mask_->tensor(), .area = middle.area};
  if (output == 0) {
    poses_ = setup_.live && setup_.live->keep_poses();
    single_run_ = false;
    if (auto ok = settle_motion(); !ok) return ok;
    // Garder les poses demande le mouvement du milieu à chaque instant : une seule passe, au milieu.
    direct_run_ = !poses_ && intermediates_ > 1 && strong_motion_ && flow_max_async_;
    if (direct_run_) {
      ++multi_pairs_;
      ++direct_pairs_;
      // Le milieu n'est pas estimé : la paire suivante ne garde pas les poses d'après lui.
      now_ = PairMotion::Unknown;
    } else {
      // RIFE au milieu ; s'il reste plus d'une image à faire, son mouvement maximal dit si cette
      // passe suffit (kernels/spec/warp.h, seuil mis à la hauteur de RIFE).
      if (auto ok = run_motion(0.5F, middle.flow, middle.mask); !ok) return ok;
      if (poses_) {
        single_run_ = true;
      } else if (intermediates_ > 1) {
        // Warp après SPAN : SPAN de b enfilé entre la mesure et sa relecture, qui ne l'attend pas.
        auto ticket = ops.flow_max_start(middle);
        std::expected<float, Refusal> biggest;
        if (ticket) {
          if (warps_spanned()) {
            if (auto ok = span_current(); !ok) return ok;
          }
          biggest = ops.flow_max_take(*ticket);
        } else {
          flow_max_async_ = false;
          biggest = ops.flow_max_now(middle);
        }
        if (!biggest) return std::unexpected(std::move(biggest.error()));
        single_run_ = *biggest < kernels::single_run_motion(setup_.rife_height);
        strong_motion_ = !single_run_;
        if (single_run_) {
          ++single_pairs_;
        } else {
          ++multi_pairs_;
        }
      }
      now_ = PairMotion::Moving;
    }
  }
  if (direct_run_) {
    if (auto ok = run_motion(t, other.flow, other.mask); !ok) return ok;
    if (output == 0) {
      // Au premier instant s, le plus grand déplacement vaut max(s, 1 − s) de celui de la paire,
      // contre 1/2 au milieu : mis à l'échelle du milieu pour le seuil, relu à la paire suivante.
      auto ticket = ops.flow_max_start(other);
      if (ticket) {
        motion_ = PendingMotion{.ticket = *ticket, .scale = 0.5F / std::max(t, 1.0F - t), .known = true};
      } else {
        flow_max_async_ = false;
      }
    }
    if (warps_spanned()) {
      if (auto ok = span_current(); !ok) return ok;
    }
    return ops.warp_blend(a, b, other, 0.5F, out);
  }
  // Warp après SPAN : b agrandie ici (si la relecture du mouvement maximal ne l'a pas déjà enfilée),
  // après la passe de RIFE et la mesure que l'hôte attend : elles ne passent pas derrière SPAN.
  if (warps_spanned()) {
    if (auto ok = span_current(); !ok) return ok;
  }
  if (poses_ && before_ != PairMotion::Unknown) {
    const gpu::Motion previous = mid_motion(mid_ ^ 1U);
    return ops.warp_blend_poses(a, b, middle, before_ == PairMotion::Moving ? &previous : nullptr, t, out);
  }
  if (single_run_ || t == 0.5F) return ops.warp_blend(a, b, middle, t, out);
  // Mouvement fort : RIFE à cet instant, son mouvement tel quel.
  if (auto ok = run_motion(t, other.flow, other.mask); !ok) return ok;
  return ops.warp_blend(a, b, other, 0.5F, out);
}

}  // namespace ajp::app::play
