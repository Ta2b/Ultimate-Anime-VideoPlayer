// RIFE dans l'étage gpu (upscale_stage.h) : l'image de RIFE à chaque instant, ou sa variante
// mouvement, warpée à la sortie (kernels/spec/warp.h). Variante mouvement : une passe au milieu de
// la paire, dont le mouvement maximal dit si elle suffit à toute la paire ; sinon une passe à
// chaque instant. Quand la paire précédente bougeait fort, la passe au milieu est sautée : RIFE
// tourne d'emblée à chaque instant (×3 : deux passes au lieu de trois), et le mouvement maximal du
// premier instant, mis à l'échelle du milieu et relu sans attente à la paire suivante, dit si elle
// reprend la passe au milieu. De même, le mouvement maximal du milieu se relit à la paire suivante
// (NL3) : la paire en cours se décide sur la mesure de la précédente, sans que l'hôte attende le GPU.
// Seule la première paire d'un plan (ou après un saut) attend sa propre mesure.
#include <algorithm>
#include <array>
#include <utility>

#include "app/play/upscale_stage.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/warp.h"

namespace ajp::app::play {

std::expected<void, Refusal> UpscaleStage::interpolate_image(float t) {
  if (auto ok = fill_rife_input(t); !ok) return ok;
  const std::array inputs{rife_in()};
  std::array results{rife_out_->tensor()};
  if (auto ok = setup_.backend->enqueue(*setup_.rife, inputs, results); !ok) return ok;
  return setup_.backend->kernels()->rife_crop(rife_out_->tensor(), middle());
}

std::expected<void, Refusal> UpscaleStage::run_motion(float t, const gpu::Tensor& flow,
                                                      const gpu::Tensor& mask, bool calm) {
  if (auto ok = fill_rife_input(t); !ok) return ok;
  const std::array inputs{rife_in()};
  std::array results{flow, mask};
  ++rife_runs_;
  const bool light = calm && setup_.rife_motion_calm != nullptr;
  if (light) ++calm_runs_;
  gpu::Engine& engine = light ? *setup_.rife_motion_calm : *setup_.rife_motion;
  // P6.15 : la passe précédente était le milieu de la paire d'avant, sur ce moteur, et la paire est calme.
  const bool mid = t == 0.5F;
  if (setup_.rife_warm && mid && calm && warm_from_ == &engine && engine.next_warm()) ++warm_runs_;
  warm_from_ = mid ? &engine : nullptr;
  return setup_.backend->enqueue(engine, inputs, results);
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
  motion_measured_ = false;
  motion_.known = false;  // une mesure encore en vol ne vaut plus
  warm_from_ = nullptr;   // ni les blocs grossiers d'une autre scène (P6.15)
}

std::expected<void, Refusal> UpscaleStage::settle_motion() {
  if (motion_.ticket == 0) return {};
  const std::uint64_t ticket = std::exchange(motion_.ticket, 0);
  auto biggest = setup_.backend->kernels()->flow_max_take(ticket);
  if (!biggest) return std::unexpected(std::move(biggest.error()));
  motion_measured_ = motion_.known;
  if (motion_.known)
    strong_motion_ = *biggest * motion_.scale >= kernels::single_run_motion(setup_.rife_height);
  return {};
}

std::expected<void, Refusal> UpscaleStage::blend(const gpu::Tensor& a, const gpu::Tensor& b,
                                                 const gpu::Motion& motion, float t,
                                                 const Target& out) const {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  return out.ring != nullptr ? ops.warp_blend_ring(a, b, motion, t, *out.ring)
                             : ops.warp_blend(a, b, motion, t, *out.tensor);
}

std::expected<void, Refusal> UpscaleStage::blend_poses(const gpu::Tensor& a, const gpu::Tensor& b,
                                                       const gpu::Motion& motion, const gpu::Motion* previous,
                                                       float t, const Target& out) const {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  return out.ring != nullptr ? ops.warp_blend_poses_ring(a, b, motion, previous, t, *out.ring)
                             : ops.warp_blend_poses(a, b, motion, previous, t, *out.tensor);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::expected<void, Refusal> UpscaleStage::interpolate_by_motion(std::uint32_t output, const gpu::Tensor& a,
                                                                 const gpu::Tensor& b, const Target& out) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const float t = instants_[output].t;
  const gpu::Motion middle = mid_motion(mid_);
  // Mouvement d'un autre instant que le milieu, tel quel (t = 1/2 au warp : « estimé »).
  const gpu::Motion other{.flow = flow_->tensor(), .mask = mask_->tensor(), .area = middle.area};
  if (first_middle()) {
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
      // P6.17 : la paire précédente calme et mesurée, le milieu peut prendre le RIFE léger.
      if (auto ok = run_motion(0.5F, middle.flow, middle.mask, motion_measured_ && !strong_motion_); !ok)
        return ok;
      if (poses_) {
        single_run_ = true;
        motion_measured_ = false;
        // Audit F10 : le mouvement du milieu mesuré en fond, relu à la paire suivante (settle_motion) ; une
        // paire qui suit un mouvement fort a RIFE à chaque instant, et non le milieu extrapolé. Mesuré à
        // chaque paire, une seule image intermédiaire comprise : en Rapide sur un GPU chargé (ou à ×1,5),
        // l'unique instant tombe rarement à 1/2, et sans mesure il gardait le milieu extrapolé en plein
        // mouvement (hachures, contours doublés).
        if (flow_max_async_) {
          if (auto ticket = ops.flow_max_start(middle))
            motion_ = PendingMotion{.ticket = *ticket, .scale = 1.0F, .known = true};
          else
            flow_max_async_ = false;
        }
        if (strong_motion_ && (intermediates_ > 1 || t != 0.5F)) ++multi_pairs_;
      } else if (intermediates_ > 1) {
        // Warp après SPAN : SPAN de b enfilé entre la mesure et sa relecture, qui ne l'attend pas.
        auto ticket = ops.flow_max_start(middle);
        std::expected<float, Refusal> biggest;
        if (ticket && motion_measured_) {
          // La paire précédente, calme, a décidé : celle-ci n'a qu'une passe ; sa mesure décide la suivante.
          motion_ = PendingMotion{.ticket = *ticket, .scale = 1.0F, .known = true};
          biggest = 0.0F;
        } else if (ticket) {
          if (warps_spanned()) {
            if (auto ok = span_current(); !ok) return ok;
          }
          biggest = ops.flow_max_take(*ticket);
        } else {
          flow_max_async_ = false;
          biggest = ops.flow_max_now(middle);
        }
        if (!biggest) return std::unexpected(std::move(biggest.error()));
        motion_measured_ = flow_max_async_;
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
    if (first_middle()) {
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
    return blend(a, b, other, 0.5F, out);
  }
  // Warp après SPAN : b agrandie ici (si la relecture du mouvement maximal ne l'a pas déjà enfilée),
  // après la passe de RIFE et la mesure que l'hôte attend : elles ne passent pas derrière SPAN.
  if (warps_spanned()) {
    if (auto ok = span_current(); !ok) return ok;
  }
  // Garder les poses, mouvement fort : RIFE à l'instant exact (l'atout de RIFE), les poses ne valent que pour
  // les paires calmes, où le milieu extrapolé ne laisse pas de double contour.
  if (poses_ && strong_motion_ && t != 0.5F) {
    if (auto ok = run_motion(t, other.flow, other.mask); !ok) return ok;
    return blend(a, b, other, 0.5F, out);
  }
  if (poses_ && before_ != PairMotion::Unknown) {
    const gpu::Motion previous = mid_motion(mid_ ^ 1U);
    return blend_poses(a, b, middle, before_ == PairMotion::Moving ? &previous : nullptr, t, out);
  }
  if (single_run_ || t == 0.5F) return blend(a, b, middle, t, out);
  // Mouvement fort : RIFE à cet instant, son mouvement tel quel.
  if (auto ok = run_motion(t, other.flow, other.mask); !ok) return ok;
  return blend(a, b, other, 0.5F, out);
}

}  // namespace ajp::app::play
