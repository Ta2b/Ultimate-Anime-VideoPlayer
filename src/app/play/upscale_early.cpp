// Passes à l'arrivée de la paire dans l'étage gpu (upscale_stage.h) : RIFE mouvement décide la paire et
// enfile toutes ses passes dès outputs(), chacune dans les tampons de sa sortie, au lieu d'une passe par slot
// que l'écran rend. À ×8 l'anneau n'a que huit places : les passes d'une paire forte partaient au rythme de
// l'affichage et le GPU attendait entre elles (39 s d'attente d'un slot sur 40 s en SPAN ×2 4K avec RIFE).
// Seuls les warps, qui écrivent dans le slot, l'attendent encore.
//
// Le même calcul, dans le même ordre sur la voie d'interpolation : même décision de la paire, mêmes passes,
// mêmes entrées ; seuls les tampons de sortie des passes changent, un par sortie. Pas en Rapide à pas
// constant (late_fate : chaque instant se juge juste avant son calcul), ni pour la copie d'une pose, ni avec
// --rife-at-slot (mesure ABAB). Une allocation refusée ramène les passes au slot pour toute la lecture.
#include <algorithm>
#include <array>
#include <optional>
#include <utility>

#include "app/play/upscale_stage.h"
#include "core/log.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {

std::optional<gpu::Motion> UpscaleStage::early_motion(std::uint32_t output, bool half) {
  const bool second = half && halves();
  const int w = second ? pair_scale_side(setup_.rife_width) : setup_.rife_width;
  const int h = second ? pair_scale_side(setup_.rife_height) : setup_.rife_height;
  const kernels::RifeGeometry g = kernels::rife_geometry(w, h, setup_.rife_align);
  MotionBuffers& buffers = second ? early_.half[output] : early_.full[output];
  const auto alloc = [&](std::unique_ptr<gpu::TensorBuffer>& into, int channels) {
    if (into) return true;
    auto tensor = setup_.backend->alloc_tensor(
        gpu::TensorType::F16, gpu::Shape{.dims = {1, channels, g.padded_height, g.padded_width}});
    if (!tensor) {
      log_channel("gpu")->warn("RIFE : tampons des passes à l'arrivée refusés ({}), passes au slot",
                               tensor.error().detail);
      return false;
    }
    into = std::move(*tensor);
    return true;
  };
  if (!alloc(buffers.flow, 4) || !alloc(buffers.mask, 1)) return std::nullopt;
  return gpu::Motion{.flow = buffers.flow->tensor(),
                     .mask = buffers.mask->tensor(),
                     .area = kernels::Window{.x = g.left, .y = g.top, .width = w, .height = h}};
}

std::expected<void, Refusal> UpscaleStage::run_early() {
  early_.ready = false;
  early_.runs.fill(std::nullopt);
  if (setup_.rife_at_slot || early_.refused || !by_motion() || drba() || intermediates_ == 0 || copy_pair_ ||
      late_fate_now_ || !previous_ || outputs_ > early_.runs.size())
    return {};
  std::uint32_t first = 0;
  while (first < outputs_ && instants_[first].t >= 1.0F) ++first;
  if (first == outputs_) return {};
  const gpu::Lane back = lane_;
  use_lane(gpu::Lane::Interpolate);
  if (auto ok = begin_motion_pair(instants_[first].t); !ok) return ok;
  early_.ready = true;
  bool queued = false;
  for (std::uint32_t k = first; k < outputs_ && !early_.refused; ++k) {
    const float t = instants_[k].t;
    if (t >= 1.0F) continue;
    const std::optional<bool> half = strong_half(t);
    if (!half) continue;
    // Le second RIFE (échelle par paire) écrit dans ses propres tampons, plus petits ; run_strong ignore
    // alors ceux de la pleine taille.
    const std::optional<gpu::Motion> into = early_motion(k, *half);
    if (!into) {
      early_.refused = true;  // les sorties sans passe la font au slot (strong_for)
      break;
    }
    auto run = run_strong(t, *into, *half, &*into);
    if (!run) return std::unexpected(std::move(run.error()));
    // Le premier instant d'une paire prise d'emblée mesure son mouvement pour la paire suivante
    // (interpolate_direct), comme s'il était fait au slot.
    if (k == first && direct_run_) {
      if (auto ticket = setup_.backend->kernels()->flow_max_start(run->motion)) {
        motion_ = PendingMotion{
            .ticket = *ticket, .scale = run->scale * 0.5F / std::max(t, 1.0F - t), .known = true};
      } else {
        flow_max_async_ = false;
      }
    }
    early_.runs[k] = *run;
    ++early_.passes;
    queued = true;
  }
  if (queued) ++early_.pairs;
  use_lane(back);
  return {};
}

std::expected<UpscaleStage::StrongRun, Refusal> UpscaleStage::strong_for(std::uint32_t output, float t,
                                                                         const gpu::Motion& full, bool half) {
  if (early_.ready && output < early_.runs.size()) {
    if (const auto& run = early_.runs[output]) return *run;
  }
  return run_strong(t, full, half);
}

}  // namespace ajp::app::play
