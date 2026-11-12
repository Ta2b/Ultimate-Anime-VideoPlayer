// Échelle par paire (PlaySettings::pair_scale) dans l'étage gpu (upscale_stage.h) : une paire qui bouge fort,
// où RIFE mouvement passe à chaque instant hors du milieu, l'estime par un second moteur, sur a et b réduites
// de moitié (pair_scale_side, filtre Area). Le warp met ce mouvement à l'échelle des images comme il le fait
// pour la taille de RIFE ; le milieu mesuré et les paires calmes restent à pleine taille. Une paire n'est
// dite forte que sur la mesure de la précédente, ou sur celle de son propre milieu (upscale_motion.cpp).
#include <array>
#include <utility>

#include "app/play/upscale_stage.h"
#include "kernels/spec/resample.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {

std::expected<void, Refusal> UpscaleStage::allocate_half() {
  if (!halves()) return {};
  const int w = pair_scale_side(setup_.rife_width);
  const int h = pair_scale_side(setup_.rife_height);
  const kernels::RifeGeometry g = kernels::rife_geometry(w, h, setup_.rife_align);
  const auto alloc = [&](std::unique_ptr<gpu::TensorBuffer>& into, int channels, int width, int height) {
    auto tensor =
        setup_.backend->alloc_tensor(gpu::TensorType::F16, gpu::Shape{.dims = {1, channels, height, width}});
    if (tensor) into = std::move(*tensor);
    return tensor ? std::expected<void, Refusal>{} : std::unexpected(std::move(tensor.error()));
  };
  for (auto& source : half_.sources) {
    if (auto ok = alloc(source, 3, w, h); !ok) return ok;
  }
  if (setup_.smooth_drawings) {
    if (auto ok = alloc(half_.smooth, 3, w, h); !ok) return ok;
  }
  for (std::uint32_t i = 0; i < copies_; ++i) {
    if (auto ok = alloc(half_.ins[i], setup_.rife_planes, g.padded_width, g.padded_height); !ok) return ok;
  }
  if (auto ok = alloc(half_.flow, 4, g.padded_width, g.padded_height); !ok) return ok;
  return alloc(half_.mask, 1, g.padded_width, g.padded_height);
}

gpu::Motion UpscaleStage::half_motion() const {
  const int w = pair_scale_side(setup_.rife_width);
  const int h = pair_scale_side(setup_.rife_height);
  const kernels::RifeGeometry g = kernels::rife_geometry(w, h, setup_.rife_align);
  return {.flow = half_.flow->tensor(),
          .mask = half_.mask->tensor(),
          .area = kernels::Window{.x = g.left, .y = g.top, .width = w, .height = h}};
}

std::expected<const gpu::Tensor*, Refusal> UpscaleStage::half_source(std::uint32_t index, bool smooth) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (smooth) {
    if (auto ok = ops.resample(smooth_b_rife(), kernels::Filter::Area, half_.smooth->tensor()); !ok)
      return std::unexpected(std::move(ok.error()));
    return &half_.smooth->tensor();
  }
  const gpu::Tensor& reduced = half_.sources[index]->tensor();
  if (half_.serials[index] == source_serial_[index]) return &reduced;
  half_.serials[index] = 0;
  if (auto ok = ops.resample(rife_source(index), kernels::Filter::Area, reduced); !ok)
    return std::unexpected(std::move(ok.error()));
  half_.serials[index] = source_serial_[index];
  return &reduced;
}

std::expected<void, Refusal> UpscaleStage::fill_half_input(float t) {
  if (half_.in_pair && half_.in_t == t) return {};
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const gpu::Tensor& input = half_.ins[rife_pair_]->tensor();
  if (half_.in_pair && rife_time_) {
    auto ok = ops.rife_time(input, t);
    if (ok) half_.in_t = t;
    if (ok || ok.error().code != RefusalCode::FormatUnsupported) return ok;
    rife_time_ = false;
  }
  auto a = half_source(current_ ^ 1U, false);
  if (!a) return std::unexpected(std::move(a.error()));
  auto b = half_source(current_, smooth_first_);
  if (!b) return std::unexpected(std::move(b.error()));
  auto ok = ops.rife_input(**a, **b, t, input);
  // Comme rife_in_images_ : B lissée n'est pas la source current_.
  half_.in_images[rife_pair_] = {ok ? source_serial_[current_ ^ 1U] : 0U,
                                 ok && !smooth_first_ ? source_serial_[current_] : 0U};
  half_.in_pair = ok.has_value();
  half_.in_t = t;
  return ok;
}

std::expected<UpscaleStage::StrongRun, Refusal> UpscaleStage::run_strong(float t, const gpu::Motion& full,
                                                                         bool half) {
  if (!half || !halves()) {
    if (auto ok = run_motion(t, full.flow, full.mask); !ok) return std::unexpected(std::move(ok.error()));
    return StrongRun{.motion = full};
  }
  if (auto ok = fill_half_input(t); !ok) return std::unexpected(std::move(ok.error()));
  ++rife_runs_;
  ++half_.runs;
  warm_from_ = nullptr;  // P6.15 : les blocs grossiers d'un autre moteur ne se reprennent pas
  const gpu::Motion motion = half_motion();
  std::array results{motion.flow, motion.mask};
  if (auto ok = enqueue_on(*setup_.rife_motion_half, half_.ins[rife_pair_]->tensor(),
                           half_.in_images[rife_pair_], half_.encoded, results);
      !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  // Un pixel de la réduite en vaut rife_height / sa hauteur de RIFE.
  return StrongRun{.motion = motion,
                   .scale = static_cast<float>(setup_.rife_height) /
                            static_cast<float>(pair_scale_side(setup_.rife_height))};
}

}  // namespace ajp::app::play
