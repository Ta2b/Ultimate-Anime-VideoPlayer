// GMFSS Fortuna dans l'étage gpu (upscale_stage.h, kernels/spec/gmfss.h) : ses tenseurs nommés
// comme ses graphes, rangés dans l'ordre de ses moteurs, puis l'image intermédiaire : le graphe
// « paire » à la première de la paire, six softmax splats et le graphe « fusion » à chacune. La
// fusion du modèle union reçoit aussi l'instant, un tenseur « time » 1 × 1 × 1 × 1. Déformée après
// SPAN, l'image intermédiaire ajoute quatre splats (les images de travail et les agrandies, par le
// flot du niveau 1) au même passage, puis la composition (spec gmfss_compose) à la taille de sortie.
#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <span>
#include <string>

#include "app/play/options.h"
#include "app/play/upscale_stage.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

}  // namespace

std::expected<void, Refusal> UpscaleStage::allocate_gmfss() {
  gpu::Backend& backend = *setup_.backend;
  const kernels::RifeGeometry g =
      kernels::rife_geometry(setup_.work_width, setup_.work_height, setup_.gmfss_align);
  const auto alloc = [&](int channels, int scale) {
    return backend.alloc_tensor(
        gpu::TensorType::F16,
        gpu::Shape{.dims = {1, channels, g.padded_height / scale, g.padded_width / scale}});
  };
  const auto add = [&](std::string name, int channels, int scale) -> std::expected<void, Refusal> {
    auto tensor = alloc(channels, scale);
    if (!tensor) return std::unexpected(std::move(tensor.error()));
    gmfss_.named.emplace_back(std::move(name), std::move(*tensor));
    return {};
  };
  auto frames = alloc(kernels::kGmfssFramePlanes, 1);
  if (!frames) return std::unexpected(std::move(frames.error()));
  gmfss_.frames = std::move(*frames);
  auto out = alloc(3, 1);
  if (!out) return std::unexpected(std::move(out.error()));
  rife_out_ = std::move(*out);
  for (int side = 0; side < 2; ++side) {
    if (auto ok = add(std::format("half{}", side), 3, 2); !ok) return ok;
    for (int level = 1; level <= kernels::kGmfssLevels; ++level) {
      const int channels = kernels::gmfss_source_channels(level);
      const int scale = 1 << level;
      if (auto ok = add(std::format("source{}_{}", side, level), channels, scale); !ok) return ok;
      if (auto ok = add(std::format("motion{}_{}", side, level), kernels::kGmfssMotionChannels, scale); !ok)
        return ok;
      if (auto ok = add(std::format("warped{}_{}", side, level), channels, scale); !ok) return ok;
    }
  }
  const std::span<const std::string> fusion_names = setup_.gmfss_fusion->input_names();
  if (std::ranges::contains(fusion_names, std::string_view("time"))) {
    auto time = backend.alloc_tensor(gpu::TensorType::F16, gpu::Shape{.dims = {1, 1, 1, 1}});
    if (!time) return std::unexpected(std::move(time.error()));
    gmfss_.time = (*time)->tensor();
    gmfss_.named.emplace_back("time", std::move(*time));
  }
  const auto tensor = [&](std::string_view name) -> const gpu::Tensor* {
    const auto found = std::ranges::find(gmfss_.named, name, [](const auto& entry) { return entry.first; });
    return found == gmfss_.named.end() ? nullptr : &found->second->tensor();
  };
  // Les moteurs disent l'ordre de leurs tenseurs ; un nom inconnu vient d'un autre export.
  const auto order = [&](std::span<const std::string> names, std::vector<gpu::Tensor>& into,
                         std::string_view graph) -> std::expected<void, Refusal> {
    for (const std::string& name : names) {
      const gpu::Tensor* found = tensor(name);
      if (found == nullptr) {
        return refuse(RefusalCode::GraphInvalid,
                      std::format("GMFSS : tenseur « {} » inconnu dans le graphe {}", name, graph));
      }
      into.push_back(*found);
    }
    return {};
  };
  if (auto ok = order(setup_.gmfss_pair->output_names(), gmfss_.pair_outputs, "paire"); !ok) return ok;
  if (auto ok = order(fusion_names, gmfss_.fusion_inputs, "fusion"); !ok) return ok;
  if (setup_.gmfss_pair->input_names().size() != 1 || setup_.gmfss_fusion->output_names().size() != 1) {
    return refuse(RefusalCode::GraphInvalid,
                  "GMFSS : une entrée à la paire, une sortie à la fusion attendues");
  }
  for (int side = 0; side < 2; ++side) {
    for (int level = 1; level <= kernels::kGmfssLevels; ++level) {
      gmfss_.splats.push_back({.side = side,
                               .source = *tensor(std::format("source{}_{}", side, level)),
                               .motion = *tensor(std::format("motion{}_{}", side, level)),
                               .warped = *tensor(std::format("warped{}_{}", side, level))});
    }
    gmfss_.level1[static_cast<std::size_t>(side)] = *tensor(std::format("motion{}_1", side));
  }
  if (!warps_spanned()) return {};
  // Le flot du niveau 1 est à la moitié des plans remplis : l'image de travail en lit deux pixels
  // par pixel de flot, son agrandissement quatre, depuis le coin de la fenêtre (spec MotionGrid).
  gmfss_.work_grid = {
      .scale = 2.0F, .x = static_cast<float>(g.left) / 2.0F, .y = static_cast<float>(g.top) / 2.0F};
  gmfss_.spanned_grid = {.scale = 2.0F * kSpanFactor, .x = gmfss_.work_grid.x, .y = gmfss_.work_grid.y};
  gmfss_.window = {.x = g.left, .y = g.top, .width = setup_.work_width, .height = setup_.work_height};
  for (int side = 0; side < 2; ++side) {
    auto low = backend.alloc_tensor(gpu::TensorType::F16,
                                    gpu::Shape{.dims = {1, 4, setup_.work_height, setup_.work_width}});
    if (!low) return std::unexpected(std::move(low.error()));
    gmfss_.low[static_cast<std::size_t>(side)] = std::move(*low);
    auto high = backend.alloc_tensor(
        gpu::TensorType::F16,
        gpu::Shape{.dims = {1, 4, setup_.work_height * kSpanFactor, setup_.work_width * kSpanFactor}});
    if (!high) return std::unexpected(std::move(high.error()));
    gmfss_.high[static_cast<std::size_t>(side)] = std::move(*high);
  }
  return {};
}

std::expected<void, Refusal> UpscaleStage::run_gmfss(std::uint32_t output,
                                                     std::span<const gpu::KernelOps::SplatJob> extra) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (output == 0) {  // le graphe « paire » sert toutes les images intermédiaires de la paire
    if (auto ok = ops.gmfss_frames(rgb_[current_ ^ 1U]->tensor(), rgb_[current_]->tensor(),
                                   gmfss_.frames->tensor());
        !ok) {
      return ok;
    }
    const std::array inputs{gmfss_.frames->tensor()};
    if (auto ok = setup_.backend->enqueue(*setup_.gmfss_pair, inputs, gmfss_.pair_outputs); !ok) return ok;
    ++gmfss_pairs_;
  }
  const float t = instants_[output].t;
  gmfss_.jobs.clear();
  for (const GmfssSplat& splat : gmfss_.splats) {
    gmfss_.jobs.push_back({.source = splat.source,
                           .motion = splat.motion,
                           .s = kernels::gmfss_side_time(splat.side, t),
                           .out = splat.warped});
  }
  gmfss_.jobs.insert(gmfss_.jobs.end(), extra.begin(), extra.end());
  if (auto ok = ops.soft_splat_all(gmfss_.jobs); !ok) return ok;
  if (gmfss_.time) {
    if (auto ok = ops.fill(*gmfss_.time, t); !ok) return ok;
  }
  std::array results{rife_out_->tensor()};
  return setup_.backend->enqueue(*setup_.gmfss_fusion, gmfss_.fusion_inputs, results);
}

std::expected<void, Refusal> UpscaleStage::interpolate_gmfss(std::uint32_t output) {
  if (auto ok = run_gmfss(output, {}); !ok) return ok;
  return setup_.backend->kernels()->rife_crop(rife_out_->tensor(), middle_->tensor());
}

std::expected<void, Refusal> UpscaleStage::interpolate_gmfss_spanned(std::uint32_t output,
                                                                     const gpu::Tensor& a,
                                                                     const gpu::Tensor& b,
                                                                     const gpu::Tensor& out) {
  // SPAN de b avant ses splats (a l'a été à son tour) ; les quatre splats du détail rejoignent le
  // passage des six de la fusion.
  if (auto ok = span_current(); !ok) return ok;
  const float t = instants_[output].t;
  const std::array<const gpu::Tensor*, 2> work{&rgb_[current_ ^ 1U]->tensor(), &rgb_[current_]->tensor()};
  const std::array<const gpu::Tensor*, 2> spanned{&a, &b};
  std::array<gpu::KernelOps::SplatJob, 4> extra;
  for (std::size_t side = 0; side < 2; ++side) {
    const float s = kernels::gmfss_side_time(static_cast<int>(side), t);
    extra[side] = {.source = *work[side],
                   .motion = gmfss_.level1[side],
                   .s = s,
                   .out = gmfss_.low[side]->tensor(),
                   .grid = gmfss_.work_grid};
    extra[2 + side] = {.source = *spanned[side],
                       .motion = gmfss_.level1[side],
                       .s = s,
                       .out = gmfss_.high[side]->tensor(),
                       .grid = gmfss_.spanned_grid};
  }
  if (auto ok = run_gmfss(output, extra); !ok) return ok;
  return setup_.backend->kernels()->gmfss_compose(gmfss_.high[0]->tensor(), gmfss_.high[1]->tensor(),
                                                  gmfss_.low[0]->tensor(), gmfss_.low[1]->tensor(),
                                                  rife_out_->tensor(), gmfss_.window, t, out);
}

}  // namespace ajp::app::play
