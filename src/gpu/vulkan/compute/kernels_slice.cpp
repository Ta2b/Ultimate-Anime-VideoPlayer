// Noyaux des extensions de la tranche verticale (gpu/kernel_ops.h) : entrée de RIFE depuis le
// RGB, mesures d'une paire RGB, plus grand déplacement d'un mouvement. Mêmes calculs que les noyaux
// CUDA du backend TensorRT (rife_kernels.cu, warp_kernels.cu).
#include <array>
#include <bit>
#include <cstdint>
#include <format>
#include <span>
#include <utility>
#include <vector>

#include "gpu/vulkan/compute/kernel_dispatch.h"
#include "gpu/vulkan/compute/kernel_runner.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/warp.h"

namespace ajp::gpu::vk {
namespace {

using detail::bytes_of;
using detail::groups_for;

constexpr auto kHeldBlock = static_cast<std::uint32_t>(kernels::kHeldBlock);

}  // namespace

Done KernelRunner::rife_frames(Recorder& recorder, const TensorView& a, const TensorView& b,
                               const TensorView& input, float black, std::uint32_t planes) {
  const Features& enabled = context_->caps().enabled;
  if (auto ok = check_tensor(a, 3, enabled, "entrée RIFE : image a"); !ok) return ok;
  if (auto ok = check_tensor(b, 3, enabled, "entrée RIFE : image b"); !ok) return ok;
  if (planes < 6) return refuse(RefusalCode::InputMismatch, std::format("entrée RIFE : {} plans", planes));
  if (auto ok = check_tensor(input, planes, enabled, "entrée RIFE"); !ok) return ok;
  if (auto ok = check_same_size(a.width, a.height, b.width, b.height, "entrée RIFE : images a et b"); !ok)
    return ok;
  if (a.precision != b.precision || a.precision != input.precision) {
    return refuse(RefusalCode::InputMismatch, "entrée RIFE : tenseurs de précisions différentes");
  }
  if (!kernels::rife_fits(static_cast<int>(a.width), static_cast<int>(a.height),
                          static_cast<int>(input.width), static_cast<int>(input.height))) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("entrée RIFE {}x{} : multiples de {} d'au moins {}x{} attendus", input.width,
                              input.height, kernels::kRifeAlign, a.width, a.height));
  }
  // La fenêtre se lit sur le tenseur : le multiple de remplissage dépend du modèle (spec/rife.h).
  const kernels::RifeGeometry g =
      kernels::rife_window(static_cast<int>(a.width), static_cast<int>(a.height),
                           static_cast<int>(input.width), static_cast<int>(input.height));
  const detail::RifeFramesPush push{a.width,
                                    a.height,
                                    input.width,
                                    input.height,
                                    static_cast<std::uint32_t>(g.left),
                                    static_cast<std::uint32_t>(g.top),
                                    black};
  const std::array buffers{a.buffer, b.buffer, input.buffer};
  return detail::run(*pipelines_, recorder, "rife_frames", input.precision, {}, buffers, {}, bytes_of(push),
                     {groups_for(input.width), groups_for(input.height), 1});
}

Done KernelRunner::rgb_pair_diff(Recorder& recorder, const TensorView& a, const TensorView& b,
                                 const BufferView& sums, const BufferView& worst, float levels) {
  const Features& enabled = context_->caps().enabled;
  if (auto ok = check_tensor(a, 3, enabled, "mesures : image a"); !ok) return ok;
  if (auto ok = check_tensor(b, 3, enabled, "mesures : image b"); !ok) return ok;
  if (auto ok = check_same_size(a.width, a.height, b.width, b.height, "mesures : images a et b"); !ok)
    return ok;
  if (a.precision != b.precision)
    return refuse(RefusalCode::InputMismatch, "mesures : précisions différentes");
  const std::uint32_t count = block_count(a.width, a.height);
  if (sums.size < VkDeviceSize{count} * sizeof(float) || worst.size < sizeof(std::uint32_t)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("mesures : {} o pour {} sommes de blocs, {} o pour le pire", sums.size, count,
                              worst.size));
  }
  if (auto ok =
          recorder.fill({.buffer = worst.buffer, .offset = worst.offset, .size = sizeof(std::uint32_t)}, 0);
      !ok) {
    return ok;
  }
  const detail::RgbPairPush push{a.width, a.height, levels};
  const std::array buffers{a.buffer, b.buffer, sums, worst};
  return detail::run(*pipelines_, recorder, "rgb_pair_diff", a.precision, {}, buffers, {}, bytes_of(push),
                     {groups_for(a.width, kHeldBlock), groups_for(a.height, kHeldBlock), 1});
}

Done KernelRunner::flow_max(Recorder& recorder, const MotionView& motion, const BufferView& result) {
  if (auto ok = check_tensor(motion.flow, 4, context_->caps().enabled, "plus grand déplacement : flot"); !ok)
    return ok;
  const kernels::Window& area = motion.area;
  const std::int64_t right = std::int64_t{area.x} + area.width;
  const std::int64_t bottom = std::int64_t{area.y} + area.height;
  if (area.x < 0 || area.y < 0 || area.width < 1 || area.height < 1 || right > motion.flow.width ||
      bottom > motion.flow.height) {
    return refuse(
        RefusalCode::InputMismatch,
        std::format("plus grand déplacement : fenêtre {}x{} en ({}, {}) hors du mouvement {}x{}", area.width,
                    area.height, area.x, area.y, motion.flow.width, motion.flow.height));
  }
  constexpr VkDeviceSize kHistogramBytes = kernels::kFlowBins * sizeof(std::uint32_t);
  if (result.size < kHistogramBytes)
    return refuse(RefusalCode::InputMismatch, "plus grand déplacement : résultat");
  if (auto ok = recorder.fill({.buffer = result.buffer, .offset = result.offset, .size = kHistogramBytes}, 0);
      !ok) {
    return ok;
  }
  const detail::FlowMaxPush push{motion.flow.width,
                                 motion.flow.height,
                                 static_cast<std::uint32_t>(area.x),
                                 static_cast<std::uint32_t>(area.y),
                                 static_cast<std::uint32_t>(area.width),
                                 static_cast<std::uint32_t>(area.height)};
  const std::array buffers{motion.flow.buffer, result};
  return detail::run(*pipelines_, recorder, "flow_max", motion.flow.precision, {}, buffers, {},
                     bytes_of(push),
                     {groups_for(static_cast<std::uint32_t>(area.width)),
                      groups_for(static_cast<std::uint32_t>(area.height)), 1});
}

Done KernelRunner::frame_stats(Recorder& recorder, const TensorView& rgb, kernels::FrameTransfer transfer,
                               const BufferView& counts) {
  if (auto ok = check_tensor(rgb, 3, context_->caps().enabled, "analyse : image"); !ok) return ok;
  if (rgb.width < 2 || rgb.height < 2)
    return refuse(RefusalCode::ShapeInvalid, "analyse : image trop petite");
  if (counts.size < sizeof(kernels::FrameCounts))
    return refuse(RefusalCode::InputMismatch, std::format("analyse : {} o pour les comptes", counts.size));
  const detail::FrameStatsPush push{rgb.width, rgb.height, static_cast<std::uint32_t>(transfer)};
  const std::array buffers{rgb.buffer, counts};
  return detail::run(*pipelines_, recorder, "frame_stats", rgb.precision, {}, buffers, {}, bytes_of(push),
                     {1, 1, 1});
}

}  // namespace ajp::gpu::vk
