// Noyaux des extensions de la tranche verticale (gpu/kernel_ops.h) : entrée de RIFE depuis le
// RGB, mesures d'une paire RGB, plus grand déplacement d'un mouvement. Mêmes calculs que les
// noyaux CUDA du backend TensorRT (rife_kernels.cu, warp_kernels.cu).
#include <array>
#include <cstdint>
#include <format>
#include <utility>

#include "gpu/vulkan/compute/kernel_dispatch.h"
#include "gpu/vulkan/compute/kernel_runner.h"
#include "kernels/spec/detect.h"

namespace ajp::gpu::vk {
namespace {

using detail::bytes_of;
using detail::groups_for;

constexpr auto kHeldBlock = static_cast<std::uint32_t>(kernels::kHeldBlock);

}  // namespace

Done KernelRunner::rife_frames(Recorder& recorder, const TensorView& a, const TensorView& b,
                               const TensorView& input, float black) {
  const Features& enabled = context_->caps().enabled;
  if (auto ok = check_tensor(a, 3, enabled, "entrée RIFE : image a"); !ok) return ok;
  if (auto ok = check_tensor(b, 3, enabled, "entrée RIFE : image b"); !ok) return ok;
  if (auto ok = check_tensor(input, kernels::kRifeInputChannels, enabled, "entrée RIFE"); !ok) return ok;
  if (auto ok = check_same_size(a.width, a.height, b.width, b.height, "entrée RIFE : images a et b"); !ok)
    return ok;
  if (a.precision != b.precision || a.precision != input.precision) {
    return refuse(RefusalCode::InputMismatch, "entrée RIFE : tenseurs de précisions différentes");
  }
  const kernels::RifeGeometry g =
      kernels::rife_geometry(static_cast<int>(a.width), static_cast<int>(a.height));
  if (std::cmp_not_equal(input.width, g.padded_width) || std::cmp_not_equal(input.height, g.padded_height)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("entrée RIFE {}x{} : {}x{} attendu pour {}x{}", input.width, input.height,
                              g.padded_width, g.padded_height, a.width, a.height));
  }
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
  if (result.size < sizeof(std::uint32_t))
    return refuse(RefusalCode::InputMismatch, "plus grand déplacement : résultat");
  if (auto ok =
          recorder.fill({.buffer = result.buffer, .offset = result.offset, .size = sizeof(std::uint32_t)}, 0);
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

}  // namespace ajp::gpu::vk
