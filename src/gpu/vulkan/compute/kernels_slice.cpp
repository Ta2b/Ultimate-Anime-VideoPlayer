// Noyaux des extensions de la tranche verticale (gpu/kernel_ops.h) : entrée de RIFE depuis le
// RGB, mesures d'une paire RGB, plus grand déplacement d'un mouvement, softmax splatting de
// GMFSS. Mêmes calculs que les noyaux CUDA du backend TensorRT (rife_kernels.cu,
// warp_kernels.cu, gmfss_kernels.cu).
#include <array>
#include <cstdint>
#include <format>
#include <span>
#include <utility>
#include <vector>

#include "gpu/vulkan/compute/kernel_dispatch.h"
#include "gpu/vulkan/compute/kernel_runner.h"
#include "kernels/spec/detect.h"

namespace ajp::gpu::vk {
namespace {

using detail::bytes_of;
using detail::groups_for;

constexpr auto kHeldBlock = static_cast<std::uint32_t>(kernels::kHeldBlock);
// Plans d'un groupe du splatting (axe z) : la géométrie d'un pixel se refait par groupe, les
// atomiques d'un plan restent dans le même groupe.
constexpr std::uint32_t kSplatGroup = 16;
constexpr std::uint32_t kFillGroup = 64;  // local_size_x de tensor_fill.comp

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

VkDeviceSize KernelRunner::splat_sums_bytes(const TensorView& source) {
  const VkDeviceSize bytes = VkDeviceSize{source.planes + 1U} * source.width * source.height * sizeof(float);
  return (bytes + kSplatSumsAlign - 1) / kSplatSumsAlign * kSplatSumsAlign;
}

Done KernelRunner::soft_splat(Recorder& recorder, const TensorView& source, const TensorView& motion, float s,
                              const BufferView& sums, const TensorView& out) {
  const SplatViews job{.source = source, .motion = motion, .s = s, .out = out};
  return soft_splat_all(recorder, std::span(&job, 1), sums);
}

Done KernelRunner::soft_splat_all(Recorder& recorder, std::span<const SplatViews> jobs,
                                  const BufferView& sums) {
  const Features& enabled = context_->caps().enabled;
  if (!enabled.float_atomic_add) {
    return refuse(RefusalCode::FeatureUnavailable,
                  std::format("splat : {} sans additions atomiques de float", context_->caps().device_name));
  }
  if (jobs.empty()) return refuse(RefusalCode::InputMismatch, "splat : aucun travail");
  VkDeviceSize total = 0;
  for (const SplatViews& job : jobs) {
    if (!(job.s >= 0.0F && job.s <= 1.0F)) {
      return refuse(RefusalCode::FieldOutOfRange, std::format("splat : instant {} hors de [0, 1]", job.s));
    }
    if (job.source.planes == 0) return refuse(RefusalCode::InputMismatch, "splat : source sans plan");
    if (auto ok = check_tensor(job.source, job.source.planes, enabled, "splat : source"); !ok) return ok;
    if (auto ok = check_tensor(job.motion, kernels::kGmfssMotionChannels, enabled, "splat : mouvement"); !ok)
      return ok;
    if (auto ok = check_tensor(job.out, job.source.planes, enabled, "splat : sortie"); !ok) return ok;
    if (auto ok = check_same_size(job.source.width, job.source.height, job.motion.width, job.motion.height,
                                  "splat : source et mouvement");
        !ok) {
      return ok;
    }
    if (auto ok = check_same_size(job.source.width, job.source.height, job.out.width, job.out.height,
                                  "splat : source et sortie");
        !ok) {
      return ok;
    }
    if (job.source.precision != job.motion.precision || job.source.precision != job.out.precision) {
      return refuse(RefusalCode::InputMismatch, "splat : tenseurs de précisions différentes");
    }
    total += splat_sums_bytes(job.source);
  }
  if (sums.size < total) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("splat : {} o de sommes, {} attendus", sums.size, total));
  }
  // Une remise à zéro pour tous, puis les additions côte à côte (sommes distinctes), puis les
  // normalisations de même : deux barrières pour le lot au lieu de deux par splat.
  const BufferView all{.buffer = sums.buffer, .offset = sums.offset, .size = total};
  if (auto ok = recorder.fill(all, 0); !ok) return ok;
  std::vector<BufferView> views;
  views.reserve(jobs.size());
  VkDeviceSize offset = sums.offset;
  for (const SplatViews& job : jobs) {
    const VkDeviceSize bytes =
        VkDeviceSize{job.source.planes + 1U} * job.source.width * job.source.height * sizeof(float);
    views.push_back({.buffer = sums.buffer, .offset = offset, .size = bytes});
    offset += splat_sums_bytes(job.source);
  }
  for (std::size_t j = 0; j < jobs.size(); ++j) {
    const SplatViews& job = jobs[j];
    const detail::SplatPush push{job.source.width, job.source.height, job.source.planes, kSplatGroup, job.s};
    const std::array scatter{job.source.buffer, job.motion.buffer, views[j]};
    if (auto ok = detail::run(*pipelines_, recorder, "gmfss_splat", job.source.precision, {}, scatter, {},
                              bytes_of(push),
                              {groups_for(job.source.width), groups_for(job.source.height),
                               groups_for(job.source.planes, kSplatGroup)},
                              j == 0 ? DispatchOrder::AfterPrevious : DispatchOrder::BesidePrevious);
        !ok) {
      return ok;
    }
  }
  for (std::size_t j = 0; j < jobs.size(); ++j) {
    const SplatViews& job = jobs[j];
    const detail::SplatNormalizePush normalize{job.source.width, job.source.height, job.source.planes};
    const std::array buffers{views[j], job.out.buffer};
    if (auto ok =
            detail::run(*pipelines_, recorder, "gmfss_normalize", job.source.precision, {}, buffers, {},
                        bytes_of(normalize), {groups_for(job.source.width), groups_for(job.source.height), 1},
                        j == 0 ? DispatchOrder::AfterPrevious : DispatchOrder::BesidePrevious);
        !ok) {
      return ok;
    }
  }
  return {};
}

Done KernelRunner::fill(Recorder& recorder, const TensorView& tensor, float value) {
  if (auto ok = check_tensor(tensor, tensor.planes, context_->caps().enabled, "remplissage"); !ok) return ok;
  const std::uint32_t count = tensor.width * tensor.height * tensor.planes;
  const detail::FillPush push{count, value};
  const std::array buffers{tensor.buffer};
  return detail::run(*pipelines_, recorder, "tensor_fill", tensor.precision, {}, buffers, {}, bytes_of(push),
                     {groups_for(count, kFillGroup), 1, 1});
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
