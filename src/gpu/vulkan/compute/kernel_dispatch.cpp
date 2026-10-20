#include "gpu/vulkan/compute/kernel_dispatch.h"

#include <format>
#include <string>

namespace ajp::gpu::vk::detail {
namespace {

// Les numéros de yuv.glsl suivent les énumérations de kernels/spec/color.h.
static_assert(static_cast<int>(kernels::Format::Nv12) == 0 && static_cast<int>(kernels::Format::P010) == 1 &&
              static_cast<int>(kernels::Format::P016) == 2 &&
              static_cast<int>(kernels::Format::Yuv420p) == 3 &&
              static_cast<int>(kernels::Format::Yuv420p10) == 4 &&
              static_cast<int>(kernels::Format::Yuv444p16) == 5);
static_assert(static_cast<int>(kernels::Matrix::Bt601) == 0 &&
              static_cast<int>(kernels::Matrix::Bt709) == 1 &&
              static_cast<int>(kernels::Matrix::Bt2020Ncl) == 2);
static_assert(static_cast<int>(kernels::Range::Limited) == 0 && static_cast<int>(kernels::Range::Full) == 1);

}  // namespace

std::array<std::uint32_t, 6> yuv_constants(kernels::Format format, kernels::Matrix matrix,
                                           kernels::Range range) {
  const kernels::FormatInfo info = kernels::info(format);
  return {static_cast<std::uint32_t>(format), static_cast<std::uint32_t>(matrix),
          static_cast<std::uint32_t>(range),  static_cast<std::uint32_t>(info.bytes),
          info.interleaved ? 1U : 0U,         info.subsampled ? 1U : 0U};
}

Done check_color(const ColorSpec& color) {
  if (static_cast<int>(color.matrix) > static_cast<int>(kernels::Matrix::Bt2020Ncl) ||
      static_cast<int>(color.range) > static_cast<int>(kernels::Range::Full) ||
      static_cast<int>(color.siting) > static_cast<int>(kernels::Siting::TopLeft)) {
    return refuse(
        RefusalCode::InputMismatch,
        std::format("couleur inconnue : matrice {}, plage {}, siting {}", static_cast<int>(color.matrix),
                    static_cast<int>(color.range), static_cast<int>(color.siting)));
  }
  return {};
}

Done check_filter(kernels::Filter filter) {
  if (static_cast<int>(filter) > static_cast<int>(kernels::Filter::Area)) {
    return refuse(RefusalCode::InputMismatch, std::format("filtre {} inconnu", static_cast<int>(filter)));
  }
  return {};
}

std::expected<WeightViews, Refusal> record_weights(Recorder& recorder, const WeightTable& table) {
  auto start = recorder.constants(std::as_bytes(std::span(table.start)));
  if (!start) return std::unexpected(start.error());
  auto weights = recorder.constants(std::as_bytes(std::span(table.w)));
  if (!weights) return std::unexpected(weights.error());
  return WeightViews{.start = *start, .weights = *weights, .taps = table.taps};
}

Done run(PipelineCache& pipelines, Recorder& recorder, std::string_view shader, Precision variant,
         std::span<const std::uint32_t> constants, std::span<const BufferView> buffers,
         std::span<const VkImageView> images, std::span<const std::byte> push,
         std::array<std::uint32_t, 3> groups, DispatchOrder order) {
  std::string name(shader);
  if (variant == Precision::F16) name += "_f16";
  auto pipeline = pipelines.get({.shader = name,
                                 .constants = constants,
                                 .buffers = static_cast<std::uint32_t>(buffers.size()),
                                 .images = static_cast<std::uint32_t>(images.size()),
                                 .push_bytes = static_cast<std::uint32_t>(push.size())});
  if (!pipeline) return std::unexpected(pipeline.error());
  return recorder.dispatch(**pipeline, buffers, images, push, groups, order);
}

}  // namespace ajp::gpu::vk::detail
