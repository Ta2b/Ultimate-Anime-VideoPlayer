// Fenêtres de RIFE (remplissage mod-64 centré et recadrage) et détections temporelles (coupe de
// scène, dessins tenus). Contrats : kernels/spec/rife.h et kernels/spec/detect.h.
#include <array>
#include <format>
#include <string_view>

#include "gpu/vulkan/compute/kernel_dispatch.h"
#include "gpu/vulkan/compute/kernel_runner.h"
#include "kernels/spec/detect.h"

namespace ajp::gpu::vk {
namespace {

using detail::bytes_of;
using detail::groups_for;

constexpr std::uint32_t kEdgeLuma = 0;
constexpr std::uint32_t kEdgeChroma = 1;

// Un plan de la fenêtre : dimensions en échantillons, décalage de la source dans la destination.
struct PlaneWindow {
  PlaneLayout src;
  PlaneLayout dst;
  std::uint32_t src_width;
  std::uint32_t src_height;
  std::uint32_t dst_width;
  std::uint32_t dst_height;
  std::int32_t shift_x;
  std::int32_t shift_y;
  std::uint32_t edge;
};

Done check_padded(const YuvView& padded, const YuvView& frame, std::string_view what) {
  if (padded.format != frame.format) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : formats {} et {} différents", what, static_cast<int>(padded.format),
                              static_cast<int>(frame.format)));
  }
  if (!kernels::rife_fits(static_cast<int>(frame.width), static_cast<int>(frame.height),
                          static_cast<int>(padded.width), static_cast<int>(padded.height))) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : {}x{} rempli, multiples de {} d'au moins {}x{} attendus", what,
                              padded.width, padded.height, kernels::kRifeAlign, frame.width, frame.height));
  }
  return {};
}

// La fenêtre de l'image dans l'image remplie, lue sur leurs tailles (spec/rife.h).
kernels::RifeGeometry window_of(std::uint32_t width, std::uint32_t height, std::uint32_t padded_width,
                                std::uint32_t padded_height) {
  return kernels::rife_window(static_cast<int>(width), static_cast<int>(height),
                              static_cast<int>(padded_width), static_cast<int>(padded_height));
}

// Chaque plan de l'image : sign = 1 place la source au centre, −1 en retire la fenêtre.
Done windows(PipelineCache& pipelines, Recorder& recorder, const YuvView& src, const YuvView& dst,
             const kernels::RifeGeometry& g, int sign) {
  const kernels::FormatInfo info = kernels::info(src.format);
  const auto bytes = static_cast<std::uint32_t>(info.bytes);
  const int d = info.subsampled ? 2 : 1;
  const std::uint32_t pairs = info.interleaved ? 2U : 1U;  // échantillons par pixel de chroma
  const std::uint32_t src_cw = chroma_width(src.format, src.width);
  const std::uint32_t dst_cw = chroma_width(dst.format, dst.width);
  const std::uint32_t src_ch = chroma_height(src.format, src.height);
  const std::uint32_t dst_ch = chroma_height(dst.format, dst.height);
  const std::int32_t cx = sign * static_cast<std::int32_t>(pairs) * (g.left / d);
  const std::int32_t cy = sign * (g.top / d);
  std::array<PlaneWindow, 3> planes{
      PlaneWindow{src.layout.luma, dst.layout.luma, src.width, src.height, dst.width, dst.height,
                  sign * g.left, sign * g.top, kEdgeLuma},
      PlaneWindow{src.layout.cb, dst.layout.cb, pairs * src_cw, src_ch, pairs * dst_cw, dst_ch, cx, cy,
                  kEdgeChroma},
      PlaneWindow{src.layout.cr, dst.layout.cr, src_cw, src_ch, dst_cw, dst_ch, cx, cy, kEdgeChroma}};
  const std::size_t count = info.interleaved ? 2 : 3;
  const auto constants = detail::yuv_constants(src.format, kernels::kRifeMatrix, kernels::Range::Limited);
  const std::array buffers{src.buffer, dst.buffer};
  for (std::size_t i = 0; i < count; ++i) {
    const PlaneWindow& p = planes[i];
    const detail::YuvWindowPush push{p.src.offset, p.src.pitch, p.src_width, p.src_height,
                                     p.dst.offset, p.dst.pitch, p.dst_width, p.dst_height,
                                     p.shift_x,    p.shift_y,   p.edge};
    if (auto ok = detail::run(pipelines, recorder, "yuv_window", Precision::F32, constants, buffers, {},
                              bytes_of(push),
                              {groups_for((p.dst_width * bytes + 3U) / 4U), groups_for(p.dst_height), 1});
        !ok) {
      return ok;
    }
  }
  return {};
}

// Deux lumas comparables : images conformes, même format, mêmes dimensions.
Done check_pair(const YuvView& a, const YuvView& b, std::string_view what) {
  if (auto ok = check_yuv(a, what); !ok) return ok;
  if (auto ok = check_yuv(b, what); !ok) return ok;
  if (a.format != b.format)
    return refuse(RefusalCode::InputMismatch, std::format("{} : formats différents", what));
  return check_same_size(a.width, a.height, b.width, b.height, what);
}

detail::PairPush pair_push(const YuvView& a, const YuvView& b) {
  return {a.layout.luma.offset,
          a.layout.luma.pitch,
          b.layout.luma.offset,
          b.layout.luma.pitch,
          a.width,
          a.height};
}

constexpr auto kHeldBlock = static_cast<std::uint32_t>(kernels::kHeldBlock);

}  // namespace

std::uint32_t KernelRunner::scene_partial_count(std::uint32_t width, std::uint32_t height) {
  return groups_for(width) * groups_for(height);
}

std::uint32_t KernelRunner::block_count(std::uint32_t width, std::uint32_t height) {
  return groups_for(width, kHeldBlock) * groups_for(height, kHeldBlock);
}

Done KernelRunner::rife_pad(Recorder& recorder, const YuvView& src, const YuvView& padded) {
  if (auto ok = check_yuv(src, "remplissage RIFE : image"); !ok) return ok;
  if (auto ok = check_yuv(padded, "remplissage RIFE : image remplie"); !ok) return ok;
  if (auto ok = check_padded(padded, src, "remplissage RIFE"); !ok) return ok;
  const kernels::RifeGeometry g = window_of(src.width, src.height, padded.width, padded.height);
  return windows(*pipelines_, recorder, src, padded, g, 1);
}

Done KernelRunner::rife_crop(Recorder& recorder, const YuvView& padded, const YuvView& dst) {
  if (auto ok = check_yuv(padded, "recadrage RIFE : image remplie"); !ok) return ok;
  if (auto ok = check_yuv(dst, "recadrage RIFE : fenêtre"); !ok) return ok;
  if (auto ok = check_padded(padded, dst, "recadrage RIFE"); !ok) return ok;
  const kernels::RifeGeometry g = window_of(dst.width, dst.height, padded.width, padded.height);
  return windows(*pipelines_, recorder, padded, dst, g, -1);
}

Done KernelRunner::rife_crop(Recorder& recorder, const TensorView& padded, const TensorView& dst) {
  const Features& enabled = context_->caps().enabled;
  if (auto ok = check_tensor(dst, dst.planes == 0 ? 1 : dst.planes, enabled, "recadrage RIFE : fenêtre");
      !ok) {
    return ok;
  }
  if (auto ok = check_tensor(padded, dst.planes, enabled, "recadrage RIFE : sortie remplie"); !ok) return ok;
  if (padded.precision != dst.precision) {
    return refuse(RefusalCode::InputMismatch, "recadrage RIFE : précisions différentes");
  }
  if (!kernels::rife_fits(static_cast<int>(dst.width), static_cast<int>(dst.height),
                          static_cast<int>(padded.width), static_cast<int>(padded.height))) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("recadrage RIFE : {}x{} rempli, multiples de {} d'au moins {}x{} attendus",
                              padded.width, padded.height, kernels::kRifeAlign, dst.width, dst.height));
  }
  const kernels::RifeGeometry g = window_of(dst.width, dst.height, padded.width, padded.height);
  const detail::RgbWindowPush push{padded.width, padded.height, dst.width, dst.height, -g.left, -g.top, 0.0F};
  const std::array buffers{padded.buffer, dst.buffer};
  return detail::run(*pipelines_, recorder, "rgb_window", dst.precision, {}, buffers, {}, bytes_of(push),
                     {groups_for(dst.width), groups_for(dst.height), dst.planes});
}

Done KernelRunner::scene_diff(Recorder& recorder, const YuvView& a, const YuvView& b,
                              const BufferView& partials) {
  if (auto ok = check_pair(a, b, "coupe de scène"); !ok) return ok;
  const std::uint32_t count = scene_partial_count(a.width, a.height);
  if (partials.size < VkDeviceSize{count} * sizeof(float)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("coupe de scène : {} o pour {} sommes partielles", partials.size, count));
  }
  const auto constants = detail::yuv_constants(a.format, kernels::Matrix::Bt709, kernels::Range::Limited);
  const detail::PairPush push = pair_push(a, b);
  const std::array buffers{a.buffer, b.buffer, partials};
  return detail::run(*pipelines_, recorder, "scene_diff", Precision::F32, constants, buffers, {},
                     bytes_of(push), {groups_for(a.width), groups_for(a.height), 1});
}

Done KernelRunner::block_diff(Recorder& recorder, const YuvView& a, const YuvView& b,
                              const BufferView& result) {
  if (auto ok = check_pair(a, b, "dessins tenus"); !ok) return ok;
  const std::uint32_t count = block_count(a.width, a.height);
  if (result.size < (VkDeviceSize{count} + 1U) * sizeof(std::uint32_t)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("dessins tenus : {} o pour {} blocs et le pire", result.size, count));
  }
  // Le pire écart repart de 0 à chaque exécution : l'atomicMax ne garde rien d'une précédente.
  if (auto ok =
          recorder.fill({.buffer = result.buffer, .offset = result.offset, .size = sizeof(std::uint32_t)}, 0);
      !ok) {
    return ok;
  }
  const auto constants = detail::yuv_constants(a.format, kernels::Matrix::Bt709, kernels::Range::Limited);
  const detail::PairPush push = pair_push(a, b);
  const std::array buffers{a.buffer, b.buffer, result};
  return detail::run(*pipelines_, recorder, "block_diff", Precision::F32, constants, buffers, {},
                     bytes_of(push), {groups_for(a.width, kHeldBlock), groups_for(a.height, kHeldBlock), 1});
}

}  // namespace ajp::gpu::vk
