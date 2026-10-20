// Les noyaux qui lisent une image décodée en place, plan par plan (planes.glsl), au lieu de sa
// copie serrée : ingestion, remplissage RIFE, coupe de scène, dessins tenus. Mêmes contrats et
// mêmes résultats que leurs homologues sur YuvView ; les blocs push gardent la disposition des
// shaders d'origine, dont les offsets et pas ne servent pas ici.
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
constexpr auto kHeldBlock = static_cast<std::uint32_t>(kernels::kHeldBlock);

// Deux images comparables : plans conformes, même format, mêmes dimensions.
Done check_pair(const YuvPlanes& a, const YuvPlanes& b, std::string_view what) {
  if (auto ok = check_planes(a, what); !ok) return ok;
  if (auto ok = check_planes(b, what); !ok) return ok;
  if (a.format != b.format)
    return refuse(RefusalCode::InputMismatch, std::format("{} : formats différents", what));
  return check_same_size(a.width, a.height, b.width, b.height, what);
}

}  // namespace

Done KernelRunner::ingest(Recorder& recorder, const YuvPlanes& src, const ColorSpec& color,
                          const TensorView& dst, kernels::Filter chroma) {
  if (auto ok = check_planes(src, "ingestion : plans de l'image"); !ok) return ok;
  if (auto ok = check_tensor(dst, 3, context_->caps().enabled, "ingestion : tenseur RGB"); !ok) return ok;
  if (auto ok = check_same_size(src.width, src.height, dst.width, dst.height, "ingestion"); !ok) return ok;
  if (auto ok = detail::check_color(color); !ok) return ok;
  if (auto ok = detail::check_filter(chroma); !ok) return ok;
  const std::uint32_t w = src.width;
  const std::uint32_t h = src.height;
  const auto constants = detail::yuv_constants(src.format, color.matrix, color.range);
  const detail::PlanesPush planes{0, 0, 0, 0, 0, 0, w, h};
  const std::uint32_t cw = chroma_width(src.format, w);
  const std::uint32_t ch = chroma_height(src.format, h);
  const kernels::ChromaShift shift = kernels::chroma_shift_up(color.siting);
  auto wx = compute_weights(static_cast<int>(cw), static_cast<int>(w), shift.x, chroma);
  if (!wx) return std::unexpected(wx.error());
  auto wy = compute_weights(static_cast<int>(ch), static_cast<int>(h), shift.y, chroma);
  if (!wy) return std::unexpected(wy.error());
  auto tx = detail::record_weights(recorder, *wx);
  if (!tx) return std::unexpected(tx.error());
  auto ty = detail::record_weights(recorder, *wy);
  if (!ty) return std::unexpected(ty.error());
  auto rows = recorder.scratch(tensor_bytes(w, ch, 2, Precision::F32));
  if (!rows) return std::unexpected(rows.error());
  auto full = recorder.scratch(tensor_bytes(w, h, 2, Precision::F32));
  if (!full) return std::unexpected(full.error());

  const detail::ChromaPush horizontal{0, 0, 0, 0, cw, ch, w, tx->taps};
  const std::array h_buffers{tx->start, tx->weights, *rows};
  const std::array chroma_plane{src.views[1]};
  if (auto ok = detail::run_planes(*pipelines_, recorder, "ingest_chroma_h_planes", Precision::F32, constants,
                                   h_buffers, chroma_plane, bytes_of(horizontal),
                                   {groups_for(w), groups_for(ch), 1});
      !ok) {
    return ok;
  }
  const detail::ResamplePush vertical{w, ch, h, ty->taps};
  const std::array v_buffers{*rows, ty->start, ty->weights, *full};
  if (auto ok = detail::run(*pipelines_, recorder, "resample_v", Precision::F32, {}, v_buffers, {},
                            bytes_of(vertical), {groups_for(w), groups_for(h), 2});
      !ok) {
    return ok;
  }
  const std::array buffers{*full, dst.buffer};
  return detail::run_planes(*pipelines_, recorder, "ingest_combine_planes", dst.precision, constants, buffers,
                            src.views, bytes_of(planes), {groups_for(w), groups_for(h), 1});
}

Done KernelRunner::rife_pad(Recorder& recorder, const YuvPlanes& src, const YuvView& padded) {
  if (auto ok = check_planes(src, "remplissage RIFE : plans de l'image"); !ok) return ok;
  if (auto ok = check_yuv(padded, "remplissage RIFE : image remplie"); !ok) return ok;
  if (padded.format != src.format) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("remplissage RIFE : formats {} et {} différents",
                              static_cast<int>(padded.format), static_cast<int>(src.format)));
  }
  if (!kernels::rife_fits(static_cast<int>(src.width), static_cast<int>(src.height),
                          static_cast<int>(padded.width), static_cast<int>(padded.height))) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("remplissage RIFE : {}x{} rempli, multiples de {} d'au moins {}x{} attendus",
                              padded.width, padded.height, kernels::kRifeAlign, src.width, src.height));
  }
  const kernels::RifeGeometry g =
      kernels::rife_window(static_cast<int>(src.width), static_cast<int>(src.height),
                           static_cast<int>(padded.width), static_cast<int>(padded.height));
  const kernels::FormatInfo info = kernels::info(src.format);
  const auto bytes = static_cast<std::uint32_t>(info.bytes);
  const std::uint32_t src_cw = chroma_width(src.format, src.width);
  const std::uint32_t dst_cw = chroma_width(padded.format, padded.width);
  const std::uint32_t src_ch = chroma_height(src.format, src.height);
  const std::uint32_t dst_ch = chroma_height(padded.format, padded.height);
  const auto constants = detail::yuv_constants(src.format, kernels::kRifeMatrix, kernels::Range::Limited);
  const std::array buffers{padded.buffer};
  // Luma depuis le plan 0, puis les deux échantillons par pixel de la chroma depuis le plan 1.
  const std::array<detail::YuvWindowPush, 2> pushes{
      detail::YuvWindowPush{0, 0, src.width, src.height, padded.layout.luma.offset, padded.layout.luma.pitch,
                            padded.width, padded.height, g.left, g.top, kEdgeLuma},
      detail::YuvWindowPush{0, 0, 2U * src_cw, src_ch, padded.layout.cb.offset, padded.layout.cb.pitch,
                            2U * dst_cw, dst_ch, 2 * (g.left / 2), g.top / 2, kEdgeChroma}};
  for (std::size_t i = 0; i < pushes.size(); ++i) {
    const detail::YuvWindowPush& p = pushes[i];
    const std::array plane{src.views[i]};
    if (auto ok = detail::run_planes(
            *pipelines_, recorder, "yuv_window_planes", Precision::F32, constants, buffers, plane,
            bytes_of(p), {groups_for((p.dst_width * bytes + 3U) / 4U), groups_for(p.dst_height), 1});
        !ok) {
      return ok;
    }
  }
  return {};
}

Done KernelRunner::scene_diff(Recorder& recorder, const YuvPlanes& a, const YuvPlanes& b,
                              const BufferView& partials) {
  if (auto ok = check_pair(a, b, "coupe de scène"); !ok) return ok;
  const std::uint32_t count = scene_partial_count(a.width, a.height);
  if (partials.size < VkDeviceSize{count} * sizeof(float)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("coupe de scène : {} o pour {} sommes partielles", partials.size, count));
  }
  const auto constants = detail::yuv_constants(a.format, kernels::Matrix::Bt709, kernels::Range::Limited);
  const detail::PairPush push{0, 0, 0, 0, a.width, a.height};
  const std::array buffers{partials};
  const std::array lumas{a.views[0], b.views[0]};
  return detail::run_planes(*pipelines_, recorder, "scene_diff_planes", Precision::F32, constants, buffers,
                            lumas, bytes_of(push), {groups_for(a.width), groups_for(a.height), 1});
}

Done KernelRunner::block_diff(Recorder& recorder, const YuvPlanes& a, const YuvPlanes& b,
                              const BufferView& result) {
  if (auto ok = check_pair(a, b, "dessins tenus"); !ok) return ok;
  const std::uint32_t count = block_count(a.width, a.height);
  if (result.size < (VkDeviceSize{count} + 1U) * sizeof(std::uint32_t)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("dessins tenus : {} o pour {} blocs et le pire", result.size, count));
  }
  if (auto ok =
          recorder.fill({.buffer = result.buffer, .offset = result.offset, .size = sizeof(std::uint32_t)}, 0);
      !ok) {
    return ok;
  }
  const auto constants = detail::yuv_constants(a.format, kernels::Matrix::Bt709, kernels::Range::Limited);
  const detail::PairPush push{0, 0, 0, 0, a.width, a.height};
  const std::array buffers{result};
  const std::array lumas{a.views[0], b.views[0]};
  return detail::run_planes(*pipelines_, recorder, "block_diff_planes", Precision::F32, constants, buffers,
                            lumas, bytes_of(push),
                            {groups_for(a.width, kHeldBlock), groups_for(a.height, kHeldBlock), 1});
}

}  // namespace ajp::gpu::vk
