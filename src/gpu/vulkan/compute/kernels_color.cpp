// Ingestion YUV → RGB, sortie RGB → YUV et RGBA16, rééchantillonnage séparable. L'ordre des
// passes est celui de kernels/spec/resample.h et de kernels/ref/color.cpp : la chroma d'entrée se
// rééchantillonne en codes bruts avant la matrice ; celle de sortie, normalisée, après.
#include <array>
#include <format>

#include "gpu/vulkan/compute/kernel_dispatch.h"
#include "gpu/vulkan/compute/kernel_runner.h"

namespace ajp::gpu::vk {

using detail::bytes_of;
using detail::groups_for;

Done KernelRunner::ingest(Recorder& recorder, const YuvView& src, const ColorSpec& color,
                          const TensorView& dst, kernels::Filter chroma) {
  if (auto ok = check_yuv(src, "ingestion : image YUV"); !ok) return ok;
  if (auto ok = check_tensor(dst, 3, context_->caps().enabled, "ingestion : tenseur RGB"); !ok) return ok;
  if (auto ok = check_same_size(src.width, src.height, dst.width, dst.height, "ingestion"); !ok) return ok;
  if (auto ok = detail::check_color(color); !ok) return ok;
  if (auto ok = detail::check_filter(chroma); !ok) return ok;
  const std::uint32_t w = src.width;
  const std::uint32_t h = src.height;
  const auto constants = detail::yuv_constants(src.format, color.matrix, color.range);
  const YuvLayout& l = src.layout;
  const detail::PlanesPush planes{l.luma.offset, l.luma.pitch, l.cb.offset, l.cb.pitch,
                                  l.cr.offset,   l.cr.pitch,   w,           h};
  const std::array<std::uint32_t, 3> groups{groups_for(w), groups_for(h), 1};
  if (!kernels::info(src.format).subsampled) {
    // 4:4:4 : la chroma se lit telle quelle ; le binding 1 est lié sans être lu.
    const std::array buffers{src.buffer, src.buffer, dst.buffer};
    return detail::run(*pipelines_, recorder, "ingest_combine", dst.precision, constants, buffers, {},
                       bytes_of(planes), groups);
  }

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

  const detail::ChromaPush horizontal{l.cb.offset, l.cb.pitch, l.cr.offset, l.cr.pitch, cw, ch, w, tx->taps};
  const std::array h_buffers{src.buffer, tx->start, tx->weights, *rows};
  if (auto ok = detail::run(*pipelines_, recorder, "ingest_chroma_h", Precision::F32, constants, h_buffers,
                            {}, bytes_of(horizontal), {groups_for(w), groups_for(ch), 1});
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
  const std::array buffers{src.buffer, *full, dst.buffer};
  return detail::run(*pipelines_, recorder, "ingest_combine", dst.precision, constants, buffers, {},
                     bytes_of(planes), groups);
}

Done KernelRunner::output_yuv(Recorder& recorder, const TensorView& src, const ColorSpec& color,
                              const YuvView& dst, kernels::Filter chroma) {
  if (auto ok = check_tensor(src, 3, context_->caps().enabled, "sortie YUV : tenseur RGB"); !ok) return ok;
  if (auto ok = check_yuv(dst, "sortie YUV : image YUV"); !ok) return ok;
  if (auto ok = check_same_size(src.width, src.height, dst.width, dst.height, "sortie YUV"); !ok) return ok;
  if (auto ok = detail::check_color(color); !ok) return ok;
  if (auto ok = detail::check_filter(chroma); !ok) return ok;
  const std::uint32_t w = dst.width;
  const std::uint32_t h = dst.height;
  const kernels::FormatInfo info = kernels::info(dst.format);
  const auto bytes = static_cast<std::uint32_t>(info.bytes);
  const auto constants = detail::yuv_constants(dst.format, color.matrix, color.range);
  const YuvLayout& l = dst.layout;
  const detail::PlanesPush planes{l.luma.offset, l.luma.pitch, l.cb.offset, l.cb.pitch,
                                  l.cr.offset,   l.cr.pitch,   w,           h};
  // Une invocation par mot de 32 bits d'une ligne de luma.
  const std::array<std::uint32_t, 3> groups{groups_for((w * bytes + 3U) / 4U), groups_for(h), 1};
  if (!info.subsampled) {
    const std::array buffers{src.buffer, dst.buffer, dst.buffer};
    return detail::run(*pipelines_, recorder, "output_matrix", src.precision, constants, buffers, {},
                       bytes_of(planes), groups);
  }

  const std::uint32_t cw = chroma_width(dst.format, w);
  const std::uint32_t ch = chroma_height(dst.format, h);
  const kernels::ChromaShift shift = kernels::chroma_shift_down(color.siting);
  auto wx = compute_weights(static_cast<int>(w), static_cast<int>(cw), shift.x, chroma);
  if (!wx) return std::unexpected(wx.error());
  auto wy = compute_weights(static_cast<int>(h), static_cast<int>(ch), shift.y, chroma);
  if (!wy) return std::unexpected(wy.error());
  auto tx = detail::record_weights(recorder, *wx);
  if (!tx) return std::unexpected(tx.error());
  auto ty = detail::record_weights(recorder, *wy);
  if (!ty) return std::unexpected(ty.error());
  auto normalized = recorder.scratch(tensor_bytes(w, h, 2, Precision::F32));
  if (!normalized) return std::unexpected(normalized.error());
  auto narrow = recorder.scratch(tensor_bytes(cw, h, 2, Precision::F32));
  if (!narrow) return std::unexpected(narrow.error());

  const std::array m_buffers{src.buffer, dst.buffer, *normalized};
  if (auto ok = detail::run(*pipelines_, recorder, "output_matrix", src.precision, constants, m_buffers, {},
                            bytes_of(planes), groups);
      !ok) {
    return ok;
  }
  const detail::ResamplePush horizontal{w, cw, h, tx->taps};
  const std::array h_buffers{*normalized, tx->start, tx->weights, *narrow};
  if (auto ok = detail::run(*pipelines_, recorder, "resample_h", Precision::F32, {}, h_buffers, {},
                            bytes_of(horizontal), {groups_for(cw), groups_for(h), 2});
      !ok) {
    return ok;
  }
  const std::uint32_t samples = info.interleaved ? 2U * cw : cw;
  const detail::ChromaPush vertical{l.cb.offset, l.cb.pitch, l.cr.offset, l.cr.pitch, cw, h, ch, ty->taps};
  const std::array v_buffers{*narrow, ty->start, ty->weights, dst.buffer};
  return detail::run(*pipelines_, recorder, "output_chroma_v", Precision::F32, constants, v_buffers, {},
                     bytes_of(vertical), {groups_for((samples * bytes + 3U) / 4U), groups_for(ch), 1});
}

Done KernelRunner::output_rgba16(Recorder& recorder, const TensorView& src, const StorageImage& dst) {
  if (auto ok = check_tensor(src, 3, context_->caps().enabled, "sortie RGBA16 : tenseur RGB"); !ok) return ok;
  if (dst.context() != context_) {
    return refuse(RefusalCode::InputMismatch, "sortie RGBA16 : image vide ou d'un autre contexte Vulkan");
  }
  if (auto ok = check_same_size(src.width, src.height, dst.width(), dst.height(), "sortie RGBA16"); !ok)
    return ok;
  const detail::SizePush size{src.width, src.height};
  const std::array buffers{src.buffer};
  const std::array images{dst.storage_view()};
  return detail::run(*pipelines_, recorder, "rgba16_store", src.precision, {}, buffers, images,
                     bytes_of(size), {groups_for(src.width), groups_for(src.height), 1});
}

Done KernelRunner::resize(Recorder& recorder, const TensorView& src, const TensorView& dst,
                          kernels::Filter filter) {
  const Features& enabled = context_->caps().enabled;
  if (auto ok = check_tensor(src, src.planes == 0 ? 1 : src.planes, enabled, "rééchantillonnage : source");
      !ok) {
    return ok;
  }
  if (auto ok = check_tensor(dst, src.planes, enabled, "rééchantillonnage : cible"); !ok) return ok;
  for (const std::uint32_t n : {src.width, src.height, dst.width, dst.height}) {
    if (!kernels::valid_frame_dimension(static_cast<int>(n))) {
      return refuse(
          RefusalCode::InputMismatch,
          std::format("rééchantillonnage {}x{} vers {}x{} : dimensions paires dans [2, {}] attendues",
                      src.width, src.height, dst.width, dst.height, kernels::kMaxDimension));
    }
  }
  if (auto ok = detail::check_filter(filter); !ok) return ok;
  auto wx = compute_weights(static_cast<int>(src.width), static_cast<int>(dst.width), 0.0, filter);
  if (!wx) return std::unexpected(wx.error());
  auto wy = compute_weights(static_cast<int>(src.height), static_cast<int>(dst.height), 0.0, filter);
  if (!wy) return std::unexpected(wy.error());
  auto tx = detail::record_weights(recorder, *wx);
  if (!tx) return std::unexpected(tx.error());
  auto ty = detail::record_weights(recorder, *wy);
  if (!ty) return std::unexpected(ty.error());
  auto rows = recorder.scratch(tensor_bytes(dst.width, src.height, src.planes, Precision::F32));
  if (!rows) return std::unexpected(rows.error());

  const detail::ResamplePush horizontal{src.width, dst.width, src.height, tx->taps};
  const std::array h_buffers{src.buffer, tx->start, tx->weights, *rows};
  if (auto ok =
          detail::run(*pipelines_, recorder, "resample_h", src.precision, {}, h_buffers, {},
                      bytes_of(horizontal), {groups_for(dst.width), groups_for(src.height), src.planes});
      !ok) {
    return ok;
  }
  const detail::ResamplePush vertical{dst.width, src.height, dst.height, ty->taps};
  const std::array v_buffers{*rows, ty->start, ty->weights, dst.buffer};
  return detail::run(*pipelines_, recorder, "resample_v", dst.precision, {}, v_buffers, {},
                     bytes_of(vertical), {groups_for(dst.width), groups_for(dst.height), src.planes});
}

}  // namespace ajp::gpu::vk
