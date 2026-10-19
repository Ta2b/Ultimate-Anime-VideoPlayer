#include "gpu/cuda/compute/yuv.h"

#include <format>
#include <utility>

#include "gpu/cuda/compute/kernels/launch.h"

namespace ajp::gpu::cuda {
namespace {

using kernels::Format;

dev::YuvToRgb yuv_to_rgb_constants(Format format, const ColorSpec& color) {
  const kernels::YuvScale s = kernels::yuv_scale(format, color.range);
  const kernels::Coefficients k = kernels::coefficients(color.matrix);
  return {.yoff = s.yoff, .yscale = s.yscale, .coff = s.coff, .cscale = s.cscale, .kr = k.kr, .kb = k.kb};
}

dev::YuvTarget target(const YuvPlanes& dst, const kernels::Window& window) {
  const kernels::FormatInfo info = kernels::info(dst.format);
  return {.y = dst.planes[0],
          .u = dst.planes[1],
          .v = dst.planes[2],
          .sample = sample_of(dst.format),
          .interleaved = info.interleaved,
          .subsampled = info.subsampled,
          .left = window.x,
          .top = window.y,
          .width = window.width,
          .height = window.height};
}

Result<> launched(cudaError_t error, std::string_view kernel) {
  return cuda_check(error, std::format("lancement de {}", kernel));
}

}  // namespace

IngestSpec rife_ingest_spec(Format format, int width, int height, kernels::Range range) {
  return {.format = format,
          .width = width,
          .height = height,
          .color = {.matrix = kernels::kRifeMatrix, .range = range, .siting = kernels::kRifeSiting},
          .chroma = kernels::kRifeChromaFilter,
          .rife_pad = true};
}

Result<IngestPlan> IngestPlan::create(DevicePool& pool, const IngestSpec& spec, cudaStream_t stream) {
  const kernels::FormatInfo info = kernels::info(spec.format);
  if (info.subsampled) {
    if (auto ok = check_frame(spec.width, spec.height, "ingestion 4:2:0"); !ok)
      return std::unexpected(ok.error());
  } else if (!kernels::valid_plane_dimension(spec.width) || !kernels::valid_plane_dimension(spec.height)) {
    return refuse(RefusalCode::ShapeInvalid, std::format("ingestion : {}x{} hors de [1, {}]", spec.width,
                                                         spec.height, kernels::kMaxDimension));
  }
  IngestPlan plan;
  plan.spec_ = spec;
  plan.out_width_ = spec.width;
  plan.out_height_ = spec.height;
  if (spec.rife_pad) {
    const kernels::RifeGeometry g = kernels::rife_geometry(spec.width, spec.height);
    if (g.padded_width > kernels::kMaxDimension || g.padded_height > kernels::kMaxDimension) {
      return refuse(RefusalCode::ShapeInvalid, std::format("RIFE : {}x{} rempli dépasse {}", spec.width,
                                                           spec.height, kernels::kMaxDimension));
    }
    plan.out_width_ = g.padded_width;
    plan.out_height_ = g.padded_height;
    plan.left_ = g.left;
    plan.top_ = g.top;
  }
  plan.matrix_ = yuv_to_rgb_constants(spec.format, spec.color);
  if (!info.subsampled) return plan;

  const kernels::ChromaShift shift = kernels::chroma_shift_up(spec.color.siting);
  auto wx = DeviceWeights::create(pool, plan.out_width_ / 2, plan.out_width_, shift.x, spec.chroma, stream);
  if (!wx) return std::unexpected(wx.error());
  auto wy = DeviceWeights::create(pool, plan.out_height_ / 2, plan.out_height_, shift.y, spec.chroma, stream);
  if (!wy) return std::unexpected(wy.error());
  const std::size_t tmp =
      2 * static_cast<std::size_t>(plan.out_width_) * static_cast<std::size_t>(plan.out_height_ / 2);
  auto buffer = pool.allocate(tmp * sizeof(float), "ingestion : chroma après la passe horizontale");
  if (!buffer) return std::unexpected(buffer.error());
  plan.wx_ = std::move(*wx);
  plan.wy_ = std::move(*wy);
  plan.tmp_ = std::move(*buffer);
  return plan;
}

Result<> IngestPlan::run(const YuvPlanes& src, const Tensor& rgb, cudaStream_t stream) const {
  if (auto ok = check_yuv(src, spec_.format, spec_.width, spec_.height, "ingestion"); !ok) return ok;
  if (auto ok = check_tensor(rgb, out_width_, out_height_, 3, "ingestion"); !ok) return ok;
  const kernels::FormatInfo info = kernels::info(spec_.format);
  const dev::YuvSource source{.y = src.planes[0],
                              .u = src.planes[1],
                              .v = src.planes[2],
                              .sample = sample_of(spec_.format),
                              .interleaved = info.interleaved,
                              .subsampled = info.subsampled,
                              .width = spec_.width,
                              .height = spec_.height,
                              .left = left_,
                              .top = top_,
                              .padded_width = out_width_,
                              .padded_height = out_height_,
                              .pad_luma = kernels::rife_pad_luma(spec_.format),
                              .pad_chroma = kernels::rife_pad_chroma(spec_.format)};
  if (!info.subsampled)
    return launched(dev::ingest_combine_444(source, matrix_, rgb, stream), "ingest_combine_444");
  if (auto ok = launched(dev::ingest_chroma_rows(source, wx_.view(), tmp_.as<float>(), stream),
                         "ingest_chroma_rows");
      !ok) {
    return ok;
  }
  return launched(dev::ingest_combine_420(source, tmp_.as<const float>(), wy_.view(), matrix_, rgb, stream),
                  "ingest_combine_420");
}

OutputSpec rife_output_spec(Format format, kernels::Range range, const kernels::RifeGeometry& geometry) {
  return {
      .format = format,
      .width = geometry.padded_width,
      .height = geometry.padded_height,
      .color = {.matrix = kernels::kRifeMatrix, .range = range, .siting = kernels::kRifeSiting},
      .chroma = kernels::kRifeChromaFilter,
      .window = {.x = geometry.left, .y = geometry.top, .width = geometry.width, .height = geometry.height}};
}

Result<OutputPlan> OutputPlan::create(DevicePool& pool, const OutputSpec& spec, cudaStream_t stream) {
  const kernels::FormatInfo info = kernels::info(spec.format);
  OutputPlan plan;
  plan.spec_ = spec;
  kernels::Window& w = plan.spec_.window;
  if (w.width == 0 && w.height == 0) w = {.x = 0, .y = 0, .width = spec.width, .height = spec.height};
  if (info.subsampled) {
    if (auto ok = check_frame(spec.width, spec.height, "sortie 4:2:0"); !ok)
      return std::unexpected(ok.error());
    if (auto ok = check_frame(w.width, w.height, "fenêtre 4:2:0"); !ok) return std::unexpected(ok.error());
  } else if (!kernels::valid_plane_dimension(spec.width) || !kernels::valid_plane_dimension(spec.height)) {
    return refuse(RefusalCode::ShapeInvalid, std::format("sortie : {}x{} hors de [1, {}]", spec.width,
                                                         spec.height, kernels::kMaxDimension));
  }
  // Écrit sans w.x + w.width, qui pourrait déborder un int.
  const bool odd_corner = info.subsampled && (w.x % 2 != 0 || w.y % 2 != 0);
  if (w.x < 0 || w.y < 0 || w.width < 1 || w.height < 1 || w.width > spec.width - w.x ||
      w.height > spec.height - w.y || odd_corner) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("sortie : fenêtre {}x{} en ({}, {}) hors de {}x{}", w.width, w.height, w.x, w.y,
                              spec.width, spec.height));
  }
  const kernels::Quantizer q = kernels::quantizer(spec.format);
  plan.matrix_ = {.scale = yuv_to_rgb_constants(spec.format, spec.color), .qdiv = q.qdiv, .qmax = q.qmax};
  if (!info.subsampled) return plan;

  const kernels::ChromaShift shift = kernels::chroma_shift_down(spec.color.siting);
  auto wx = DeviceWeights::create(pool, spec.width, spec.width / 2, shift.x, spec.chroma, stream);
  if (!wx) return std::unexpected(wx.error());
  auto wy = DeviceWeights::create(pool, spec.height, spec.height / 2, shift.y, spec.chroma, stream);
  if (!wy) return std::unexpected(wy.error());
  const std::size_t area = static_cast<std::size_t>(spec.width) * static_cast<std::size_t>(spec.height);
  auto tmp = pool.allocate(2 * area * sizeof(float), "sortie : Cb et Cr normalisés");
  if (!tmp) return std::unexpected(tmp.error());
  auto rows = pool.allocate(area * sizeof(float), "sortie : chroma après la passe horizontale");
  if (!rows) return std::unexpected(rows.error());
  plan.wx_ = std::move(*wx);
  plan.wy_ = std::move(*wy);
  plan.tmp_ = std::move(*tmp);
  plan.rows_ = std::move(*rows);
  return plan;
}

Result<> OutputPlan::run(const Tensor& rgb, const YuvPlanes& dst, cudaStream_t stream) const {
  const kernels::Window& w = spec_.window;
  if (auto ok = check_tensor(rgb, spec_.width, spec_.height, 3, "sortie"); !ok) return ok;
  if (auto ok = check_yuv(dst, spec_.format, w.width, w.height, "sortie"); !ok) return ok;
  const dev::YuvTarget out = target(dst, w);
  if (!kernels::info(spec_.format).subsampled) {
    return launched(dev::output_matrix_444(rgb, matrix_, out, stream), "output_matrix_444");
  }
  if (auto ok =
          launched(dev::output_matrix_420(rgb, matrix_, out, tmp_.as<float>(), stream), "output_matrix_420");
      !ok) {
    return ok;
  }
  const Tensor chroma{
      .ptr = tmp_.address(), .elem = Elem::F32, .width = spec_.width, .height = spec_.height, .channels = 2};
  if (auto ok = launched(dev::rows(chroma, 2, wx_.view(), rows_.as<float>(), stream), "rows (chroma)"); !ok) {
    return ok;
  }
  return launched(
      dev::output_chroma_store(rows_.as<const float>(), spec_.width / 2, wy_.view(), matrix_, out, stream),
      "output_chroma_store");
}

Result<> write_rgba16(const Tensor& rgb, const Plane& dst, cudaStream_t stream) {
  if (auto ok = check_tensor(rgb, rgb.width, rgb.height, 3, "RGBA16"); !ok) return ok;
  const std::size_t row = static_cast<std::size_t>(rgb.width) * 8;
  if (dst.ptr == 0 || dst.ptr % 8 != 0 || dst.pitch % 8 != 0 || dst.pitch < row) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("RGBA16 : destination {:#x} de pas {} pour des lignes de {} "
                              "octets",
                              dst.ptr, dst.pitch, row));
  }
  return launched(dev::rgba16_to_linear(rgb, dst, stream), "rgba16_to_linear");
}

}  // namespace ajp::gpu::cuda
