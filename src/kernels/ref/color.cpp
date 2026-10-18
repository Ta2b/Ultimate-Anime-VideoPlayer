// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:99-313 et src/resample.h:86-129.
#include "kernels/ref/color.h"

#include <cstddef>
#include <format>
#include <string_view>
#include <utility>

#include "kernels/ref/check.h"
#include "kernels/ref/resample.h"

namespace ajp::kernels::ref {
namespace {

PlaneF to_float(const PlaneU16& p) {
  PlaneF out(p.width, p.height);
  for (std::size_t i = 0; i < p.data.size(); ++i) out.data[i] = static_cast<float>(p.data[i]);
  return out;
}

int chroma_size(Format f, int n) {
  return info(f).subsampled ? n / 2 : n;
}

Result<void> check_samples(const PlaneU16& p, Format f, std::string_view what) {
  const std::uint32_t low_bits = (1U << info(f).shift) - 1U;
  for (const std::uint16_t code : p.data) {
    if (code > max_code(f) || (code & low_bits) != 0) {
      return fail(ErrorCode::SampleOutOfRange, std::format("{} : code {} hors du conteneur", what, code));
    }
  }
  return {};
}

// Un plan de chroma 4:2:0 passé à la taille de la luma, ou l'inverse : horizontal puis vertical.
Result<PlaneF> resample_chroma(const PlaneF& c, int width, int height, ChromaShift shift, Filter filter) {
  const auto wx = compute_weights(c.width, width, shift.x, filter);
  if (!wx) return std::unexpected(wx.error());
  const auto wy = compute_weights(c.height, height, shift.y, filter);
  if (!wy) return std::unexpected(wy.error());
  return resample_rows(c, *wx).and_then([&wy](const PlaneF& rows) { return resample_columns(rows, *wy); });
}

}  // namespace

Result<void> check_yuv(const YuvImage& image) {
  const int w = image.width();
  const int h = image.height();
  if (info(image.format).subsampled) {
    if (auto ok = check_frame(w, h, "image 4:2:0"); !ok) return ok;
  } else if (auto ok = check_plane(image.y, "image 4:4:4"); !ok) {
    return ok;
  }
  const PlaneU16 chroma_shape(chroma_size(image.format, w), chroma_size(image.format, h));
  if (auto ok = check_same_size(image.u, chroma_shape, "plan Cb"); !ok) return ok;
  if (auto ok = check_same_size(image.v, chroma_shape, "plan Cr"); !ok) return ok;
  for (const PlaneU16* p : {&image.y, &image.u, &image.v}) {
    if (auto ok = check_samples(*p, image.format, "échantillon YUV"); !ok) return ok;
  }
  return {};
}

Result<RgbImage> yuv_to_rgb(const YuvImage& in, const ColorSpec& color, Filter chroma) {
  if (auto ok = check_yuv(in); !ok) return std::unexpected(ok.error());
  const int w = in.width();
  const int h = in.height();
  PlaneF u = to_float(in.u);
  PlaneF v = to_float(in.v);
  if (info(in.format).subsampled) {
    // La chroma se rééchantillonne en codes bruts, avant la normalisation (k_uv_h, k_v_f32).
    const ChromaShift shift = chroma_shift_up(color.siting);
    auto up_u = resample_chroma(u, w, h, shift, chroma);
    if (!up_u) return std::unexpected(up_u.error());
    auto up_v = resample_chroma(v, w, h, shift, chroma);
    if (!up_v) return std::unexpected(up_v.error());
    u = std::move(*up_u);
    v = std::move(*up_v);
  }

  const YuvScale scale = yuv_scale(in.format, color.range);
  const Coefficients k = coefficients(color.matrix);
  RgbImage out(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Rgb p = to_rgb(normalize(static_cast<float>(in.y.at(x, y)), u.at(x, y), v.at(x, y), scale), k);
      out.planes[0].at(x, y) = p.r;
      out.planes[1].at(x, y) = p.g;
      out.planes[2].at(x, y) = p.b;
    }
  }
  return out;
}

Result<YuvImage> rgb_to_yuv(const RgbImage& in, Format format, const ColorSpec& color, Filter chroma) {
  if (auto ok = check_rgb(in, "image RGB à convertir"); !ok) return std::unexpected(ok.error());
  const int w = in.width();
  const int h = in.height();
  if (info(format).subsampled) {
    if (auto ok = check_frame(w, h, "sortie 4:2:0"); !ok) return std::unexpected(ok.error());
  }

  const YuvScale scale = yuv_scale(format, color.range);
  const Coefficients k = coefficients(color.matrix);
  YuvImage out(format, w, h, 0, 0, 0);
  PlaneF u(w, h);
  PlaneF v(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Yuv p = to_yuv(in.at(x, y), k);
      out.y.at(x, y) = encode_luma(p.y, format, scale);
      u.at(x, y) = p.u;
      v.at(x, y) = p.v;
    }
  }
  if (info(format).subsampled) {
    // Cb et Cr normalisés se rééchantillonnent avant la quantification (k_h_f32, k_uv_v_store).
    const ChromaShift shift = chroma_shift_down(color.siting);
    auto down_u = resample_chroma(u, w / 2, h / 2, shift, chroma);
    if (!down_u) return std::unexpected(down_u.error());
    auto down_v = resample_chroma(v, w / 2, h / 2, shift, chroma);
    if (!down_v) return std::unexpected(down_v.error());
    u = std::move(*down_u);
    v = std::move(*down_v);
  }
  for (std::size_t i = 0; i < u.data.size(); ++i) {
    out.u.data[i] = encode_chroma(u.data[i], format, scale);
    out.v.data[i] = encode_chroma(v.data[i], format, scale);
  }
  return out;
}

Result<std::vector<std::uint16_t>> to_rgba16(const RgbImage& in) {
  if (auto ok = check_rgb(in, "image RGBA16"); !ok) return std::unexpected(ok.error());
  std::vector<std::uint16_t> out;
  out.reserve(in.planes[0].data.size() * 4);
  for (std::size_t i = 0; i < in.planes[0].data.size(); ++i) {
    for (const PlaneF& p : in.planes) out.push_back(unorm16(p.data[i]));
    out.push_back(kOpaqueAlpha16);
  }
  return out;
}

}  // namespace ajp::kernels::ref
