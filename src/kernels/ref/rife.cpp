// Repris de animejanai-inference-rtx@118c272 src/aji_trt.cpp:1294-1385 (géométrie),
// 3700-3860 (bords, conversion, recadrage) et src/kernels.cu:629-644 (k_rife_consts).
#include "kernels/ref/rife.h"

#include <cstddef>
#include <format>
#include <utility>

#include "kernels/ref/check.h"
#include "kernels/ref/color.h"

namespace ajp::kernels::ref {
namespace {

template <class T>
Plane<T> place(const Plane<T>& src, int width, int height, int left, int top, T fill) {
  Plane<T> out(width, height, fill);
  for (int y = 0; y < src.height; ++y) {
    for (int x = 0; x < src.width; ++x) out.at(left + x, top + y) = src.at(x, y);
  }
  return out;
}

template <class T>
Plane<T> window(const Plane<T>& src, int left, int top, int width, int height) {
  Plane<T> out(width, height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) out.at(x, y) = src.at(left + x, top + y);
  }
  return out;
}

// La sortie remplie doit avoir la taille remplie de la géométrie, et la géométrie être une fenêtre
// de rife_window (multiple de 64 ou 128 selon le modèle) : un recadrage ne sort jamais du plan.
Result<void> check_geometry(int width, int height, const RifeGeometry& g) {
  const RifeGeometry canonical = rife_window(g.width, g.height, g.padded_width, g.padded_height);
  if (!valid_plane_dimension(g.width) || !valid_plane_dimension(g.height) ||
      !rife_fits(g.width, g.height, g.padded_width, g.padded_height) || g.left != canonical.left ||
      g.top != canonical.top) {
    return fail(ErrorCode::ParameterOutOfRange,
                std::format("géométrie RIFE {}x{} incohérente", g.width, g.height));
  }
  if (width != g.padded_width || height != g.padded_height) {
    return fail(ErrorCode::SizeMismatch, std::format("sortie RIFE {}x{}, {}x{} attendu", width, height,
                                                     g.padded_width, g.padded_height));
  }
  return {};
}

ColorSpec rife_color(Range range) {
  return {.matrix = kRifeMatrix, .range = range, .siting = kRifeSiting};
}

}  // namespace

Result<YuvImage> rife_pad(const YuvImage& frame) {
  if (auto ok = check_yuv(frame); !ok) return std::unexpected(ok.error());
  const RifeGeometry g = rife_geometry(frame.width(), frame.height());
  const std::uint16_t black = rife_pad_luma(frame.format);
  const std::uint16_t neutral = rife_pad_chroma(frame.format);
  YuvImage out;
  out.format = frame.format;
  out.y = place(frame.y, g.padded_width, g.padded_height, g.left, g.top, black);
  if (info(frame.format).subsampled) {
    // left et top sont pairs : la chroma reste alignée.
    const int cw = g.padded_width / 2;
    const int ch = g.padded_height / 2;
    out.u = place(frame.u, cw, ch, g.left / 2, g.top / 2, neutral);
    out.v = place(frame.v, cw, ch, g.left / 2, g.top / 2, neutral);
  } else {
    out.u = place(frame.u, g.padded_width, g.padded_height, g.left, g.top, neutral);
    out.v = place(frame.v, g.padded_width, g.padded_height, g.left, g.top, neutral);
  }
  return out;
}

Result<RgbImage> rife_crop(const RgbImage& padded, const RifeGeometry& geometry) {
  if (auto ok = check_rgb(padded, "sortie RIFE"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_geometry(padded.width(), padded.height(), geometry); !ok)
    return std::unexpected(ok.error());
  RgbImage out;
  for (std::size_t c = 0; c < out.planes.size(); ++c) {
    out.planes[c] = window(padded.planes[c], geometry.left, geometry.top, geometry.width, geometry.height);
  }
  return out;
}

Result<YuvImage> rife_crop(const YuvImage& padded, const RifeGeometry& geometry) {
  if (auto ok = check_yuv(padded); !ok) return std::unexpected(ok.error());
  if (auto ok = check_geometry(padded.width(), padded.height(), geometry); !ok)
    return std::unexpected(ok.error());
  const bool sub = info(padded.format).subsampled;
  if (sub) {
    if (auto ok = check_frame(geometry.width, geometry.height, "fenêtre 4:2:0"); !ok) {
      return std::unexpected(ok.error());
    }
  }
  const int d = sub ? 2 : 1;
  YuvImage out;
  out.format = padded.format;
  out.y = window(padded.y, geometry.left, geometry.top, geometry.width, geometry.height);
  out.u = window(padded.u, geometry.left / d, geometry.top / d, geometry.width / d, geometry.height / d);
  out.v = window(padded.v, geometry.left / d, geometry.top / d, geometry.width / d, geometry.height / d);
  return out;
}

Result<RgbImage> rife_prepare(const YuvImage& frame, Range range) {
  return rife_pad(frame).and_then(
      [range](const YuvImage& padded) { return yuv_to_rgb(padded, rife_color(range), kRifeChromaFilter); });
}

Result<YuvImage> rife_finish(const RgbImage& padded, Format format, Range range,
                             const RifeGeometry& geometry) {
  if (auto ok = check_rgb(padded, "sortie RIFE"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_geometry(padded.width(), padded.height(), geometry); !ok)
    return std::unexpected(ok.error());
  return rgb_to_yuv(padded, format, rife_color(range), kRifeChromaFilter)
      .and_then([&geometry](const YuvImage& yuv) { return rife_crop(yuv, geometry); });
}

Result<std::array<PlaneF, 4>> rife_constant_planes(int padded_width, int padded_height) {
  for (const int n : {padded_width, padded_height}) {
    if (n < kRifeAlign || n > kMaxDimension || n % kRifeAlign != 0) {
      return fail(ErrorCode::DimensionOutOfRange, std::format("entrée RIFE {}x{} : multiples de {} attendus",
                                                              padded_width, padded_height, kRifeAlign));
    }
  }
  std::array<PlaneF, 4> planes{PlaneF(padded_width, padded_height), PlaneF(padded_width, padded_height),
                               PlaneF(padded_width, padded_height, rife_mesh_step(padded_width)),
                               PlaneF(padded_width, padded_height, rife_mesh_step(padded_height))};
  for (int y = 0; y < padded_height; ++y) {
    for (int x = 0; x < padded_width; ++x) {
      planes[0].at(x, y) = rife_mesh(x, padded_width);
      planes[1].at(x, y) = rife_mesh(y, padded_height);
    }
  }
  return planes;
}

Result<std::vector<PlaneF>> rife_input(const RgbImage& a, const RgbImage& b, float t) {
  if (auto ok = check_same_size(a, b, "images de la paire RIFE"); !ok) return std::unexpected(ok.error());
  if (!(t >= 0.0F && t <= 1.0F)) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("instant RIFE {} hors de [0, 1]", t));
  }
  auto constants = rife_constant_planes(a.width(), a.height());
  if (!constants) return std::unexpected(constants.error());
  std::vector<PlaneF> input;
  input.reserve(kRifeInputChannels);
  for (const PlaneF& p : a.planes) input.push_back(p);
  for (const PlaneF& p : b.planes) input.push_back(p);
  input.emplace_back(a.width(), a.height(), t);
  for (PlaneF& p : *constants) input.push_back(std::move(p));
  return input;
}

}  // namespace ajp::kernels::ref
