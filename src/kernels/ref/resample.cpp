// Repris de animejanai-inference-rtx@118c272 src/resample.h:53-84 (compute_filter de zimg),
// src/kernels.cu:415-445 (passes) et src/pipeline_kernels.cu:34-50 (empreinte de Area).
#include "kernels/ref/resample.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <string_view>
#include <utility>

#include "kernels/ref/check.h"

namespace ajp::kernels::ref {
namespace {

std::size_t at(int i) {
  return static_cast<std::size_t>(i);
}

double kernel(Filter filter, double x) {
  return filter == Filter::Bilinear ? bilinear(x) : spline36(x);
}

// Spline36 et bilinéaire : compute_filter de zimg.
Weights kernel_weights(int src, int dst, double shift, Filter filter) {
  const double scale = static_cast<double>(dst) / static_cast<double>(src);
  const double step = std::min(scale, 1.0);
  const int taps = std::max(static_cast<int>(std::ceil(support(filter) / step)) * 2, 1);

  Weights w{.taps = taps, .src = src, .dst = dst, .start = std::vector<int>(at(dst)), .w = {}};
  w.w.resize(at(dst) * at(taps));
  std::vector<double> row(at(taps));
  for (int i = 0; i < dst; ++i) {
    const double pos = (i + 0.5) / scale + shift;
    const double begin = std::floor(pos - taps / 2.0 + 0.5) + 0.5;
    double sum = 0.0;
    for (int j = 0; j < taps; ++j) {
      row[at(j)] = kernel(filter, (begin + j - pos) * step);
      sum += row[at(j)];
    }
    w.start[at(i)] = static_cast<int>(begin - 0.5);
    for (int j = 0; j < taps; ++j) w.w[at(i) * at(taps) + at(j)] = static_cast<float>(row[at(j)] / sum);
  }
  return w;
}

// Area : recouvrement exact de l'empreinte [i src / dst, (i + 1) src / dst].
Weights area_weights(int src, int dst) {
  const auto footprint = [src, dst](int i) { return static_cast<double>(i) * src / dst; };
  int taps = 1;
  for (int i = 0; i < dst; ++i) {
    const int x0 = static_cast<int>(std::floor(footprint(i)));
    const int x1 = std::min(src, static_cast<int>(std::ceil(footprint(i + 1))));
    taps = std::max(taps, x1 - x0);
  }

  Weights w{.taps = taps, .src = src, .dst = dst, .start = std::vector<int>(at(dst)), .w = {}};
  w.w.assign(at(dst) * at(taps), 0.0F);
  for (int i = 0; i < dst; ++i) {
    const double a = footprint(i);
    const double b = footprint(i + 1);
    const int x0 = static_cast<int>(std::floor(a));
    const int x1 = std::min(src, static_cast<int>(std::ceil(b)));
    w.start[at(i)] = x0;
    for (int sx = x0; sx < x1; ++sx) {
      const double overlap = std::min(b, sx + 1.0) - std::max(a, static_cast<double>(sx));
      w.w[at(i) * at(taps) + at(sx - x0)] = static_cast<float>(overlap / (b - a));
    }
  }
  return w;
}

// Une table cohérente (ses champs sont publics) et faite pour ce plan : sans cela, une passe
// lirait hors de la table.
Result<void> check_weights(const Weights& w, int src, std::string_view what) {
  if (!valid_plane_dimension(w.dst) || w.taps < 1 || w.start.size() != at(w.dst) ||
      w.w.size() != at(w.dst) * at(w.taps)) {
    return fail(ErrorCode::SizeMismatch,
                std::format("{} : table de {} sorties × {} prises incohérente", what, w.dst, w.taps));
  }
  if (w.src != src) {
    return fail(ErrorCode::SizeMismatch,
                std::format("{} : table pour {} échantillons, plan de {}", what, w.src, src));
  }
  return {};
}

}  // namespace

std::span<const float> Weights::row(int i) const {
  return std::span(w).subspan(at(i) * at(taps), at(taps));
}

Result<Weights> compute_weights(int src, int dst, double shift, Filter filter) {
  if (!valid_plane_dimension(src) || !valid_plane_dimension(dst)) {
    return fail(ErrorCode::DimensionOutOfRange,
                std::format("poids : {} vers {} hors de [1, {}]", src, dst, kMaxDimension));
  }
  if (!std::isfinite(shift) || std::abs(shift) > 1.0) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("poids : décalage {} hors de [-1, 1]", shift));
  }
  if (filter != Filter::Area) return kernel_weights(src, dst, shift, filter);
  if (dst > src) {
    return fail(ErrorCode::FilterDirection, std::format("Area : {} vers {} agrandirait", src, dst));
  }
  if (shift != 0.0) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("Area : décalage {} non nul", shift));
  }
  return area_weights(src, dst);
}

Result<PlaneF> resample_rows(const PlaneF& in, const Weights& wx) {
  if (auto ok = check_plane(in, "passe horizontale"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_weights(wx, in.width, "passe horizontale"); !ok) return std::unexpected(ok.error());
  PlaneF out(wx.dst, in.height);
  for (int y = 0; y < in.height; ++y) {
    for (int x = 0; x < wx.dst; ++x) {
      const std::span<const float> w = wx.row(x);
      float a = 0.0F;
      for (int j = 0; j < wx.taps; ++j) a += w[at(j)] * in.at(mirror_index(wx.start[at(x)] + j, wx.src), y);
      out.at(x, y) = a;
    }
  }
  return out;
}

Result<PlaneF> resample_columns(const PlaneF& in, const Weights& wy) {
  if (auto ok = check_plane(in, "passe verticale"); !ok) return std::unexpected(ok.error());
  if (auto ok = check_weights(wy, in.height, "passe verticale"); !ok) return std::unexpected(ok.error());
  PlaneF out(in.width, wy.dst);
  for (int y = 0; y < wy.dst; ++y) {
    const std::span<const float> w = wy.row(y);
    for (int x = 0; x < in.width; ++x) {
      float a = 0.0F;
      for (int j = 0; j < wy.taps; ++j) a += w[at(j)] * in.at(x, mirror_index(wy.start[at(y)] + j, wy.src));
      out.at(x, y) = a;
    }
  }
  return out;
}

Result<PlaneF> resize(const PlaneF& in, int width, int height, Filter filter) {
  if (auto ok = check_frame(in.width, in.height, "source du rééchantillonnage"); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = check_frame(width, height, "cible du rééchantillonnage"); !ok)
    return std::unexpected(ok.error());
  const auto wx = compute_weights(in.width, width, 0.0, filter);
  if (!wx) return std::unexpected(wx.error());
  const auto wy = compute_weights(in.height, height, 0.0, filter);
  if (!wy) return std::unexpected(wy.error());
  return resample_rows(in, *wx).and_then([&wy](const PlaneF& rows) { return resample_columns(rows, *wy); });
}

Result<RgbImage> resize(const RgbImage& in, int width, int height, Filter filter) {
  if (auto ok = check_rgb(in, "image à rééchantillonner"); !ok) return std::unexpected(ok.error());
  RgbImage out;
  for (std::size_t c = 0; c < out.planes.size(); ++c) {
    auto plane = resize(in.planes[c], width, height, filter);
    if (!plane) return std::unexpected(plane.error());
    out.planes[c] = std::move(*plane);
  }
  return out;
}

}  // namespace ajp::kernels::ref
