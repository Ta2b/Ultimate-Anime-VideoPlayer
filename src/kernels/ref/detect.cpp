// Repris de animejanai-inference-rtx@118c272 src/kernels.cu:698-743 (k_scd_diff),
// 923-959 (k_block_diff) et src/aji_trt.cpp:3729-3829 (normalisation par l'aire remplie).
#include "kernels/ref/detect.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <format>

#include "kernels/ref/check.h"

namespace ajp::kernels::ref {
namespace {

// Deux lumas de même taille, aux codes valides du format (P010 compris : 6 bits bas nuls).
Result<void> check_pair(const PlaneU16& a, const PlaneU16& b, Format format) {
  if (auto ok = check_same_size(a, b, "plans de luma comparés"); !ok) return ok;
  for (const PlaneU16* p : {&a, &b}) {
    if (auto ok = check_plane(*p, "plans de luma comparés"); !ok) return ok;
    if (auto ok = check_samples(*p, format, "luma"); !ok) return ok;
  }
  return {};
}

}  // namespace

Result<double> scene_diff_sum(const PlaneU16& a, const PlaneU16& b, Format format) {
  if (auto ok = check_pair(a, b, format); !ok) return std::unexpected(ok.error());
  const float norm = scene_norm(format);
  double sum = 0.0;
  for (std::size_t i = 0; i < a.data.size(); ++i) {
    const auto pa = static_cast<float>(a.data[i]);
    const auto pb = static_cast<float>(b.data[i]);
    sum += static_cast<double>(std::fabs(pa - pb) * norm);
  }
  return sum;
}

Result<double> scene_score(const PlaneU16& a, const PlaneU16& b, Format format,
                           const RifeGeometry& geometry) {
  if (a.width != geometry.width || a.height != geometry.height) {
    return fail(ErrorCode::SizeMismatch, std::format("luma {}x{} pour une géométrie RIFE {}x{}", a.width,
                                                     a.height, geometry.width, geometry.height));
  }
  if (!rife_fits(geometry.width, geometry.height, geometry.padded_width, geometry.padded_height)) {
    return fail(ErrorCode::ParameterOutOfRange,
                "aire remplie hors des multiples de 64 ou plus petite que l'image");
  }
  const double area =
      static_cast<double>(geometry.padded_width) * static_cast<double>(geometry.padded_height);
  return scene_diff_sum(a, b, format).transform([area](double sum) { return sum / area; });
}

Result<std::uint32_t> block_diff_max(const PlaneU16& a, const PlaneU16& b, Format format) {
  if (auto ok = check_pair(a, b, format); !ok) return std::unexpected(ok.error());
  const std::uint32_t scale = held_scale(format);
  std::uint32_t worst = 0;
  for (int y0 = 0; y0 < a.height; y0 += kHeldBlock) {
    for (int x0 = 0; x0 < a.width; x0 += kHeldBlock) {
      const int x1 = std::min(x0 + kHeldBlock, a.width);
      const int y1 = std::min(y0 + kHeldBlock, a.height);
      std::uint32_t sum = 0;
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
          sum += static_cast<std::uint32_t>(std::abs(int{a.at(x, y)} - int{b.at(x, y)}));
        }
      }
      const auto n = static_cast<std::uint32_t>((x1 - x0) * (y1 - y0));
      worst = std::max(worst, sum * scale / n);
    }
  }
  return worst;
}

}  // namespace ajp::kernels::ref

namespace ajp::kernels::ref {

Result<PairMeasure> pair_scores(const RgbImage& a, const RgbImage& b, float levels,
                                const RifeGeometry& geometry, float kr, float kb) {
  if (a.width() != b.width() || a.height() != b.height() || a.width() != geometry.width ||
      a.height() != geometry.height) {
    return fail(ErrorCode::SizeMismatch,
                std::format("images {}x{} et {}x{} pour une géométrie RIFE {}x{}", a.width(), a.height(),
                            b.width(), b.height(), geometry.width, geometry.height));
  }
  if (!rife_fits(geometry.width, geometry.height, geometry.padded_width, geometry.padded_height)) {
    return fail(ErrorCode::ParameterOutOfRange,
                "aire remplie hors des multiples de 64 ou plus petite que l'image");
  }
  if (!(levels > 0.0F)) return fail(ErrorCode::ParameterOutOfRange, "niveaux de luma nuls ou négatifs");
  const float kg = 1.0F - kr - kb;
  const auto luma = [&](const RgbImage& image, int x, int y) {
    return kr * image.planes[0].at(x, y) + kg * image.planes[1].at(x, y) + kb * image.planes[2].at(x, y);
  };
  PairMeasure out;
  for (int y0 = 0; y0 < a.height(); y0 += kHeldBlock) {
    for (int x0 = 0; x0 < a.width(); x0 += kHeldBlock) {
      const int x1 = std::min(x0 + kHeldBlock, a.width());
      const int y1 = std::min(y0 + kHeldBlock, a.height());
      float sum = 0.0F;
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) sum += std::fabs(luma(a, x, y) - luma(b, x, y));
      }
      const float mean = sum / static_cast<float>((x1 - x0) * (y1 - y0));
      out.worst_block = std::max(out.worst_block, static_cast<std::uint32_t>(mean * levels * 256.0F));
      out.block_sums.push_back(static_cast<double>(sum));
      out.total += static_cast<double>(sum);
    }
  }
  const double area =
      static_cast<double>(geometry.padded_width) * static_cast<double>(geometry.padded_height);
  out.scene_score = pair_scene_score(out.total, levels, area);
  return out;
}

}  // namespace ajp::kernels::ref
