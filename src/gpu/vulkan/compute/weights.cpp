// Repris de animejanai-inference-rtx@118c272 src/resample.h:53-84 (compute_filter de zimg) et
// src/pipeline_kernels.cu:34-50 (empreinte de Area), d'après les formules de
// kernels/spec/resample.h et kernels/spec/transfer.h. Le calcul suit l'ordre écrit dans la
// spécification : les tables sont identiques au bit près à celles de la référence CPU.
#include "gpu/vulkan/compute/weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>

#include "gpu/vulkan/compute/refuse.h"
#include "kernels/spec/transfer.h"

namespace ajp::gpu::vk {
namespace {

std::size_t at(int i) {
  return static_cast<std::size_t>(i);
}

double kernel(kernels::Filter filter, double x) {
  return filter == kernels::Filter::Bilinear ? kernels::bilinear(x) : kernels::spline36(x);
}

WeightTable table(int taps, int src, int dst) {
  return {.taps = static_cast<std::uint32_t>(taps),
          .src = static_cast<std::uint32_t>(src),
          .dst = static_cast<std::uint32_t>(dst),
          .start = std::vector<std::int32_t>(at(dst)),
          .w = std::vector<float>(at(dst) * at(taps), 0.0F)};
}

// Spline36 et bilinéaire : scale = dst / src, step = min(scale, 1), taps = max(⌈support / step⌉ × 2, 1).
WeightTable kernel_weights(int src, int dst, double shift, kernels::Filter filter) {
  const double scale = static_cast<double>(dst) / static_cast<double>(src);
  const double step = std::min(scale, 1.0);
  const int taps = std::max(static_cast<int>(std::ceil(kernels::support(filter) / step)) * 2, 1);
  WeightTable out = table(taps, src, dst);
  std::vector<double> row(at(taps));
  for (int i = 0; i < dst; ++i) {
    const double pos = (i + 0.5) / scale + shift;
    const double begin = std::floor(pos - taps / 2.0 + 0.5) + 0.5;
    double sum = 0.0;
    for (int j = 0; j < taps; ++j) {
      row[at(j)] = kernel(filter, (begin + j - pos) * step);
      sum += row[at(j)];
    }
    out.start[at(i)] = static_cast<std::int32_t>(begin - 0.5);
    for (int j = 0; j < taps; ++j) out.w[at(i) * at(taps) + at(j)] = static_cast<float>(row[at(j)] / sum);
  }
  return out;
}

// Area : recouvrement exact de l'empreinte [i src / dst, (i + 1) src / dst].
WeightTable area_weights(int src, int dst) {
  const auto footprint = [src, dst](int i) { return static_cast<double>(i) * src / dst; };
  int taps = 1;
  for (int i = 0; i < dst; ++i) {
    const int x0 = static_cast<int>(std::floor(footprint(i)));
    const int x1 = std::min(src, static_cast<int>(std::ceil(footprint(i + 1))));
    taps = std::max(taps, x1 - x0);
  }
  WeightTable out = table(taps, src, dst);
  for (int i = 0; i < dst; ++i) {
    const double a = footprint(i);
    const double b = footprint(i + 1);
    const int x0 = static_cast<int>(std::floor(a));
    const int x1 = std::min(src, static_cast<int>(std::ceil(b)));
    out.start[at(i)] = x0;
    for (int sx = x0; sx < x1; ++sx) {
      const double overlap = std::min(b, sx + 1.0) - std::max(a, static_cast<double>(sx));
      out.w[at(i) * at(taps) + at(sx - x0)] = static_cast<float>(overlap / (b - a));
    }
  }
  return out;
}

}  // namespace

std::expected<WeightTable, Refusal> compute_weights(int src, int dst, double shift, kernels::Filter filter) {
  if (!kernels::valid_plane_dimension(src) || !kernels::valid_plane_dimension(dst)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("poids : {} vers {} hors de [1, {}]", src, dst, kernels::kMaxDimension));
  }
  if (!std::isfinite(shift) || std::abs(shift) > 1.0) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("poids : décalage {} hors de [-1, 1]", shift));
  }
  if (filter != kernels::Filter::Area) return kernel_weights(src, dst, shift, filter);
  if (dst > src)
    return refuse(RefusalCode::InputMismatch, std::format("Area : {} vers {} agrandirait", src, dst));
  if (shift != 0.0)
    return refuse(RefusalCode::FieldOutOfRange, std::format("Area : décalage {} non nul", shift));
  return area_weights(src, dst);
}

std::expected<HlgDisplay, Refusal> hlg_display(float peak_nits) {
  namespace hlg = kernels::hlg;
  if (!(peak_nits >= hlg::kMinPeak && peak_nits <= hlg::kMaxPeak)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("pic HLG {} cd/m² hors de [{}, {}]", peak_nits, hlg::kMinPeak, hlg::kMaxPeak));
  }
  return HlgDisplay{.peak = peak_nits,
                    .gamma = hlg::kGammaAt1000 + hlg::kGammaSlope * std::log10(peak_nits / 1000.0F)};
}

std::expected<Bt1886Display, Refusal> bt1886_display(float white, float black) {
  constexpr float gamma = kernels::bt1886::kGamma;
  if (!(black >= 0.0F && white > black && std::isfinite(white))) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("BT.1886 : blanc {} et noir {} invalides", white, black));
  }
  const float wg = std::pow(white, 1.0F / gamma);
  const float bg = std::pow(black, 1.0F / gamma);
  return Bt1886Display{.white = white, .black = black, .a = std::pow(wg - bg, gamma), .b = bg / (wg - bg)};
}

}  // namespace ajp::gpu::vk
