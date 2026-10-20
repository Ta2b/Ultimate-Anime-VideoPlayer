#include "gpu/vulkan/backend/reduce_weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <vector>

namespace ajp::gpu::vk {
namespace {

// Noyau à la distance d, en pixels de sortie (slice_kernels.cu, reduce_weight).
double weight(const Reduction& r, double d) {
  d = std::fabs(d);
  if (r.kind == ReduceKind::Triangle) return std::max(0.0, 1.0 - d);
  if (r.kind == ReduceKind::Spline36) {
    if (d < 1.0) return ((13.0 / 11.0 * d - 453.0 / 209.0) * d - 3.0 / 209.0) * d + 1.0;
    if (d < 2.0) return ((-6.0 / 11.0 * (d - 1.0) + 270.0 / 209.0) * (d - 1.0) - 156.0 / 209.0) * (d - 1.0);
    if (d < 3.0) return ((1.0 / 11.0 * (d - 2.0) - 45.0 / 209.0) * (d - 2.0) + 26.0 / 209.0) * (d - 2.0);
    return 0.0;
  }
  const double b = r.b;
  const double c = r.c;
  if (d < 1.0) return ((12 - 9 * b - 6 * c) * d * d * d + (-18 + 12 * b + 6 * c) * d * d + (6 - 2 * b)) / 6.0;
  if (d < 2.0) {
    return ((-b - 6 * c) * d * d * d + (6 * b + 30 * c) * d * d + (-12 * b - 48 * c) * d + (8 * b + 24 * c)) /
           6.0;
  }
  return 0.0;
}

double radius(ReduceKind kind) {
  switch (kind) {
    case ReduceKind::Triangle:
      return 1.0;
    case ReduceKind::Spline36:
      return 3.0;
    case ReduceKind::Area:
    case ReduceKind::Cubic:
      break;
  }
  return 2.0;
}

}  // namespace

std::expected<WeightTable, Refusal> reduction_weights(int src, int dst, const Reduction& reduction) {
  if (src < 1 || dst < 1 || dst > src) {
    return std::unexpected(
        Refusal{RefusalCode::ShapeInvalid,
                std::format("réduction de {} vers {} : 1 ≤ sortie ≤ source attendu", src, dst)});
  }
  if (reduction.kind == ReduceKind::Area) return compute_weights(src, dst, 0.0, kernels::Filter::Area);
  if (!std::isfinite(reduction.b) || !std::isfinite(reduction.c)) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, "réduction cubique : B ou C non fini"});
  }
  const double scale = static_cast<double>(src) / static_cast<double>(dst);
  const double reach = radius(reduction.kind) * scale;
  // Poids de chaque sortie par indice de source répété au bord, puis la fenêtre commune.
  std::vector<std::vector<double>> rows(static_cast<std::size_t>(dst));
  std::vector<int> lows(static_cast<std::size_t>(dst));
  std::uint32_t taps = 1;
  for (int x = 0; x < dst; ++x) {
    const double center = (x + 0.5) * scale - 0.5;
    const int first = static_cast<int>(std::ceil(center - reach));
    const int last = static_cast<int>(std::floor(center + reach));
    const int low = std::clamp(first, 0, src - 1);
    const int high = std::clamp(last, 0, src - 1);
    std::vector<double>& row = rows[static_cast<std::size_t>(x)];
    row.assign(static_cast<std::size_t>(high - low + 1), 0.0);
    for (int i = first; i <= last; ++i) {
      row[static_cast<std::size_t>(std::clamp(i, 0, src - 1) - low)] +=
          weight(reduction, (i - center) / scale);
    }
    lows[static_cast<std::size_t>(x)] = low;
    taps = std::max(taps, static_cast<std::uint32_t>(row.size()));
  }
  WeightTable table{.taps = taps,
                    .src = static_cast<std::uint32_t>(src),
                    .dst = static_cast<std::uint32_t>(dst),
                    .start = std::vector<std::int32_t>(static_cast<std::size_t>(dst)),
                    .w = std::vector<float>(static_cast<std::size_t>(dst) * taps, 0.0F)};
  for (int x = 0; x < dst; ++x) {
    const std::vector<double>& row = rows[static_cast<std::size_t>(x)];
    const int low = lows[static_cast<std::size_t>(x)];
    const int start = std::min(low, src - static_cast<int>(taps));
    double total = 0.0;
    for (const double v : row) total += v;
    table.start[static_cast<std::size_t>(x)] = start;
    for (std::size_t k = 0; k < row.size(); ++k) {
      const std::size_t j = static_cast<std::size_t>(low - start) + k;
      table.w[(static_cast<std::size_t>(x) * taps) + j] = static_cast<float>(row[k] / total);
    }
  }
  return table;
}

}  // namespace ajp::gpu::vk
