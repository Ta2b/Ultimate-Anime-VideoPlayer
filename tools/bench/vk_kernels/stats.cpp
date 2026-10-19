#include "stats.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <utility>
#include <vector>

namespace ajp::bench {

namespace {

std::unexpected<Refusal> out_of_range(std::string detail) {
  return std::unexpected(Refusal{.code = RefusalCode::FieldOutOfRange, .detail = std::move(detail)});
}

}  // namespace

std::expected<Summary, Refusal> summarize(std::span<const double> samples) {
  if (samples.empty()) return out_of_range("statistiques : série vide");
  for (const double v : samples) {
    if (!std::isfinite(v) || v < 0.0) return out_of_range(std::format("statistiques : durée {} refusée", v));
  }
  std::vector<double> sorted(samples.begin(), samples.end());
  std::ranges::sort(sorted);
  const std::size_t n = sorted.size();

  Summary s{.count = n, .min = sorted.front(), .max = sorted.back()};
  s.median = n % 2 == 1 ? sorted[n / 2] : (sorted[(n / 2) - 1] + sorted[n / 2]) / 2.0;
  // ⌈0,95 n⌉ en entiers : (95 n + 99) / 100, rang 1 au moins.
  const std::size_t rank = std::max<std::size_t>(1, ((95 * n) + 99) / 100);
  s.p95 = sorted[rank - 1];
  s.slow =
      static_cast<std::size_t>(std::ranges::count_if(sorted, [&](double v) { return v > 1.5 * s.median; }));

  if (n > 1) {
    double mean = 0.0;
    for (const double v : sorted) mean += v;
    mean /= static_cast<double>(n);
    double squares = 0.0;
    for (const double v : sorted) squares += (v - mean) * (v - mean);
    s.stddev = std::sqrt(squares / static_cast<double>(n - 1));
  }
  return s;
}

std::expected<double, Refusal> gigabytes_per_second(std::uint64_t bytes, double milliseconds) {
  if (!std::isfinite(milliseconds) || milliseconds <= 0.0) {
    return out_of_range(std::format("débit : durée {} ms refusée", milliseconds));
  }
  return static_cast<double>(bytes) / (milliseconds * 1.0e6);
}

}  // namespace ajp::bench
