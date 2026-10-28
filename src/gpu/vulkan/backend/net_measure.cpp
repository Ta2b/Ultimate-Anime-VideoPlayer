#include "gpu/vulkan/backend/net_measure.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ajp::gpu::vk::net {
namespace {

constexpr std::int64_t kBlockPixels = 32;  // côté des aplats du damier
constexpr std::int64_t kLinePixels = 16;   // pas des traits sombres
constexpr std::int64_t kShiftPixels = 2;   // décalage par unité de shift
constexpr double kLineValue = 0.04;
constexpr double kBase = 0.1;
constexpr double kGradient = 0.55;
constexpr double kRaised = 0.2;      // les cases hautes du damier
constexpr double kPlaneStep = 0.05;  // décalage d'un plan au suivant (trois valeurs, en boucle)
constexpr double kFlatValue = 0.5;

std::int64_t floor_div(std::int64_t a, std::int64_t b) {
  return a >= 0 ? a / b : -((-a + b - 1) / b);
}
std::int64_t floor_mod(std::int64_t a, std::int64_t b) {
  return a - (floor_div(a, b) * b);
}

}  // namespace

std::vector<float> probe_image(std::span<const std::int64_t> dims, int shift) {
  std::size_t count = 1;
  for (const std::int64_t d : dims) {
    if (d <= 0) return {};
    count *= static_cast<std::size_t>(d);
  }
  if (dims.size() < 2) return std::vector<float>(count, static_cast<float>(kFlatValue));
  const std::int64_t width = dims.back();
  const std::int64_t height = dims[dims.size() - 2];
  const auto plane_size = static_cast<std::size_t>(width * height);
  const double dx = width > 1 ? 1.0 / static_cast<double>(width - 1) : 0.0;
  const double dy = height > 1 ? 1.0 / static_cast<double>(height - 1) : 0.0;
  std::vector<float> out(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t plane = i / plane_size;
    const auto at = static_cast<std::int64_t>(i % plane_size);
    const std::int64_t y = at / width;
    const std::int64_t x = (at % width) - (kShiftPixels * shift);
    double v = kLineValue;
    if (floor_mod(x, kLinePixels) != 0 && floor_mod(y, kLinePixels) != 0) {
      const bool raised = ((floor_div(x, kBlockPixels) + floor_div(y, kBlockPixels)) & 1) != 0;
      v = kBase + (kGradient * ((0.6 * static_cast<double>(x) * dx) + (0.4 * static_cast<double>(y) * dy))) +
          (raised ? kRaised : 0.0) + (kPlaneStep * static_cast<double>(plane % 3));
    }
    out[i] = static_cast<float>(std::clamp(v, 0.0, 1.0));
  }
  return out;
}

double median(std::span<const double> samples) {
  if (samples.empty()) return 0.0;
  std::vector<double> sorted(samples.begin(), samples.end());
  std::ranges::sort(sorted);
  const std::size_t n = sorted.size();
  return n % 2 == 1 ? sorted[n / 2] : 0.5 * (sorted[(n / 2) - 1] + sorted[n / 2]);
}

double mean_error(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  if (a.empty()) return 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
  return sum / static_cast<double>(a.size());
}

std::optional<std::size_t> choose_unit(std::span<const std::optional<UnitMeasure>> measured,
                                       std::size_t preferred, double tolerance, double max_error) {
  // Le temps d'une unité juste ; rien pour une unité en échec ou imprécise (une erreur NaN
  // compte comme imprécise : la comparaison est fausse).
  const auto fair_ms = [max_error](const std::optional<UnitMeasure>& m) -> std::optional<double> {
    if (m && m->error <= max_error) return m->ms;
    return std::nullopt;
  };
  std::optional<std::size_t> fastest;
  double fastest_ms = 0.0;
  for (std::size_t i = 0; i < measured.size(); ++i) {
    const std::optional<double> ms = fair_ms(measured[i]);
    if (ms && (!fastest || *ms < fastest_ms)) {
      fastest = i;
      fastest_ms = *ms;
    }
  }
  if (!fastest) return std::nullopt;
  if (preferred < measured.size()) {
    if (const std::optional<double> ms = fair_ms(measured[preferred]); ms && *ms <= tolerance * fastest_ms)
      return preferred;
  }
  return fastest;
}

}  // namespace ajp::gpu::vk::net
