#include "core/cost_model.h"

#include <algorithm>
#include <cmath>

namespace ajp::core {

double CostModel::weight_of(const ImageCost& c, std::int64_t now_us) const {
  if (c.weight <= 0.0) return 0.0;
  const double age = static_cast<double>(std::max<std::int64_t>(now_us - c.at_us, 0));
  return c.weight * std::exp2(-age / static_cast<double>(kHalfLifeUs));
}

void CostModel::add(int images, std::int64_t cost_us, std::int64_t now_us) {
  if (cost_us <= 0 || cost_us > kMaxCostUs) return;
  ImageCost& c = costs_[static_cast<std::size_t>(std::clamp(images, 1, kMaxImages))];
  double w = weight_of(c, now_us);
  const double value = static_cast<double>(cost_us);
  if (w < 0.5) {  // une autre scène : la moyenne d'avant ne vaut plus
    c.ewma_us = value;
    c.filled = 0;
    c.next = 0;
    w = 0.0;
  } else {
    const double kept = std::min(value, kOutlierFactor * c.ewma_us);
    c.ewma_us += std::max(0.15, 1.0 / (w + 1.0)) * (kept - c.ewma_us);
  }
  c.weight = std::min(w + 1.0, 24.0);
  c.at_us = now_us;
  c.ring[c.next] = static_cast<float>(value);
  c.next = (c.next + 1) % c.ring.size();
  c.filled = std::min(c.filled + 1, c.ring.size());
}

std::optional<CostLine> CostModel::fit(std::int64_t now_us) const {
  double w_sum = 0.0;
  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double sxy = 0.0;
  for (int n = 1; n <= kMaxImages; ++n) {
    const ImageCost& c = costs_[static_cast<std::size_t>(n)];
    const double w = std::min(weight_of(c, now_us), 12.0);
    if (w < 2.0) continue;
    const double x = n;
    w_sum += w;
    sx += w * x;
    sy += w * c.ewma_us;
    sxx += w * x * x;
    sxy += w * x * c.ewma_us;
  }
  if (w_sum < 3.0) return std::nullopt;
  const double spread = w_sum * sxx - sx * sx;
  if (spread < 1e-6 * w_sum * w_sum) {
    // Un seul nombre d'images vu : le mouvement pèse 40 % du coût, le reste se répartit par image.
    const double y = sy / w_sum;
    const double x = sx / w_sum;
    return CostLine{.a = 0.4 * y, .b = 0.6 * y / x};
  }
  double b = (w_sum * sxy - sx * sy) / spread;
  double a = (sy - b * sx) / w_sum;
  if (b < 0.0) {
    b = 0.0;
    a = sy / w_sum;
  } else if (a < 0.0) {
    a = 0.0;
    b = sxy / sxx;
  }
  return CostLine{.a = a, .b = b};
}

double CostModel::spike(std::int64_t now_us) const {
  const ImageCost* best = &costs_[1];
  for (const ImageCost& e : costs_) {
    if (weight_of(e, now_us) > weight_of(*best, now_us)) best = &e;
  }
  const ImageCost& c = *best;
  if (weight_of(c, now_us) < 1.0 || c.filled < 6 || c.ewma_us <= 0.0) return 1.0;
  std::array<float, 16> sorted = c.ring;
  const auto end = sorted.begin() + static_cast<std::ptrdiff_t>(c.filled);
  const auto p90 = sorted.begin() + static_cast<std::ptrdiff_t>((c.filled - 1) * 9 / 10);
  std::nth_element(sorted.begin(), p90, end);
  return std::clamp(static_cast<double>(*p90) / c.ewma_us, 1.0, 1.5);
}

void CostModel::fade(std::int64_t now_us) {
  for (ImageCost& c : costs_) {
    const double w = weight_of(c, now_us);
    c.weight = w * 0.5;
    c.at_us = now_us;
    // Les pointes de la scène d'avant ne disent rien de celle-ci : spike() ne lit que les mesures d'après
    // la coupe, et vaut 1 tant qu'elles sont trop peu nombreuses.
    c.filled = 0;
    c.next = 0;
  }
}

void CostModel::clear() {
  costs_ = {};
}

int CostModel::samples(int images) const {
  return static_cast<int>(costs_[static_cast<std::size_t>(std::clamp(images, 1, kMaxImages))].filled);
}

}  // namespace ajp::core
