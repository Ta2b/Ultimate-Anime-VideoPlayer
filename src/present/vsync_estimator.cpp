#include "present/vsync_estimator.h"

#include <algorithm>
#include <cmath>
#include <span>

namespace ajp::present {

namespace {

// Bornes d'une période plausible : de 1000 Hz à 1 Hz.
constexpr double kMinPeriodUs = 1'000.0;
constexpr double kMaxPeriodUs = 1'000'000.0;
// Facteur de l'écart absolu médian vers un écart type, pour un bruit gaussien.
constexpr double kMadToSigma = 1.4826;
constexpr double kOutlierSigmas = 3.0;

using Scratch = std::array<double, VsyncEstimator::kWindow>;

// Médiane des n premières valeurs ; les réordonne.
double median_of(Scratch& values, std::size_t n) {
  const auto middle = values.begin() + static_cast<std::ptrdiff_t>(n / 2);
  std::nth_element(values.begin(), middle, values.begin() + static_cast<std::ptrdiff_t>(n));
  return *middle;
}

struct Line {
  double origin = 0.0;  // instant du rang x = 0, relatif au premier instant
  double slope = 0.0;   // période
};

// Moindres carrés sur les points gardés ; nullopt si les rangs sont tous égaux.
std::optional<Line> least_squares(std::span<const double> x, std::span<const double> y,
                                  std::span<const bool> keep) {
  double n = 0.0;
  double sx = 0.0;
  double sy = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!keep[i]) continue;
    n += 1.0;
    sx += x[i];
    sy += y[i];
  }
  if (n < 2.0) return std::nullopt;
  const double mx = sx / n;
  const double my = sy / n;
  double sxx = 0.0;
  double sxy = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!keep[i]) continue;
    sxx += (x[i] - mx) * (x[i] - mx);
    sxy += (x[i] - mx) * (y[i] - my);
  }
  if (sxx <= 0.0) return std::nullopt;
  const double slope = sxy / sxx;
  return Line{.origin = my - slope * mx, .slope = slope};
}

}  // namespace

void VsyncEstimator::reset() {
  head_ = 0;
  count_ = 0;
  estimate_.reset();
  misfits_ = 0;
  bootstrapped_ = false;
}

void VsyncEstimator::push(const Sample& sample) {
  if (count_ == kWindow) {
    samples_[head_] = sample;
    head_ = (head_ + 1) % kWindow;
  } else {
    at(count_) = sample;
    ++count_;
  }
}

bool VsyncEstimator::fits(const Sample& sample) const {
  if (!estimate_) return true;
  const double predicted = estimate_->anchor_host_us +
                           static_cast<double>(sample.index - estimate_->anchor_index) * estimate_->period_us;
  const double threshold = std::max(kOutlierSigmas * estimate_->jitter_us, VsyncEstimator::kMinThresholdUs);
  return std::abs(static_cast<double>(sample.host_us) - predicted) <= threshold;
}

void VsyncEstimator::add(std::int64_t host_us) {
  if (count_ > 0 && host_us <= at(count_ - 1).host_us) return;
  if (count_ == 0) {
    push({.index = 0, .host_us = host_us});
    return;
  }
  const Sample& last = at(count_ - 1);
  if (!bootstrapped_) {
    push({.index = last.index + 1, .host_us = host_us});
    if (count_ >= kMinSamples) {
      bootstrap();
      bootstrapped_ = true;
      refit();
    }
    return;
  }

  Sample sample{.index = last.index + 1, .host_us = host_us};
  if (estimate_) {
    // Le rang vient de la droite, pas du dernier instant : un instant en retard ne décale
    // pas les suivants.
    const auto by_line = index_at(host_us).value_or(sample.index);
    sample.index = std::max(by_line, last.index);
  } else {
    // Droite perdue : l'écart au dernier instant, compté en intervalles médians.
    Scratch values{};
    std::size_t n = 0;
    for (std::size_t i = 1; i < count_; ++i) {
      const auto di = at(i).index - at(i - 1).index;
      if (di <= 0) continue;
      values[n++] = static_cast<double>(at(i).host_us - at(i - 1).host_us) / static_cast<double>(di);
    }
    if (n > 0) {
      const double period = median_of(values, n);
      const auto steps = std::llround(static_cast<double>(host_us - last.host_us) / period);
      sample.index = last.index + std::max<std::int64_t>(1, steps);
    }
  }

  // Compte net : un instant hors de la droite ajoute un, un instant sur la droite retire un.
  // Un retard isolé ne fait rien ; une autre grille (dont un instant sur quatre peut tomber
  // sur l'ancienne, 60 Hz contre 165 Hz) fait monter le compte.
  misfits_ = fits(sample) ? std::max(0, misfits_ - 1) : misfits_ + 1;
  if (misfits_ >= kMaxMisfits) {
    // La grille a changé (mode, fenêtre déplacée vers un autre écran) : on recommence.
    reset();
    push({.index = 0, .host_us = host_us});
    return;
  }
  push(sample);
  refit();
}

void VsyncEstimator::bootstrap() {
  Scratch intervals{};
  std::size_t n = 0;
  for (std::size_t i = 1; i < count_; ++i) {
    intervals[n++] = static_cast<double>(at(i).host_us - at(i - 1).host_us);
  }
  const double period = median_of(intervals, n);
  if (period <= 0.0) return;
  for (std::size_t i = 1; i < count_; ++i) {
    const double gap = static_cast<double>(at(i).host_us - at(i - 1).host_us);
    at(i).index = at(i - 1).index + std::max<std::int64_t>(1, std::llround(gap / period));
  }
}

void VsyncEstimator::refit() {
  estimate_.reset();
  if (count_ < kMinSamples) return;
  const Sample& first = at(0);
  Scratch x{};
  Scratch y{};
  for (std::size_t i = 0; i < count_; ++i) {
    x[i] = static_cast<double>(at(i).index - first.index);
    y[i] = static_cast<double>(at(i).host_us - first.host_us);
  }

  // Pente de départ robuste : la médiane des pentes entre instants voisins.
  Scratch work{};
  std::size_t n = 0;
  for (std::size_t i = 1; i < count_; ++i) {
    if (x[i] > x[i - 1]) work[n++] = (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
  }
  if (n == 0) return;
  Line line{.origin = 0.0, .slope = median_of(work, n)};
  for (std::size_t i = 0; i < count_; ++i) work[i] = y[i] - line.slope * x[i];
  line.origin = median_of(work, count_);

  std::array<bool, kWindow> keep{};
  double threshold = kMinThresholdUs;
  std::size_t kept = 0;
  // Deux passes : résidus à la droite robuste, puis à la droite des moindres carrés.
  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t i = 0; i < count_; ++i) work[i] = y[i] - (line.origin + line.slope * x[i]);
    Scratch spread = work;
    const double center = median_of(spread, count_);
    for (std::size_t i = 0; i < count_; ++i) spread[i] = std::abs(work[i] - center);
    const double mad = median_of(spread, count_);
    threshold = std::max(kOutlierSigmas * kMadToSigma * mad, kMinThresholdUs);
    kept = 0;
    for (std::size_t i = 0; i < count_; ++i) {
      keep[i] = std::abs(work[i] - center) <= threshold;
      if (keep[i]) ++kept;
    }
    if (2 * kept < count_) return;
    const auto fitted = least_squares(std::span(x).first(count_), std::span(y).first(count_),
                                      std::span<const bool>(keep).first(count_));
    if (!fitted) return;
    line = *fitted;
  }
  if (!(line.slope >= kMinPeriodUs && line.slope <= kMaxPeriodUs)) return;

  double squares = 0.0;
  for (std::size_t i = 0; i < count_; ++i) {
    if (!keep[i]) continue;
    const double r = y[i] - (line.origin + line.slope * x[i]);
    squares += r * r;
  }
  const Sample& last = at(count_ - 1);
  const double last_x = static_cast<double>(last.index - first.index);
  estimate_ = VsyncEstimate{
      .period_us = line.slope,
      .anchor_index = last.index,
      .anchor_host_us = static_cast<double>(first.host_us) + line.origin + line.slope * last_x,
      .jitter_us = std::sqrt(squares / static_cast<double>(kept)),
      .samples = kept,
  };
}

std::optional<std::int64_t> VsyncEstimator::index_at(std::int64_t host_us) const {
  if (!estimate_) return std::nullopt;
  const double periods = (static_cast<double>(host_us) - estimate_->anchor_host_us) / estimate_->period_us;
  return estimate_->anchor_index + std::llround(periods);
}

std::optional<std::int64_t> VsyncEstimator::host_at(std::int64_t index) const {
  if (!estimate_) return std::nullopt;
  return std::llround(estimate_->anchor_host_us +
                      static_cast<double>(index - estimate_->anchor_index) * estimate_->period_us);
}

}  // namespace ajp::present
