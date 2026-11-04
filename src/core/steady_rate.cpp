#include "core/steady_rate.h"

#include <algorithm>
#include <cmath>
#include <optional>

#include "core/cadence.h"
#include "core/pacing.h"

namespace ajp::core {
namespace {

bool below(const Ratio& a, const Ratio& b) {
  return a.num * b.den < b.num * a.den;
}

}  // namespace

void SteadyRate::configure(const FreeParams& params, const FreeDisplay& display, double source_fps) {
  params_ = params;
  display_ = display;
  fps_ = source_fps > 0.0 ? source_fps : 24.0;
  candidates_.assign(1, Ratio{1, 1});
  const double top = 1.0 + 0.5 * static_cast<double>(std::max(params_.half_steps - 1, 0));
  const double cap = display_.hz > 0.0 ? display_.hz / fps_ : top;
  if (display_.vrr) {
    for (int k = 1; k < params_.half_steps; ++k) {
      const Ratio n = reduced(Ratio{2 + k, 2});
      if (n.value() <= cap * (1.0 + 1e-9)) candidates_.push_back(n);
    }
  } else {
    // À fréquence fixe, les facteurs qui calent l'écran : hz / (k·fps) pour k vsyncs par image source.
    for (int k = 1; k <= 8; ++k) {
      const std::optional<Ratio> n = lock_factor(display_.hz / (static_cast<double>(k) * fps_));
      if (!n || n->value() > top || n->value() <= 1.0) continue;
      if (std::ranges::find(candidates_, *n) == candidates_.end()) candidates_.push_back(*n);
    }
    std::ranges::sort(candidates_, below);
  }
  if (std::ranges::find(candidates_, factor_) == candidates_.end()) set_factor(Ratio{1, 1});
  if (has_prior_) set_factor(choose(scaled_prior()));
}

void SteadyRate::prior(CostLine line) {
  prior_ = line;
  has_prior_ = line.a > 0.0 || line.b > 0.0;
  if (has_prior_) set_factor(choose(scaled_prior()));
}

double SteadyRate::cost_at(const CostLine& line, double n) const {
  return line.a + line.b * n + display_us_ * n;
}

Ratio SteadyRate::choose(const CostLine& line) const {
  const double budget = 1e6 / fps_ / (1.0 + params_.margin);
  Ratio best{1, 1};
  for (const Ratio& c : candidates_) {
    if (cost_at(line, c.value()) <= budget) best = c;
  }
  return best;
}

CostLine SteadyRate::scaled_prior() const {
  return CostLine{.a = prior_.a * scale_, .b = prior_.b * scale_};
}

void SteadyRate::set_factor(Ratio factor) {
  factor = reduced(factor);
  if (factor == factor_) return;
  factor_ = factor;
  changed_ = true;
}

bool SteadyRate::take_changed() {
  const bool was = changed_;
  changed_ = false;
  return was;
}

void SteadyRate::display_cost(std::int64_t d_us) {
  if (d_us <= 0) return;
  const auto d = static_cast<double>(d_us);
  display_us_ = display_us_ <= 0.0 ? d : display_us_ + params_.display_smoothing * (d - display_us_);
}

void SteadyRate::rechoose_from_plan(std::int64_t now_us) {
  std::optional<CostLine> line;
  if (has_prior_ && plan_ratio_n_ > 0) {
    // La forme a priori, à l'échelle mesurée dans ce plan : une droite ajustée sur un seul nombre d'images
    // ne sépare pas a de b.
    scale_ = plan_ratio_sum_ / static_cast<double>(plan_ratio_n_);
    line = scaled_prior();
  } else {
    line = plan_.fit(now_us);
  }
  if (!line) return;
  Ratio next = choose(*line);
  if (frozen_ && below(factor_, next)) next = factor_;  // gelé : ne remonte pas avant la coupe
  set_factor(next);
}

void SteadyRate::pair_cost(int images, std::int64_t gpu_us, std::int64_t now_us) {
  if (images < 1 || gpu_us <= 0 || now_us < settled_until_us_) return;
  plan_.add(images, gpu_us, now_us);
  ++plan_pairs_;
  if (has_prior_) {
    const double expected = prior_.a + prior_.b * static_cast<double>(images);
    if (expected > 0.0) {
      plan_ratio_sum_ += static_cast<double>(gpu_us) / expected;
      ++plan_ratio_n_;
    }
  }
  const double budget = 1e6 / fps_ / (1.0 + params_.margin);
  // Les premières paires d'un plan plus lourd que prévu : le choix se fait tout de suite.
  if (!adjusted_ && plan_pairs_ <= params_.early_pairs) {
    const double spent = static_cast<double>(gpu_us) + display_us_ * static_cast<double>(images);
    if (spent > budget * static_cast<double>(images) / factor_.value()) ++early_over_;
    if (early_over_ >= 2) {
      rechoose_from_plan(now_us);
      adjusted_ = true;
    }
  }
  if (!adjusted_ && plan_pairs_ >= params_.fit_pairs) {
    rechoose_from_plan(now_us);
    adjusted_ = true;
  }
}

void SteadyRate::scene_cut(std::int64_t now_us) {
  if (has_prior_ && plan_ratio_n_ > 0) scale_ = plan_ratio_sum_ / static_cast<double>(plan_ratio_n_);
  plan_.clear();
  plan_pairs_ = 0;
  plan_ratio_sum_ = 0.0;
  plan_ratio_n_ = 0;
  early_over_ = 0;
  adjusted_ = false;
  frozen_ = false;
  recent_losses_.fill(0);
  if (has_prior_) set_factor(choose(scaled_prior()));
  (void)now_us;
}

SteadyRate::Bucket& SteadyRate::bucket(std::int64_t now_us) {
  constexpr std::int64_t kBucketUs = 250'000;
  const std::int64_t index = now_us / kBucketUs;
  Bucket& b = buckets_[static_cast<std::size_t>(index) % buckets_.size()];
  if (b.index != index) b = Bucket{.index = index};
  return b;
}

void SteadyRate::produced(int images, std::int64_t now_us) {
  if (images > 0) bucket(now_us).produced += images;
}

void SteadyRate::lost(std::int64_t now_us, int count) {
  if (count <= 0) return;
  lost_total_ += static_cast<std::uint32_t>(count);
  bucket(now_us).lost += count;
  if (now_us < settled_until_us_) return;
  for (int i = 0; i < count; ++i) recent_losses_[next_loss_++ % recent_losses_.size()] = now_us;
  int inside = 0;
  for (const std::int64_t at : recent_losses_) {
    if (at > 0 && now_us - at <= params_.urgent_window_us) ++inside;
  }
  if (inside < params_.urgent_losses) return;
  // Urgence : un demi-pas plus bas (ou le choix a priori s'il est plus bas encore), gelé jusqu'à la coupe.
  ++urgencies_;
  frozen_ = true;
  recent_losses_.fill(0);
  Ratio lower{1, 1};
  for (const Ratio& c : candidates_) {
    if (below(c, factor_)) lower = c;
  }
  const Ratio guess = has_prior_ ? choose(scaled_prior()) : lower;
  set_factor(below(guess, lower) ? guess : lower);
}

void SteadyRate::settle(std::int64_t now_us) {
  settled_until_us_ = now_us + params_.settle_us;
  recent_losses_.fill(0);
}

bool SteadyRate::should_fall_back(std::int64_t now_us) const {
  if (!(factor_ == Ratio{1, 1})) return false;
  constexpr std::int64_t kBucketUs = 250'000;
  const std::int64_t last = now_us / kBucketUs;
  const std::int64_t first = (now_us - params_.fallback_window_us) / kBucketUs;
  int produced = 0;
  int lost = 0;
  for (const Bucket& b : buckets_) {
    if (b.index > first && b.index <= last) {
      produced += b.produced;
      lost += b.lost;
    }
  }
  return lost >= params_.fallback_min_lost && lost * 1000 >= params_.fallback_per_mille * (produced + lost);
}

SteadyInfo SteadyRate::info() const {
  return SteadyInfo{.factor = factor_,
                    .scale = scale_,
                    .display_us = std::llround(display_us_),
                    .budget_us = std::llround(1e6 / fps_ / (1.0 + params_.margin)),
                    .frozen = frozen_,
                    .lost = lost_total_,
                    .urgencies = urgencies_};
}

}  // namespace ajp::core
