#include "core/free_rate.h"

#include <algorithm>
#include <cmath>

namespace ajp::core {

FreeRate::Bucket& FreeRate::bucket(std::int64_t now_us) {
  const std::int64_t index = now_us / kBucketUs;
  Bucket& b = buckets_[static_cast<std::size_t>(index % static_cast<std::int64_t>(buckets_.size()))];
  if (b.index != index) b = Bucket{.index = index};
  return b;
}

void FreeRate::sum(std::int64_t now_us, std::int64_t span_us, int& produced, int& lost) const {
  produced = 0;
  lost = 0;
  const std::int64_t last = now_us / kBucketUs;
  const std::int64_t first = last - (span_us + kBucketUs - 1) / kBucketUs + 1;
  for (const Bucket& b : buckets_) {
    if (b.index < first || b.index > last) continue;
    produced += b.produced;
    lost += b.lost;
  }
}

void FreeRate::cost(int images, std::int64_t busy_us, std::int64_t now_us) {
  if (images < 2 || busy_us <= 0 || now_us < settled_until_us_) return;
  const double sample = static_cast<double>(std::min(busy_us, kMaxSampleUs * images)) / images;
  if (service_us_ <= 0.0) {
    service_us_ = sample;
  } else if (sample > service_us_) {
    service_us_ += 0.6 * (sample - service_us_);  // une scène lourde : tout de suite
  } else {
    service_us_ += 0.12 * (sample - service_us_);  // une scène légère : lentement
  }
}

void FreeRate::produced(int images, std::int64_t now_us) {
  bucket(now_us).produced += images;
}

void FreeRate::missed(std::int64_t now_us, std::int64_t lost) {
  if (lost <= 0 || now_us < settled_until_us_) return;
  const std::int64_t counted = std::min<std::int64_t>(lost, 5);
  pressure_ =
      std::min(kPressureMax, pressure(now_us) * std::pow(kPressureStep, static_cast<double>(counted)));
  last_loss_us_ = now_us;
  lost_total_ += static_cast<std::uint32_t>(lost);
  bucket(now_us).lost += static_cast<int>(std::min<std::int64_t>(lost, 1000));
}

void FreeRate::settle(std::int64_t now_us) {
  settled_until_us_ = now_us + kSettleUs;
}

void FreeRate::reset(std::int64_t now_us) {
  const std::int64_t gap = min_gap_us_;
  *this = FreeRate{};
  min_gap_us_ = gap;
  settled_until_us_ = now_us + kSettleUs;
}

double FreeRate::pressure(std::int64_t now_us) const {
  if (pressure_ <= 1.0) return 1.0;
  const std::int64_t calm = now_us - last_loss_us_ - kCalmUs;
  if (calm <= 0) return pressure_;
  return 1.0 + (pressure_ - 1.0) * std::exp(-static_cast<double>(calm) / static_cast<double>(kReleaseUs));
}

std::int64_t FreeRate::step_us(std::int64_t now_us) const {
  if (service_us_ <= 0.0) return 0;
  const auto step = static_cast<std::int64_t>(std::llround(service_us_ * kMargin * pressure(now_us)));
  return std::max(step, min_gap_us_);
}

bool FreeRate::should_fall_back(std::int64_t now_us) const {
  int produced = 0;
  int lost = 0;
  sum(now_us, kFallbackWindowUs, produced, lost);
  return lost >= kFallbackMinLost && produced > 0 && lost * 1000 >= produced * kFallbackPerMille;
}

FreeInfo FreeRate::info(std::int64_t now_us) const {
  FreeInfo out;
  int produced = 0;
  int lost = 0;
  sum(now_us, 2'000'000, produced, lost);
  out.fps = static_cast<double>(produced) * 1e6 / 2'000'000.0;
  out.service_us = static_cast<std::int64_t>(std::llround(service_us_));
  out.step_us = step_us(now_us);
  out.pressure = pressure(now_us);
  out.lost = lost_total_;
  if (service_us_ > 0.0 && out.fps > 0.0)
    out.load_pct = static_cast<int>(std::lround(service_us_ * out.fps / 10'000.0));
  return out;
}

}  // namespace ajp::core
