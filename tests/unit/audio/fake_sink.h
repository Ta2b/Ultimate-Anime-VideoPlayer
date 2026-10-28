// Périphérique simulé : le test décide combien de trames il consomme et quand.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <utility>
#include <vector>

#include "audio/sink.h"
#include "core/clock.h"

namespace ajp::audio::test {

class FakeSink final : public AudioSink {
 public:
  explicit FakeSink(PcmFormat format, std::int64_t latency_frames = 0, bool keep_data = true)
      : format_(format), latency_(latency_frames), keep_data_(keep_data) {}

  PcmFormat format() const override { return format_; }
  std::expected<void, Refusal> put(std::span<const float> samples) override {
    if (keep_data_) data_.insert(data_.end(), samples.begin(), samples.end());
    pushed_ += std::ssize(samples) / format_.channels;
    return {};
  }
  std::int64_t pushed_frames() const override { return pushed_; }
  DeviceReading reading() const override { return {.consumed_frames = consumed_, .host_us = host_us_}; }
  std::int64_t latency_frames() const override { return latency_; }
  void set_paused(bool paused) override { paused_ = paused; }
  void set_gain(float gain) override { gain_ = gain; }
  float gain() const override { return gain_; }
  void set_ratio(double ratio) override { ratio_ = ratio; }
  double ratio() const override { return ratio_; }
  std::int64_t clear() override {
    const std::int64_t consumed = consumed_;
    consumed_ = pushed_;
    ++clears_;
    return consumed;
  }
  void on_demand(std::function<void(const DeviceReading&)> callback) override {
    demand_ = std::move(callback);
  }

  // Le périphérique joue frames trames à vitesse nominale (rapport de fréquence
  // compris), puis mesure la file à l'instant host_us. En pause, il ne joue rien.
  DeviceReading play(double frames, std::int64_t host_us) {
    if (!paused_) {
      owed_ += frames * ratio_;
      const auto whole = static_cast<std::int64_t>(std::floor(owed_));
      owed_ -= static_cast<double>(whole);
      consumed_ = std::min(pushed_, consumed_ + whole);
    }
    host_us_ = host_us;
    const DeviceReading now = reading();
    if (demand_) demand_(now);
    return now;
  }

  std::span<const float> data() const { return data_; }
  bool paused() const { return paused_; }
  int clears() const { return clears_; }

 private:
  PcmFormat format_;
  std::int64_t latency_;
  bool keep_data_;
  std::vector<float> data_;
  std::int64_t pushed_ = 0;
  std::int64_t consumed_ = 0;
  std::int64_t host_us_ = host_now_us();
  double owed_ = 0.0;
  bool paused_ = true;
  float gain_ = 1.0F;
  double ratio_ = 1.0;
  int clears_ = 0;
  std::function<void(const DeviceReading&)> demand_;
};

// Horloge maître réglée par le test, comme present la publierait.
class FakeMaster final : public MasterClock {
 public:
  void set(const ClockSnapshot& snapshot) { snapshot_ = snapshot; }
  ClockSnapshot snapshot() const override { return snapshot_; }

 private:
  ClockSnapshot snapshot_;
};

}  // namespace ajp::audio::test
