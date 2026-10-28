#include "audio/output.h"

#include <cmath>
#include <format>
#include <vector>

#include "audio/refuse.h"

namespace ajp::audio {
namespace {

// Les bornes de la correction de dérive sont incluses, à l'arrondi près.
constexpr double kRatioTolerance = 1e-12;

}  // namespace

AudioOutput::AudioOutput(std::unique_ptr<AudioSink> sink)
    : sink_(std::move(sink)), format_(sink_->format()), timeline_(format_.sample_rate) {
  sink_->set_paused(true);
  sink_->set_ratio(1.0);
}

std::expected<std::unique_ptr<AudioOutput>, Refusal> AudioOutput::open_sdl(const OutputRequest& request) {
  auto sink = open_sdl_sink(request);
  if (!sink) return std::unexpected(sink.error());
  return std::make_unique<AudioOutput>(std::move(*sink));
}

std::expected<void, Refusal> AudioOutput::push(std::int64_t pts_us, std::span<const float> samples) {
  const std::int64_t start = sink_->pushed_frames();
  if (auto queued = sink_->put(samples); !queued) return queued;
  timeline_.append(start, sink_->pushed_frames() - start, pts_us);
  forget_consumed(sink_->reading().consumed_frames);
  unplayed_.push_back({.start = start,
                       .block = {.pts_us = pts_us,
                                 .channels = format_.channels,
                                 .samples = std::vector<float>(samples.begin(), samples.end())}});
  return {};
}

std::expected<void, Refusal> AudioOutput::push_silence(std::int64_t frames) {
  if (frames <= 0) return {};
  const std::vector<float> silence(static_cast<std::size_t>(frames * format_.channels), 0.0F);
  const std::int64_t start = sink_->pushed_frames();
  if (auto queued = sink_->put(silence); !queued) return queued;
  timeline_.append(start, sink_->pushed_frames() - start, std::nullopt);
  return {};
}

std::int64_t AudioOutput::queued_us() const {
  return frames_to_us(sink_->pushed_frames() - sink_->reading().consumed_frames, format_.sample_rate);
}

std::int64_t AudioOutput::latency_us() const {
  return frames_to_us(sink_->latency_frames(), format_.sample_rate);
}

std::optional<std::int64_t> AudioOutput::position(const DeviceReading& reading) {
  const std::int64_t heard = reading.consumed_frames - sink_->latency_frames();
  const auto pts = timeline_.pts_at(heard);
  timeline_.forget_before(heard);
  forget_consumed(reading.consumed_frames);
  if (!pts) return std::nullopt;
  return *pts + delay_us_;
}

void AudioOutput::pause() {
  sink_->set_paused(true);
  paused_ = true;
}

void AudioOutput::resume() {
  sink_->set_paused(false);
  paused_ = false;
}

std::expected<void, Refusal> AudioOutput::set_volume(float gain) {
  if (!(gain >= limits::kMinVolume && gain <= limits::kMaxVolume)) {  // refuse aussi NaN
    return refuse(RefusalCode::FieldOutOfRange, std::format("audio.volume = {}, attendu de {} à {}", gain,
                                                            limits::kMinVolume, limits::kMaxVolume));
  }
  sink_->set_gain(gain);
  return {};
}

std::expected<void, Refusal> AudioOutput::set_delay(std::int64_t delay_us) {
  if (delay_us < -limits::kMaxDelayUs || delay_us > limits::kMaxDelayUs) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("audio.delay = {} µs, attendu à ±{} µs", delay_us, limits::kMaxDelayUs));
  }
  delay_us_ = delay_us;
  return {};
}

std::expected<void, Refusal> AudioOutput::set_ratio(double ratio) {
  if (!(std::abs(ratio - 1.0) <= limits::kMaxDriftCorrection + kRatioTolerance)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("rapport de fréquence {} hors de 1 ± {}", ratio, limits::kMaxDriftCorrection));
  }
  sink_->set_ratio(ratio);
  ratio_ = ratio;
  return {};
}

void AudioOutput::clear() {
  sink_->clear();
  timeline_.clear();
  unplayed_.clear();
}

std::vector<PcmBlock> AudioOutput::take_back() {
  const std::int64_t consumed = sink_->clear();
  timeline_.clear();
  std::vector<PcmBlock> back;
  for (Unplayed& unplayed : unplayed_) {
    if (unplayed.start + unplayed.block.frames() <= consumed) continue;
    unplayed.block.drop_front(consumed - unplayed.start, format_.sample_rate);
    back.push_back(std::move(unplayed.block));
  }
  unplayed_.clear();
  return back;
}

void AudioOutput::forget_consumed(std::int64_t consumed) {
  while (!unplayed_.empty() && unplayed_.front().start + unplayed_.front().block.frames() <= consumed) {
    unplayed_.pop_front();
  }
}

void AudioOutput::on_demand(std::function<void(const DeviceReading&)> callback) {
  sink_->on_demand(std::move(callback));
}

}  // namespace ajp::audio
