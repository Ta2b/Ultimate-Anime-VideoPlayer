#include "audio/output.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/clock.h"
#include "dummy_audio.h"
#include "fake_sink.h"

using ajp::host_now_us;
using ajp::RefusalCode;
using namespace ajp::audio;
using ajp::audio::test::FakeSink;
using ajp::audio::test::use_dummy_audio;
using namespace std::chrono_literals;

namespace {

std::vector<float> frames_of(std::int64_t frames, int channels, float value = 0.25F) {
  return std::vector<float>(static_cast<std::size_t>(frames * channels), value);
}

std::unique_ptr<AudioOutput> open_dummy() {
  use_dummy_audio();
  auto output = AudioOutput::open_sdl({.sample_rate = 48'000, .channels = 2});
  REQUIRE(output);
  return std::move(*output);
}

}  // namespace

TEST_CASE("AudioOutput : position = poussées - en file - tampon du périphérique, délai compris",
          "[audio][output]") {
  auto sink = std::make_unique<FakeSink>(PcmFormat{.sample_rate = 48'000, .channels = 2}, 480);
  FakeSink& fake = *sink;
  AudioOutput output(std::move(sink));
  REQUIRE(output.push(2'000'000, frames_of(4'800, 2)));
  CHECK(output.queued_us() == 100'000);
  output.resume();
  const DeviceReading reading = fake.play(2'400, host_now_us());
  CHECK(output.queued_us() == 50'000);
  CHECK(output.position(reading) == 2'040'000);  // (2400 - 480) trames entendues
  REQUIRE(output.set_delay(150'000));
  CHECK(output.position(reading) == 2'190'000);
  REQUIRE(output.set_delay(-150'000));
  CHECK(output.position(reading) == 1'890'000);
}

TEST_CASE("AudioOutput : délai hors de ±10 s refusé, l'ancien reste", "[audio][output]") {
  AudioOutput output(std::make_unique<FakeSink>(PcmFormat{}));
  REQUIRE(output.set_delay(10'000'000));
  const auto refused = output.set_delay(10'000'001);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
  CHECK(output.delay_us() == 10'000'000);
  CHECK_FALSE(output.set_delay(-10'000'001));
}

TEST_CASE("AudioOutput : un silence de calage ne donne aucune position", "[audio][output]") {
  auto sink = std::make_unique<FakeSink>(PcmFormat{.sample_rate = 48'000, .channels = 2});
  FakeSink& fake = *sink;
  AudioOutput output(std::move(sink));
  REQUIRE(output.push_silence(480));
  REQUIRE(output.push(1'000'000, frames_of(4'800, 2)));
  output.resume();
  CHECK_FALSE(output.position(fake.play(240, host_now_us())));
  CHECK(output.position(fake.play(720, host_now_us())) == 1'010'000);
  CHECK(fake.data()[0] == 0.0F);
  CHECK(fake.data()[480 * 2] == 0.25F);
}

TEST_CASE("AudioOutput : après un vidage, une mesure d'avant est périmée", "[audio][output]") {
  auto sink = std::make_unique<FakeSink>(PcmFormat{.sample_rate = 48'000, .channels = 2});
  FakeSink& fake = *sink;
  AudioOutput output(std::move(sink));
  output.resume();
  REQUIRE(output.push(0, frames_of(4'800, 2)));
  const DeviceReading stale = fake.play(1'000, host_now_us());
  output.clear();
  CHECK(fake.clears() == 1);
  CHECK(output.queued_us() == 0);
  REQUIRE(output.push(9'000'000, frames_of(4'800, 2)));
  CHECK_FALSE(output.position(stale));
  CHECK(output.position(fake.play(480, host_now_us())) == 9'010'000);
}

TEST_CASE("AudioOutput : take_back rend l'audio pas encore consommé, sans le silence", "[audio][output]") {
  auto sink = std::make_unique<FakeSink>(PcmFormat{.sample_rate = 48'000, .channels = 2});
  FakeSink& fake = *sink;
  AudioOutput output(std::move(sink));
  output.resume();
  REQUIRE(output.push_silence(480));
  REQUIRE(output.push(0, frames_of(4'800, 2, 0.5F)));
  REQUIRE(output.push(100'000, frames_of(4'800, 2, 0.75F)));
  fake.play(480 + 2'400, host_now_us());  // le silence et la moitié du premier bloc
  const std::vector<PcmBlock> back = output.take_back();
  REQUIRE(back.size() == 2);
  CHECK(back[0].pts_us == 50'000);
  CHECK(back[0].frames() == 2'400);
  CHECK(back[0].samples.front() == 0.5F);
  CHECK(back[1].pts_us == 100'000);
  CHECK(back[1].frames() == 4'800);
  CHECK(output.queued_us() == 0);
  CHECK(output.take_back().empty());
}

TEST_CASE("AudioOutput : volume de 0 à 2 et rapport à 0,125 % près, sinon refusés", "[audio][output]") {
  auto sink = std::make_unique<FakeSink>(PcmFormat{});
  FakeSink& fake = *sink;
  AudioOutput output(std::move(sink));
  REQUIRE(output.set_volume(1.5F));
  CHECK(fake.gain() == 1.5F);
  CHECK(output.set_volume(0.0F));
  CHECK(output.set_volume(2.0F));
  for (const float bad : {2.01F, -0.1F, std::numeric_limits<float>::quiet_NaN()}) {
    const auto refused = output.set_volume(bad);
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
  }
  CHECK(fake.gain() == 2.0F);

  CHECK(output.set_ratio(1.0 + limits::kMaxDriftCorrection));
  CHECK(output.set_ratio(1.0 - limits::kMaxDriftCorrection));
  REQUIRE(output.set_ratio(1.001));
  CHECK(fake.ratio() == 1.001);
  CHECK_FALSE(output.set_ratio(1.002));
  CHECK_FALSE(output.set_ratio(0.998));
  CHECK(output.ratio() == 1.001);
}

TEST_CASE("AudioOutput SDL : ouverture du pilote dummy au format demandé ou préféré",
          "[audio][output][sdl]") {
  use_dummy_audio();
  auto preferred = AudioOutput::open_sdl();
  REQUIRE(preferred);
  CHECK(checked((*preferred)->format()));
  CHECK((*preferred)->paused());
  auto exact = AudioOutput::open_sdl({.sample_rate = 48'000, .channels = 6});
  REQUIRE(exact);
  CHECK((*exact)->format() == PcmFormat{.sample_rate = 48'000, .channels = 6});
  const auto refused = AudioOutput::open_sdl({.sample_rate = 48'000, .channels = 9});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("AudioOutput SDL : la position suit le périphérique, la pause l'arrête", "[audio][output][sdl]") {
  const auto output = open_dummy();
  REQUIRE(output->push(1'000'000, frames_of(48'000, 2)));
  std::this_thread::sleep_for(50ms);
  CHECK(output->queued_us() == 1'000'000);  // en pause, rien ne part

  const std::int64_t start = host_now_us();
  output->resume();
  std::this_thread::sleep_for(300ms);
  const DeviceReading reading = output->reading();
  const std::int64_t consumed_us = frames_to_us(reading.consumed_frames, 48'000);
  CHECK(consumed_us > 100'000);
  CHECK(consumed_us < reading.host_us - start + 50'000);  // un tampon d'avance au plus
  const auto heard = output->position(reading);
  REQUIRE(heard);
  CHECK(std::abs(*heard - (1'000'000 + consumed_us - output->latency_us())) <= 1);

  output->pause();
  std::this_thread::sleep_for(50ms);
  const std::int64_t paused_at = output->reading().consumed_frames;
  std::this_thread::sleep_for(150ms);
  CHECK(output->reading().consumed_frames == paused_at);
}

TEST_CASE("AudioOutput SDL : le périphérique rappelle à chaque demande, mesure cohérente",
          "[audio][output][sdl]") {
  const auto output = open_dummy();
  std::mutex mutex;
  std::vector<DeviceReading> readings;
  output->on_demand([&](const DeviceReading& reading) {
    const std::scoped_lock lock(mutex);
    readings.push_back(reading);
  });
  REQUIRE(output->push(0, frames_of(24'000, 2)));
  output->resume();
  std::this_thread::sleep_for(250ms);
  output->on_demand({});
  const std::int64_t pushed = output->pushed_frames();

  const std::scoped_lock lock(mutex);
  REQUIRE(readings.size() >= 3);
  for (std::size_t i = 1; i < readings.size(); ++i) {
    CHECK(readings[i].consumed_frames >= readings[i - 1].consumed_frames);
    CHECK(readings[i].host_us >= readings[i - 1].host_us);
  }
  CHECK(readings.back().consumed_frames <= pushed);
}

TEST_CASE("AudioOutput SDL : volume relu dans la file SDL, vidage au seek", "[audio][output][sdl]") {
  const auto output = open_dummy();
  REQUIRE(output->set_volume(0.5F));
  CHECK(output->volume() == 0.5F);
  REQUIRE(output->set_ratio(1.00125));
  REQUIRE(output->push(0, frames_of(48'000, 2)));
  CHECK(output->queued_us() == 1'000'000);
  output->clear();
  CHECK(output->queued_us() == 0);
  CHECK(output->pushed_frames() == 48'000);  // les indices ne reculent pas
}
