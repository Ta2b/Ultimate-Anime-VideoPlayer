// AudioPlayer branché sur un périphérique simulé : chaque mesure est choisie
// par le test, et le temps est celui du faux périphérique (FakeSink::play).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "audio/player.h"
#include "core/clock.h"
#include "fake_sink.h"
#include "sine.h"

namespace ajp::audio::test {

inline constexpr std::int64_t kLatency = 480;  // tampon du faux périphérique : 10 ms à 48 kHz

class Rig {
 public:
  explicit Rig(const PlayerConfig& config = {}, PcmFormat format = {}, std::int64_t latency = kLatency) {
    auto sink = std::make_unique<FakeSink>(format, latency);
    fake = sink.get();
    player = std::make_unique<AudioPlayer>(
        std::make_unique<AudioOutput>(std::move(sink)), config,
        PlayerEvents{.packet_released = [this] { ++released; },
                     .refused = [this](const Refusal& refusal) { refusals.push_back(refusal); }});
  }
  Rig(const Rig&) = delete;
  Rig& operator=(const Rig&) = delete;

  void feed(const EncodedClip& clip) {
    player->handle(AudioTrack{clip.codec});
    for (AudioPacket packet : clip.packets) player->handle(std::move(packet));
  }
  ClockSnapshot clock() const { return player->clock()->snapshot(); }
  std::int64_t queued_us() const { return player->output().queued_us(); }
  // Instant présent du faux périphérique : le temps de sa dernière mesure.
  std::int64_t now_us() const { return fake->reading().host_us; }

  FakeSink* fake = nullptr;
  std::unique_ptr<AudioPlayer> player;
  std::size_t released = 0;
  std::vector<Refusal> refusals;
};

}  // namespace ajp::audio::test
