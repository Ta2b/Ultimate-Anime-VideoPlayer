// Le thread audio de bout en bout, sur le pilote SDL « dummy » qui consomme en temps réel.
#include <atomic>
#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "audio/audio_thread.h"
#include "core/clock.h"
#include "dummy_audio.h"
#include "sine.h"

using ajp::ClockSnapshot;
using ajp::Command;
using ajp::Credit;
using ajp::host_now_us;
using ajp::RefusalCode;
using namespace ajp::audio;
using ajp::audio::test::encode_sine;
using ajp::audio::test::use_dummy_audio;
using namespace std::chrono_literals;
namespace cmd = ajp::cmd;

namespace {

std::unique_ptr<AudioOutput> open_dummy() {
  use_dummy_audio();
  auto output = AudioOutput::open_sdl({.sample_rate = 48'000, .channels = 2});
  REQUIRE(output);
  return std::move(*output);
}

// Attend qu'une condition devienne vraie, au plus timeout.
template <class Condition>
bool eventually(Condition condition, std::chrono::milliseconds timeout = 2'000ms) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!condition()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(5ms);
  }
  return true;
}

}  // namespace

TEST_CASE("AudioThread : un sinus joue, l'horloge avance au rythme du périphérique", "[audio][thread][sdl]") {
  auto thread = AudioThread::start(open_dummy(), {});
  REQUIRE(thread);
  const auto clip = encode_sine({.seconds = 2.0, .start_us = 1'000'000});
  REQUIRE((*thread)->post(AudioTrack{clip.codec}));
  for (AudioPacket packet : clip.packets) REQUIRE((*thread)->push_packet(std::move(packet)));
  const auto clock = (*thread)->clock();
  REQUIRE(eventually([&] { return clock->snapshot().valid; }));
  CHECK(clock->snapshot().media_us == 1'000'000);  // position connue, pas encore entendue

  const std::int64_t start = host_now_us();
  REQUIRE((*thread)->post(Command{cmd::Play{}}));
  std::this_thread::sleep_for(400ms);
  const ClockSnapshot playing = clock->snapshot();
  CHECK(playing.rate == 1.0);
  const std::int64_t elapsed = host_now_us() - start;
  const std::int64_t advanced = playing.media_at(host_now_us()) - 1'000'000;
  CHECK(advanced > elapsed - 150'000);  // démarrage du périphérique et son tampon
  CHECK(advanced < elapsed + 30'000);

  REQUIRE((*thread)->post(Command{cmd::Pause{}}));
  REQUIRE(eventually([&] { return clock->snapshot().rate == 0.0; }));
  const std::int64_t paused_at = clock->snapshot().media_at(host_now_us());
  std::this_thread::sleep_for(150ms);
  CHECK(clock->snapshot().media_at(host_now_us()) == paused_at);
}

TEST_CASE("AudioThread : Flush et fin de flux passent par la boîte, l'horloge suit l'époque",
          "[audio][thread][sdl]") {
  auto thread = AudioThread::start(open_dummy(), {.max_packets = 64});
  REQUIRE(thread);
  const auto clip = encode_sine({.seconds = 1.0});
  REQUIRE((*thread)->post(AudioTrack{clip.codec}));
  for (AudioPacket packet : clip.packets) REQUIRE((*thread)->push_packet(std::move(packet)));
  const auto clock = (*thread)->clock();
  REQUIRE(eventually([&] { return clock->snapshot().valid; }));
  REQUIRE((*thread)->post(Command{cmd::Play{}}));

  // Seek : l'époque 0 est jetée, ses places reviennent, l'époque 1 reprend à 30 s.
  REQUIRE((*thread)->post(ajp::Flush{.epoch = 1}));
  const auto later = encode_sine({.seconds = 0.3, .start_us = 30'000'000, .epoch = 1});
  for (AudioPacket packet : later.packets) {
    REQUIRE(eventually([&] { return (*thread)->push_packet(AudioPacket{packet}); }));
  }
  REQUIRE((*thread)->post(EndOfStream{.epoch = 1}));
  REQUIRE(eventually([&] {
    const ClockSnapshot now = clock->snapshot();
    return now.valid && now.epoch == 1 && now.rate == 1.0;
  }));
  const ClockSnapshot playing = clock->snapshot();
  CHECK(playing.media_us >= 30'000'000);
  CHECK(playing.media_us < 30'300'000);
  // Après les 300 ms de la piste, l'horloge continue sur l'hôte au lieu de s'arrêter.
  std::this_thread::sleep_for(500ms);
  const ClockSnapshot ended = clock->snapshot();
  CHECK(ended.rate == 1.0);
  CHECK(ended.media_at(host_now_us()) > 30'350'000);
}

TEST_CASE("AudioThread : borne des paquets atteinte, un crédit revient quand une place se libère",
          "[audio][thread][sdl]") {
  std::atomic<int> credits{0};
  auto thread = AudioThread::start(
      open_dummy(),
      {.max_packets = 4, .player = {.queue_target_us = 20'000}, .credit = [&credits](Credit) { ++credits; }});
  REQUIRE(thread);
  const auto clip = encode_sine({.seconds = 1.0});
  REQUIRE((*thread)->post(AudioTrack{clip.codec}));
  std::size_t sent = 0;
  while (sent < clip.packets.size()) {
    AudioPacket packet = clip.packets[sent];
    if (!(*thread)->push_packet(std::move(packet))) break;
    ++sent;
  }
  // Le périphérique est en pause : la borne et un paquet décodé, puis plus rien.
  CHECK(sent >= 4);
  CHECK(sent < 12);
  REQUIRE(sent < clip.packets.size());

  // Le producteur suit le contrat : après un refus, chaque Credit vaut un essai.
  // Un Credit peut déjà être là (place libérée pendant le remplissage, aussitôt
  // reprise par la boucle) : cet essai échoue, le Credit suivant rend la place.
  REQUIRE((*thread)->post(Command{cmd::Play{}}));
  int tried = 0;  // crédits reçus au dernier essai
  CHECK(eventually([&] {
    const int received = credits.load();
    if (received == tried) return false;
    tried = received;
    return (*thread)->push_packet(AudioPacket{clip.packets[sent]});
  }));
}

TEST_CASE("AudioThread : réglages refusés hors bornes, signalés à l'OSD", "[audio][thread][sdl]") {
  const auto no_packets = AudioThread::start(open_dummy(), {.max_packets = 0});
  REQUIRE_FALSE(no_packets);
  CHECK(no_packets.error().code == RefusalCode::FieldOutOfRange);
  const auto tiny_queue = AudioThread::start(open_dummy(), {.player = {.queue_target_us = 1'000}});
  REQUIRE_FALSE(tiny_queue);
  CHECK(tiny_queue.error().code == RefusalCode::FieldOutOfRange);
  const auto no_output = AudioThread::start(nullptr, {});
  REQUIRE_FALSE(no_output);
  CHECK(no_output.error().code == RefusalCode::AudioDeviceUnavailable);

  std::atomic<int> refusals{0};
  auto thread =
      AudioThread::start(open_dummy(), {.refused = [&refusals](const ajp::Refusal&) { ++refusals; }});
  REQUIRE(thread);
  REQUIRE((*thread)->post(Command{cmd::SetVolume{.gain = 2.5F}}));
  REQUIRE((*thread)->post(AudioTrack{.codec = {.codec = "inexistant"}}));
  CHECK(eventually([&] { return refusals.load() == 2; }));
}

TEST_CASE("AudioThread : l'arrêt par stop_token est immédiat, même en pleine lecture",
          "[audio][thread][sdl]") {
  auto thread = AudioThread::start(open_dummy(), {});
  REQUIRE(thread);
  const auto clip = encode_sine({.seconds = 1.0});
  REQUIRE((*thread)->post(AudioTrack{clip.codec}));
  for (AudioPacket packet : clip.packets) REQUIRE((*thread)->push_packet(std::move(packet)));
  REQUIRE((*thread)->post(Command{cmd::Play{}}));
  std::this_thread::sleep_for(50ms);
  const auto before = std::chrono::steady_clock::now();
  thread->reset();
  CHECK(std::chrono::steady_clock::now() - before < 500ms);
}
