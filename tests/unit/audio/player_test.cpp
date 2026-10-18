// AudioPlayer sur un périphérique simulé : chaque mesure est choisie par le test.
#include "audio/player.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/clock.h"
#include "fake_sink.h"
#include "sine.h"

using ajp::ClockSnapshot;
using ajp::Command;
using ajp::Flush;
using ajp::host_now_us;
using ajp::Refusal;
using ajp::RefusalCode;
using namespace ajp::audio;
using ajp::audio::test::encode_sine;
using ajp::audio::test::EncodedClip;
using ajp::audio::test::FakeMaster;
using ajp::audio::test::FakeSink;
namespace cmd = ajp::cmd;

namespace {

constexpr std::int64_t kLatency = 480;  // 10 ms à 48 kHz

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

  FakeSink* fake = nullptr;
  std::unique_ptr<AudioPlayer> player;
  std::size_t released = 0;
  std::vector<Refusal> refusals;
};

}  // namespace

TEST_CASE("AudioPlayer : audio maître, l'horloge publie la position entendue", "[audio][player]") {
  Rig rig;
  rig.feed(encode_sine({.seconds = 1.0, .start_us = 2'000'000}));
  const ClockSnapshot before = rig.clock();  // connue avant d'être entendue, arrêtée
  CHECK(before.valid);
  CHECK(before.rate == 0.0);
  CHECK(before.media_us == 2'000'000);

  rig.player->handle(Command{cmd::Play{}});
  CHECK_FALSE(rig.fake->paused());
  const std::int64_t t = host_now_us();
  rig.player->handle(rig.fake->play(4'800, t));
  const ClockSnapshot now = rig.clock();
  CHECK(now.rate == 1.0);
  CHECK(now.host_us == t);
  CHECK(now.media_us == 2'000'000 + frames_to_us(4'800 - kLatency, 48'000));
  CHECK(now.media_at(t + 50'000) == now.media_us + 50'000);
}

TEST_CASE("AudioPlayer : le délai décale l'horloge audio, tout de suite", "[audio][player]") {
  Rig rig;
  rig.feed(encode_sine({.seconds = 1.0}));
  rig.player->handle(Command{cmd::Play{}});
  const std::int64_t t = host_now_us();
  rig.player->handle(rig.fake->play(4'800, t));
  const std::int64_t heard = rig.clock().media_us;

  rig.player->handle(Command{cmd::SetAudioDelay{.delay_us = 100'000}});
  CHECK(rig.clock().media_us == heard + 100'000);
  rig.player->handle(rig.fake->play(4'800, t + 100'000));
  CHECK(rig.clock().media_us == frames_to_us(9'600 - kLatency, 48'000) + 100'000);

  rig.player->handle(Command{cmd::SetAudioDelay{.delay_us = 20'000'000}});
  REQUIRE(rig.refusals.size() == 1);
  CHECK(rig.refusals[0].code == RefusalCode::FieldOutOfRange);
  CHECK(rig.player->output().delay_us() == 100'000);
}

TEST_CASE("AudioPlayer : la pause arrête le périphérique et l'horloge", "[audio][player]") {
  Rig rig;
  rig.feed(encode_sine({.seconds = 1.0}));
  rig.player->handle(Command{cmd::TogglePause{}});
  rig.player->handle(rig.fake->play(4'800, host_now_us()));
  rig.player->handle(Command{cmd::Pause{}});
  CHECK(rig.fake->paused());
  const ClockSnapshot paused = rig.clock();
  CHECK(paused.rate == 0.0);
  CHECK(paused.media_at(paused.host_us + 5'000'000) == paused.media_us);
  rig.player->handle(rig.fake->play(4'800, host_now_us()));  // rien ne joue en pause
  CHECK(rig.clock().media_us == paused.media_us);

  rig.player->handle(Command{cmd::TogglePause{}});
  CHECK_FALSE(rig.fake->paused());
  CHECK(rig.clock().rate == 1.0);
  CHECK(rig.clock().media_us == paused.media_us);
}

TEST_CASE("AudioPlayer : la file reste à sa cible, les paquets attendent leur tour", "[audio][player]") {
  Rig rig({.queue_target_us = 50'000});
  const auto clip = encode_sine({.seconds = 1.0});
  rig.feed(clip);
  CHECK(rig.queued_us() >= 50'000);
  CHECK(rig.queued_us() < 50'000 + 21'334);  // un paquet de 1024 trames au plus au-delà
  CHECK(rig.player->held_packets() + rig.released == clip.packets.size());
  CHECK(rig.player->held_packets() > 30);

  rig.player->handle(Command{cmd::Play{}});
  const std::size_t released = rig.released;
  rig.player->handle(rig.fake->play(4'800, host_now_us()));
  CHECK(rig.released > released);  // la place libérée a fait entrer d'autres paquets
  CHECK(rig.queued_us() >= 50'000);
}

TEST_CASE("AudioPlayer : Flush jette l'époque passée et rend toutes les places", "[audio][player]") {
  Rig rig;
  const auto clip = encode_sine({.seconds = 1.0});
  rig.feed(clip);
  REQUIRE(rig.player->held_packets() > 0);
  rig.player->handle(Flush{.epoch = 1});
  CHECK(rig.player->held_packets() == 0);
  CHECK(rig.released == clip.packets.size());
  CHECK(rig.fake->clears() == 1);
  CHECK(rig.queued_us() == 0);
  CHECK_FALSE(rig.clock().valid);
  CHECK(rig.clock().epoch == 1);

  AudioPacket late = clip.packets.at(5);  // époque 0, arrivé après le Flush
  rig.player->handle(std::move(late));
  CHECK(rig.queued_us() == 0);
  CHECK(rig.released == clip.packets.size() + 1);

  for (AudioPacket packet : encode_sine({.seconds = 0.2, .start_us = 30'000'000, .epoch = 1}).packets) {
    rig.player->handle(std::move(packet));
  }
  CHECK(rig.queued_us() > 0);
  CHECK(rig.clock().valid);
  CHECK(rig.clock().media_us == 30'000'000);
}

TEST_CASE("AudioPlayer : des paquets en avance sur leur Flush ouvrent l'époque", "[audio][player]") {
  Rig rig;
  rig.feed(encode_sine({.seconds = 0.5}));
  for (AudioPacket packet : encode_sine({.seconds = 0.5, .start_us = 10'000'000, .epoch = 2}).packets) {
    rig.player->handle(std::move(packet));
  }
  const std::int64_t queued = rig.queued_us();
  CHECK(queued > 0);
  CHECK(rig.clock().media_us == 10'000'000);
  rig.player->handle(Flush{.epoch = 2});  // déjà appliqué : ne jette rien
  rig.player->handle(Flush{.epoch = 1});
  CHECK(rig.queued_us() == queued);
  CHECK(rig.fake->clears() == 1);
}

TEST_CASE("AudioPlayer : seek exact, les échantillons avant la cible sont jetés", "[audio][player]") {
  Rig rig;
  EncodedClip clip = encode_sine({.seconds = 1.0});
  clip.packets.front().discard_before_us = 250'000;
  rig.feed(clip);
  CHECK(std::abs(rig.clock().media_us - 250'000) <= 21);
  CHECK(rig.fake->pushed_frames() <= 48'000 - 12'000);
}

TEST_CASE("AudioPlayer : écran maître, l'audio en retard est coupé jusqu'à l'image", "[audio][player]") {
  Rig rig;
  auto master = std::make_shared<FakeMaster>();
  master->set({.media_us = 300'000, .host_us = host_now_us(), .rate = 0.0, .valid = true});
  rig.player->handle(SyncTo{master});
  rig.feed(encode_sine({.hz = 1'001.0, .seconds = 1.0}));  // sin(2π·1001·0,3) ≠ 0
  CHECK(std::abs(rig.clock().media_us - 300'000) <= 21);
  CHECK(rig.fake->data()[0] != 0.0F);  // aucun silence : on a coupé, pas attendu
}

TEST_CASE("AudioPlayer : écran maître, l'audio en avance attend dans un silence", "[audio][player]") {
  Rig rig;
  auto master = std::make_shared<FakeMaster>();
  master->set({.media_us = 0, .host_us = host_now_us(), .rate = 0.0, .valid = true});
  rig.player->handle(SyncTo{master});
  rig.feed(encode_sine({.hz = 1'001.0, .seconds = 1.0, .start_us = 200'000}));
  const auto data = rig.fake->data();
  REQUIRE(data.size() > 9'600 * 2 + 100);
  for (std::size_t i = 0; i < 9'600 * 2; ++i) REQUIRE(data[i] == 0.0F);
  CHECK(data[9'600 * 2 + 2] != 0.0F);
  CHECK(rig.clock().media_us == 200'000);
}

TEST_CASE("AudioPlayer : écran maître pas encore valide, l'audio attend sans rien pousser",
          "[audio][player]") {
  Rig rig;
  auto master = std::make_shared<FakeMaster>();
  rig.player->handle(SyncTo{master});
  rig.feed(encode_sine({.seconds = 1.0}));
  CHECK(rig.queued_us() == 0);
  master->set({.media_us = 100'000, .host_us = host_now_us(), .rate = 0.0, .valid = true});
  rig.player->handle(rig.fake->play(0, host_now_us()));
  CHECK(rig.queued_us() > 0);
  CHECK(std::abs(rig.clock().media_us - 100'000) <= 21);
}

TEST_CASE("AudioPlayer : écran maître, la dérive 1,001 se suit par le rapport, borné", "[audio][player]") {
  // 8 kHz mono pour une minute de simulation légère ; le périphérique joue 256 trames toutes les 32 ms.
  Rig rig({}, {.sample_rate = 8'000, .channels = 1}, 0);
  const auto clip = encode_sine({.sample_rate = 8'000, .channels = 1, .hz = 440.0, .seconds = 70.0});
  auto master = std::make_shared<FakeMaster>();
  const std::int64_t t0 = host_now_us();
  master->set({.media_us = 0, .host_us = t0, .rate = 1.001, .valid = true});
  rig.player->handle(SyncTo{master});
  rig.feed(clip);
  rig.player->handle(Command{cmd::Play{}});

  double worst = 0.0;
  std::int64_t error_us = 0;
  for (std::int64_t k = 1; k <= 60'000 / 32; ++k) {
    const std::int64_t host = t0 + k * 32'000;
    rig.player->handle(rig.fake->play(256, host));
    worst = std::max(worst, std::abs(rig.fake->ratio() - 1.0));
    error_us = rig.clock().media_at(host) - master->snapshot().media_at(host);
  }
  CHECK(worst <= limits::kMaxDriftCorrection + 1e-12);
  CHECK(rig.fake->ratio() > 1.0005);  // l'audio lit plus vite pour suivre l'écran
  CHECK(std::abs(error_us) < 5'000);
  CHECK(rig.fake->clears() == 1);  // le seul vidage est celui du passage à l'écran maître
}

TEST_CASE("AudioPlayer : écran maître, un grand écart recale franchement", "[audio][player]") {
  Rig rig;
  const auto clip = encode_sine({.seconds = 2.0});
  auto master = std::make_shared<FakeMaster>();
  const std::int64_t t0 = host_now_us();
  master->set({.media_us = 0, .host_us = t0, .rate = 1.0, .valid = true});
  rig.player->handle(SyncTo{master});
  rig.feed(clip);
  rig.player->handle(Command{cmd::Play{}});
  rig.player->handle(rig.fake->play(1'024, t0 + 21'333));
  const int clears = rig.fake->clears();

  master->set({.media_us = 500'000, .host_us = t0 + 21'333, .rate = 1.0, .valid = true});  // l'écran saute
  rig.player->handle(rig.fake->play(1'024, t0 + 42'666));
  CHECK(rig.fake->clears() == clears + 1);
  // Le bloc suivant est entendu quand l'écran atteint son temps.
  const DeviceReading reading = rig.fake->play(static_cast<double>(kLatency), host_now_us());
  rig.player->handle(reading);
  CHECK(std::abs(rig.clock().media_us - master->snapshot().media_at(reading.host_us)) < 30'000);
}

TEST_CASE("AudioPlayer : écran maître, le délai recale l'audio sans rien perdre", "[audio][player]") {
  Rig rig;
  auto master = std::make_shared<FakeMaster>();
  master->set({.media_us = 1'000'000, .host_us = host_now_us(), .rate = 0.0, .valid = true});
  rig.player->handle(SyncTo{master});
  rig.feed(encode_sine({.hz = 1'001.0, .seconds = 2.0}));
  const auto pushed = static_cast<std::size_t>(rig.fake->pushed_frames());

  rig.player->handle(Command{cmd::SetAudioDelay{.delay_us = 100'000}});
  // L'audio de 1 s, déjà en file, revient après 100 ms de silence : il sera
  // entendu quand l'écran montrera 1,1 s.
  const auto data = rig.fake->data();
  for (std::size_t i = 0; i < 4'800 * 2; ++i) REQUIRE(data[pushed * 2 + i] == 0.0F);
  for (std::size_t i = 0; i < 2'000; ++i) REQUIRE(data[(pushed + 4'800) * 2 + i] == data[i]);
  rig.player->handle(Command{cmd::Play{}});
  rig.player->handle(rig.fake->play(static_cast<double>(4'800 + kLatency), host_now_us()));
  CHECK(std::abs(rig.clock().media_us - 1'100'000) <= 21);
}

TEST_CASE("AudioPlayer : volume appliqué, ou refusé hors de 0 à 2", "[audio][player]") {
  Rig rig;
  rig.player->handle(Command{cmd::SetVolume{.gain = 0.25F}});
  CHECK(rig.fake->gain() == 0.25F);
  rig.player->handle(Command{cmd::SetVolume{.gain = 3.0F}});
  REQUIRE(rig.refusals.size() == 1);
  CHECK(rig.refusals[0].code == RefusalCode::FieldOutOfRange);
  CHECK(rig.fake->gain() == 0.25F);
}

TEST_CASE("AudioPlayer : piste sans décodeur refusée, ses paquets rendent leur place", "[audio][player]") {
  Rig rig;
  rig.player->handle(AudioTrack{.codec = {.codec = "inexistant"}});
  REQUIRE(rig.refusals.size() == 1);
  CHECK(rig.refusals[0].code == RefusalCode::AudioCodecUnavailable);
  rig.player->handle(AudioPacket{.data = {1, 2, 3, 4}, .pts_us = 0});
  CHECK(rig.released == 1);
  CHECK(rig.queued_us() == 0);
}

TEST_CASE("AudioPlayer : à court d'audio l'horloge attend, en fin de piste elle continue",
          "[audio][player]") {
  Rig starving;
  starving.feed(encode_sine({.seconds = 0.1}));
  starving.player->handle(Command{cmd::Play{}});
  starving.player->handle(starving.fake->play(48'000, host_now_us()));
  CHECK(starving.clock().rate == 0.0);  // l'image attend l'audio qui manque

  Rig ended;
  ended.feed(encode_sine({.seconds = 0.1}));
  ended.player->handle(EndOfStream{.epoch = 0});
  ended.player->handle(Command{cmd::Play{}});
  ended.player->handle(ended.fake->play(2'400, host_now_us()));
  const ClockSnapshot playing = ended.clock();
  ended.player->handle(ended.fake->play(48'000, host_now_us()));
  CHECK(ended.clock().rate == 1.0);
  CHECK(ended.clock().media_us == playing.media_us);  // plus de mesure : l'hôte prend le relais
}
