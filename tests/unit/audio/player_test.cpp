// AudioPlayer sur un périphérique simulé, audio maître (VRR) : file, époques,
// seek, réglages et horloge publiée. L'écran maître est dans sync_test.cpp.
#include "audio/player.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "rig.h"

using ajp::ClockSnapshot;
using ajp::Command;
using ajp::Flush;
using ajp::RefusalCode;
using namespace ajp::audio;
using ajp::audio::test::encode_sine;
using ajp::audio::test::EncodedClip;
using ajp::audio::test::kLatency;
using ajp::audio::test::Rig;
namespace cmd = ajp::cmd;

TEST_CASE("AudioPlayer : audio maître, l'horloge publie la position entendue", "[audio][player]") {
  Rig rig;
  rig.feed(encode_sine({.seconds = 1.0, .start_us = 2'000'000}));
  const ClockSnapshot before = rig.clock();  // connue avant d'être entendue, arrêtée
  CHECK(before.valid);
  CHECK(before.rate == 0.0);
  CHECK(before.media_us == 2'000'000);

  rig.player->handle(Command{cmd::Play{}});
  CHECK_FALSE(rig.fake->paused());
  const std::int64_t t = rig.now_us() + 100'000;
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
  const std::int64_t t = rig.now_us() + 100'000;
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
  const std::int64_t t0 = rig.now_us();
  rig.player->handle(Command{cmd::TogglePause{}});
  rig.player->handle(rig.fake->play(4'800, t0 + 100'000));
  rig.player->handle(Command{cmd::Pause{}});
  CHECK(rig.fake->paused());
  const ClockSnapshot paused = rig.clock();
  CHECK(paused.rate == 0.0);
  CHECK(paused.media_us == frames_to_us(4'800 - kLatency, 48'000));
  CHECK(paused.media_at(paused.host_us + 5'000'000) == paused.media_us);
  rig.player->handle(rig.fake->play(4'800, t0 + 5'000'000));  // rien ne joue en pause
  CHECK(rig.clock().media_us == paused.media_us);

  // À la reprise, l'horloge attend que le périphérique joue, puis repart d'où elle était.
  rig.player->handle(Command{cmd::TogglePause{}});
  CHECK_FALSE(rig.fake->paused());
  CHECK(rig.clock().rate == 0.0);
  rig.player->handle(rig.fake->play(0, t0 + 5'030'000));  // démarrage du périphérique
  CHECK(rig.clock().rate == 1.0);
  CHECK(rig.clock().media_us == paused.media_us);
  rig.player->handle(rig.fake->play(480, t0 + 5'040'000));
  CHECK(rig.clock().media_us == paused.media_us + 10'000);
}

TEST_CASE("AudioPlayer : l'horloge audio ne recule pas quand le périphérique tarde à démarrer",
          "[audio][player]") {
  Rig rig;
  rig.feed(encode_sine({.seconds = 1.0, .start_us = 4'000'000}));
  const std::int64_t t0 = rig.now_us();
  rig.player->handle(Command{cmd::Play{}});
  std::int64_t last = rig.clock().media_at(t0);
  // Le périphérique met 30 ms à tirer ses premières trames, puis joue 10 ms toutes les 10 ms.
  rig.player->handle(rig.fake->play(0, t0 + 30'000));
  for (std::int64_t k = 1; k <= 20; ++k) {
    const std::int64_t host = t0 + 30'000 + k * 10'000;
    const std::int64_t before = rig.clock().media_at(host);
    rig.player->handle(rig.fake->play(480, host));
    const std::int64_t after = rig.clock().media_at(host);
    CHECK(after >= before);
    CHECK(after >= last);
    last = after;
  }
  CHECK(last == 4'000'000 + 190'000);  // 20 × 10 ms joués, moins le tampon de 10 ms
}

TEST_CASE("AudioPlayer : un délai posé avant le premier paquet, négatif ou non, laisse l'horloge courir",
          "[audio][player]") {
  for (const std::int64_t delay_us : {std::int64_t{-300'000}, std::int64_t{300'000}}) {
    CAPTURE(delay_us);
    Rig rig;
    rig.player->handle(Command{cmd::SetAudioDelay{.delay_us = delay_us}});  // comme au début d'une lecture
    rig.feed(encode_sine({.seconds = 1.0}));
    CHECK(rig.clock().valid);
    CHECK(rig.clock().media_us == delay_us);
    rig.player->handle(Command{cmd::Play{}});
    const std::int64_t t0 = rig.now_us();
    rig.player->handle(rig.fake->play(0, t0 + 30'000));  // démarrage du périphérique
    for (std::int64_t k = 1; k <= 10; ++k) rig.player->handle(rig.fake->play(480, t0 + 30'000 + k * 10'000));
    const ClockSnapshot now = rig.clock();
    CHECK(now.valid);
    CHECK(now.rate == 1.0);
    CHECK(now.media_us == delay_us + 90'000);  // 10 × 10 ms joués, moins le tampon de 10 ms
  }
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
  rig.player->handle(rig.fake->play(4'800, rig.now_us() + 100'000));
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

TEST_CASE("AudioPlayer : une fin de flux en avance sur son Flush ouvre l'époque", "[audio][player]") {
  Rig rig;
  const auto clip = encode_sine({.seconds = 0.5});
  rig.feed(clip);
  // Seek après le dernier paquet audio : la source n'envoie que la fin de flux.
  rig.player->handle(EndOfStream{.epoch = 1});
  CHECK(rig.fake->clears() == 1);
  CHECK(rig.queued_us() == 0);
  CHECK(rig.player->held_packets() == 0);
  CHECK(rig.released == clip.packets.size());
  CHECK_FALSE(rig.clock().valid);
  CHECK(rig.clock().epoch == 1);
  rig.player->handle(Flush{.epoch = 1});  // déjà appliqué
  CHECK(rig.fake->clears() == 1);
  rig.player->handle(EndOfStream{.epoch = 0});  // d'avant le seek : ignorée
  CHECK(rig.clock().epoch == 1);
}

TEST_CASE("AudioPlayer : seek exact, les échantillons avant la cible sont jetés", "[audio][player]") {
  Rig rig;
  EncodedClip clip = encode_sine({.seconds = 1.0});
  clip.packets.front().discard_before_us = 250'000;
  rig.feed(clip);
  CHECK(std::abs(rig.clock().media_us - 250'000) <= 21);
  CHECK(rig.fake->pushed_frames() <= 48'000 - 12'000);
}

TEST_CASE("AudioPlayer : volume appliqué, ou refusé hors de 0 à 2", "[audio][player]") {
  Rig rig;
  rig.player->handle(Command{cmd::SetVolume{.gain = 0.25F}});
  CHECK(rig.fake->gain() == 0.25F);
  for (const float bad : {3.0F, -0.5F, std::numeric_limits<float>::quiet_NaN()}) {
    rig.player->handle(Command{cmd::SetVolume{.gain = bad}});
  }
  REQUIRE(rig.refusals.size() == 3);
  for (const ajp::Refusal& refusal : rig.refusals) CHECK(refusal.code == RefusalCode::FieldOutOfRange);
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
  const std::int64_t t0 = starving.now_us();
  starving.player->handle(starving.fake->play(48'000, t0 + 200'000));
  const ClockSnapshot stopped = starving.clock();
  CHECK(stopped.rate == 0.0);  // l'image attend l'audio qui manque
  CHECK(stopped.media_us == 100'000 - frames_to_us(kLatency, 48'000));
  // La fin de piste arrive pendant le manque : l'horloge repart sur l'hôte, d'où elle était.
  starving.player->handle(EndOfStream{.epoch = 0});
  const DeviceReading later = starving.fake->play(480, t0 + 210'000);
  starving.player->handle(later);
  CHECK(starving.clock().rate == 1.0);
  CHECK(starving.clock().media_us == stopped.media_us);
  CHECK(starving.clock().host_us == later.host_us);

  Rig ended;
  ended.feed(encode_sine({.seconds = 0.1}));
  ended.player->handle(EndOfStream{.epoch = 0});
  ended.player->handle(Command{cmd::Play{}});
  const std::int64_t t1 = ended.now_us();
  ended.player->handle(ended.fake->play(2'400, t1 + 50'000));
  const ClockSnapshot playing = ended.clock();
  ended.player->handle(ended.fake->play(48'000, t1 + 1'050'000));
  CHECK(ended.clock().rate == 1.0);
  CHECK(ended.clock().media_us == playing.media_us);  // plus de mesure : l'hôte prend le relais
  CHECK(ended.clock().host_us == playing.host_us);

  // Pause puis reprise après la fin : l'horloge repart sur l'hôte depuis la pause.
  ended.player->handle(Command{cmd::Pause{}});
  const ClockSnapshot paused = ended.clock();
  CHECK(paused.rate == 0.0);
  ended.player->handle(Command{cmd::Play{}});
  ended.player->handle(ended.fake->play(480, t1 + 3'000'000));
  CHECK(ended.clock().rate == 1.0);
  CHECK(ended.clock().media_us == paused.media_us);
}
