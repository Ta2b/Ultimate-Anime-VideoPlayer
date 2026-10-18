// AudioPlayer quand l'écran est maître (rafraîchissement fixe) : calage sur
// l'horloge de present, suivi de la dérive, recalages. Périphérique simulé.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "audio/player.h"
#include "rig.h"

using ajp::Command;
using namespace ajp::audio;
using ajp::audio::test::encode_sine;
using ajp::audio::test::EncodedClip;
using ajp::audio::test::FakeMaster;
using ajp::audio::test::kLatency;
using ajp::audio::test::Rig;
namespace cmd = ajp::cmd;

TEST_CASE("AudioPlayer : écran maître, l'audio en retard est coupé jusqu'à l'image", "[audio][sync]") {
  Rig rig;
  auto master = std::make_shared<FakeMaster>();
  master->set({.media_us = 300'000, .host_us = rig.now_us(), .rate = 0.0, .valid = true});
  rig.player->handle(SyncTo{master});
  rig.feed(encode_sine({.hz = 1'001.0, .seconds = 1.0}));  // sin(2π·1001·0,3) ≠ 0
  CHECK(std::abs(rig.clock().media_us - 300'000) <= 21);
  CHECK(rig.fake->data()[0] != 0.0F);  // aucun silence : on a coupé, pas attendu
}

TEST_CASE("AudioPlayer : écran maître, l'audio en avance attend dans un silence", "[audio][sync]") {
  Rig rig;
  auto master = std::make_shared<FakeMaster>();
  master->set({.media_us = 0, .host_us = rig.now_us(), .rate = 0.0, .valid = true});
  rig.player->handle(SyncTo{master});
  rig.feed(encode_sine({.hz = 1'001.0, .seconds = 1.0, .start_us = 200'000}));
  const auto data = rig.fake->data();
  REQUIRE(data.size() > 9'600 * 2 + 100);
  for (std::size_t i = 0; i < 9'600 * 2; ++i) REQUIRE(data[i] == 0.0F);
  CHECK(data[9'600 * 2 + 2] != 0.0F);
  CHECK(rig.clock().media_us == 200'000);
}

TEST_CASE("AudioPlayer : écran maître, une longue avance part en silence par morceaux bornés",
          "[audio][sync]") {
  Rig rig;
  auto master = std::make_shared<FakeMaster>();
  const std::int64_t t0 = rig.now_us();
  master->set({.media_us = 0, .host_us = t0, .rate = 1.0, .valid = true});
  rig.player->handle(SyncTo{master});
  // L'audio commence 5 s après l'image : 5 s de silence, mais jamais plus que la file en mémoire.
  rig.feed(encode_sine({.hz = 1'001.0, .seconds = 1.0, .start_us = 5'000'000}));
  CHECK(rig.queued_us() == 200'000);
  rig.player->handle(Command{cmd::Play{}});

  std::int64_t longest_queue_us = 0;
  std::int64_t host = t0;
  for (int k = 1; k <= 550; ++k) {  // 5,5 s de lecture, 10 ms par mesure
    host = t0 + k * 10'000;
    rig.player->handle(rig.fake->play(480, host));
    longest_queue_us = std::max(longest_queue_us, rig.queued_us());
  }
  CHECK(longest_queue_us <= 2 * 200'000 + 21'334);  // la cible, le dernier morceau et un bloc
  REQUIRE(rig.clock().valid);
  CHECK(std::abs(rig.clock().media_at(host) - master->snapshot().media_at(host)) <= 21);
  // Le premier échantillon non nul est entendu quand l'écran atteint 5 s : le
  // périphérique joue 480 trames par mesure, et l'indice i s'entend une fois
  // consommées i + kLatency trames.
  const auto data = rig.fake->data();
  const auto first = std::ranges::find_if(data, [](float sample) { return sample != 0.0F; });
  const auto frame = static_cast<std::int64_t>(std::distance(data.begin(), first)) / 2;
  CHECK(std::abs(frames_to_us(frame + kLatency, 48'000) - 5'000'000) <= 21);
}

TEST_CASE("AudioPlayer : écran maître et seek exact, seul l'écart restant devient silence", "[audio][sync]") {
  Rig rig;
  auto master = std::make_shared<FakeMaster>();
  master->set({.media_us = 0, .host_us = rig.now_us(), .rate = 0.0, .valid = true});
  rig.player->handle(SyncTo{master});
  EncodedClip clip = encode_sine({.hz = 1'001.0, .seconds = 1.0});
  clip.packets.front().discard_before_us = 100'000;
  rig.feed(clip);
  // L'écran est à 0 : 100 ms de silence, puis l'audio à partir de 100 ms. Les
  // blocs entièrement avant la cible du seek n'ajoutent aucun silence.
  const auto data = rig.fake->data();
  REQUIRE(data.size() > 4'800 * 2 + 100);
  for (std::size_t i = 0; i < 4'800 * 2; ++i) REQUIRE(data[i] == 0.0F);
  CHECK(data[4'800 * 2 + 2] != 0.0F);
}

TEST_CASE("AudioPlayer : écran maître pas encore valide, l'audio attend sans rien pousser", "[audio][sync]") {
  Rig rig;
  auto master = std::make_shared<FakeMaster>();
  rig.player->handle(SyncTo{master});
  rig.feed(encode_sine({.seconds = 1.0}));
  CHECK(rig.queued_us() == 0);
  master->set({.media_us = 100'000, .host_us = rig.now_us(), .rate = 0.0, .valid = true});
  rig.player->handle(rig.fake->play(0, rig.now_us()));
  CHECK(rig.queued_us() > 0);
  CHECK(std::abs(rig.clock().media_us - 100'000) <= 21);
}

TEST_CASE("AudioPlayer : écran maître, la dérive 1,001 se suit par le rapport, borné", "[audio][sync]") {
  // 8 kHz mono pour une minute de simulation légère ; le périphérique joue 256 trames toutes les 32 ms.
  Rig rig({}, {.sample_rate = 8'000, .channels = 1}, 0);
  const auto clip = encode_sine({.sample_rate = 8'000, .channels = 1, .hz = 440.0, .seconds = 70.0});
  auto master = std::make_shared<FakeMaster>();
  const std::int64_t t0 = rig.now_us();
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

TEST_CASE("AudioPlayer : écran maître, un grand écart recale franchement", "[audio][sync]") {
  Rig rig;
  const auto clip = encode_sine({.seconds = 2.0});
  auto master = std::make_shared<FakeMaster>();
  const std::int64_t t0 = rig.now_us();
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
  const DeviceReading reading = rig.fake->play(static_cast<double>(kLatency), t0 + 52'666);
  rig.player->handle(reading);
  CHECK(std::abs(rig.clock().media_us - master->snapshot().media_at(reading.host_us)) <= 21);
}

TEST_CASE("AudioPlayer : écran maître, le délai recale l'audio sans rien perdre", "[audio][sync]") {
  Rig rig;
  auto master = std::make_shared<FakeMaster>();
  master->set({.media_us = 1'000'000, .host_us = rig.now_us(), .rate = 0.0, .valid = true});
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
  rig.player->handle(rig.fake->play(static_cast<double>(4'800 + kLatency), rig.now_us() + 110'000));
  CHECK(std::abs(rig.clock().media_us - 1'100'000) <= 21);
}

TEST_CASE("AudioPlayer : retour à l'audio maître, le rapport revient à 1 sans vidage", "[audio][sync]") {
  Rig rig;
  auto master = std::make_shared<FakeMaster>();
  const std::int64_t t0 = rig.now_us();
  master->set({.media_us = 0, .host_us = t0, .rate = 1.001, .valid = true});
  rig.player->handle(SyncTo{master});
  rig.feed(encode_sine({.seconds = 2.0}));
  rig.player->handle(Command{cmd::Play{}});
  for (std::int64_t k = 1; k <= 20; ++k) rig.player->handle(rig.fake->play(1'024, t0 + k * 21'333));
  CHECK(rig.fake->ratio() > 1.0);
  const int clears = rig.fake->clears();

  rig.player->handle(SyncTo{});  // écran VRR : l'audio redevient maître
  CHECK(rig.fake->ratio() == 1.0);
  CHECK(rig.fake->clears() == clears);
  const DeviceReading reading = rig.fake->play(1'024, t0 + 21 * 21'333);
  rig.player->handle(reading);
  CHECK(rig.clock().rate == 1.0);
  CHECK(rig.clock().host_us == reading.host_us);
}
