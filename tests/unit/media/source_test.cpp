#include "media/source.h"

#include <string_view>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "clips.h"
#include "source_harness.h"

using namespace ajp::media;
using namespace ajp::media::test;
using ajp::Command;
using ajp::Credit;
using ajp::DecodedFrame;
using ajp::Flush;
using ajp::RefusalCode;
namespace cmd = ajp::cmd;

namespace {

// threads 0 : automatique, jusqu'à kAutoDecoderThreads selon la machine.
SourceMedia open_media(const ClipSpec& spec, int threads = 0) {
  auto demuxer = Demuxer::open(require_clip(spec));
  REQUIRE(demuxer.has_value());
  auto media =
      open_source(std::move(*demuxer), 0, DecoderOptions{.attempts = {HwApi::Software}, .threads = threads});
  REQUIRE(media.has_value());
  return std::move(*media);
}

// Une source en marche avec ses consommateurs de test.
struct Running {
  SourceInbox inbox{kSourceInboxBounds};
  TestSink sink;
  TestListener listener;
  std::unique_ptr<SourceThread> thread;

  Running(const ClipSpec& spec, std::size_t capacity, int refuse_first = 0,
          std::vector<PacketRoute> routes = {}, int threads = 0,
          std::size_t reserve_bytes = kDefaultReserveBytes)
      : sink(capacity, refuse_first) {
    auto started = SourceThread::start(open_media(spec, threads),
                                       {.frames = &sink,
                                        .listener = &listener,
                                        .packets = std::move(routes),
                                        .reserve_bytes = reserve_bytes},
                                       inbox);
    REQUIRE(started.has_value());
    thread = std::move(*started);
  }

  DecodedFrame pop() {
    auto frame = sink.pop(inbox);
    REQUIRE(frame.has_value());
    return std::move(*frame);
  }

  // Première image de l'époque voulue ; les images d'une époque passée sont jetées comme gpu le ferait.
  DecodedFrame pop_epoch(std::uint32_t epoch) {
    while (true) {
      DecodedFrame frame = pop();
      if (frame.epoch == epoch) return frame;
      REQUIRE(frame.epoch < epoch);
    }
  }
};

int luma_frame(const DecodedFrame& frame) {
  const auto plane = frame.data->host_plane(0);
  return frame_of_flat_luma(static_cast<int>(plane[20 * frame.data->host_stride(0) + 30]));
}

}  // namespace

TEST_CASE("SourceThread : toutes les images, dans l'ordre, identités croissantes", "[media][source]") {
  Running source(full_clip(), 2);
  std::uint32_t previous_id = 0;
  for (int n = 0; n < 30; ++n) {
    const DecodedFrame frame = source.pop();
    CHECK(frame.pts_us == n * kFrameUs);
    CHECK(frame.epoch == 0);
    CHECK(frame.id.value > previous_id);
    previous_id = frame.id.value;
  }
  CHECK(source.listener.wait_end(0));
  CHECK_FALSE(source.sink.pop(source.inbox, 50ms).has_value());
  CHECK(source.listener.failure_count() == 0);
}

TEST_CASE("SourceThread : un puits qui refuse fait attendre un Credit, sans boucle active",
          "[media][source]") {
  Running source(full_clip(), 8, 3);
  REQUIRE(source.sink.wait_refusals(1));
  std::this_thread::sleep_for(50ms);
  CHECK(source.sink.attempts() == 1);  // pas de nouvel essai sans Credit
  CHECK(source.sink.queued() == 0);

  for (int refusal = 2; refusal <= 3; ++refusal) {
    REQUIRE(source.inbox.try_push(Credit{}));
    REQUIRE(source.sink.wait_refusals(refusal));
  }
  REQUIRE(source.inbox.try_push(Credit{}));  // le quatrième essai passe
  for (int n = 0; n < 30; ++n) CHECK(source.pop().pts_us == n * kFrameUs);
  CHECK(source.listener.wait_end(0));
}

TEST_CASE("SourceThread : Flush puis seek exact livrent l'image visible à la cible", "[media][source]") {
  Running source(seek_clip(), 2);
  std::uint32_t last_id = 0;
  for (int n = 0; n < 3; ++n) last_id = source.pop().id.value;

  // Au milieu de l'image 17 : points clés à 0, 10, 20, donc 7 images décodées puis jetées.
  REQUIRE(source.inbox.try_push(Flush{.epoch = 1}));
  REQUIRE(source.inbox.try_push(Command{cmd::Seek{.target_us = 17 * kFrameUs + 15'000, .exact = true}}));
  const DecodedFrame first = source.pop_epoch(1);
  CHECK(first.pts_us == 17 * kFrameUs);
  CHECK(luma_frame(first) == 17);
  CHECK(first.id.value > last_id);
  for (int n = 18; n < 30; ++n) {
    const DecodedFrame frame = source.pop();
    CHECK(frame.epoch == 1);
    CHECK(frame.pts_us == n * kFrameUs);
    CHECK(luma_frame(frame) == n);
  }
  CHECK(source.listener.wait_end(1));

  // Après la fin, un nouveau seek relance la lecture.
  REQUIRE(source.inbox.try_push(Flush{.epoch = 2}));
  REQUIRE(source.inbox.try_push(Command{cmd::Seek{.target_us = 3 * kFrameUs, .exact = true}}));
  const DecodedFrame again = source.pop_epoch(2);
  CHECK(again.pts_us == 3 * kFrameUs);
  CHECK(luma_frame(again) == 3);
}

TEST_CASE("SourceThread : seek rapide sur le point clé le plus proche, sans rejet", "[media][source]") {
  Running source(seek_clip(), 2);
  source.pop();
  REQUIRE(source.inbox.try_push(Flush{.epoch = 1}));
  REQUIRE(source.inbox.try_push(Command{cmd::Seek{.target_us = 18 * kFrameUs, .exact = false}}));
  const DecodedFrame frame = source.pop_epoch(1);
  CHECK(frame.pts_us == 20 * kFrameUs);
  CHECK(luma_frame(frame) == 20);
}

TEST_CASE("SourceThread : paquets audio et sous-titres livrés par piste", "[media][source]") {
  SourceInbox* inbox = nullptr;
  auto credit = [&inbox] { inbox->try_push(Credit{}); };
  auto audio = PacketQueue::create({.max_duration_us = 80'000}, credit);  // 4 paquets de 20 ms
  auto subs = PacketQueue::create({.max_duration_us = std::nullopt}, credit);
  REQUIRE(audio.has_value());
  REQUIRE(subs.has_value());
  Running source(seek_clip(), 4, 0, {{.track = 1, .sink = audio->get()}, {.track = 2, .sink = subs->get()}});
  inbox = &source.inbox;  // avant tout try_pop : aucun Credit n'a encore pu être rendu

  std::vector<Packet> audio_packets;
  std::vector<Packet> sub_packets;
  int frames = 0;
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while ((frames < 30 || audio_packets.size() < 60) && std::chrono::steady_clock::now() < deadline) {
    if (auto frame = source.sink.pop(source.inbox, 1ms)) ++frames;
    while (auto packet = (*audio)->try_pop()) audio_packets.push_back(std::move(*packet));
    while (auto packet = (*subs)->try_pop()) sub_packets.push_back(std::move(*packet));
  }
  CHECK(frames == 30);
  REQUIRE(audio_packets.size() == 60);
  for (std::size_t k = 0; k < audio_packets.size(); ++k) {
    CHECK(audio_packets[k].track == 1);
    CHECK(audio_packets[k].pts_us == static_cast<std::int64_t>(k) * 20'000);
    CHECK(audio_packets[k].epoch == 0);
    CHECK(audio_packets[k].av != nullptr);
  }
  REQUIRE(sub_packets.size() == 3);
  const auto text = [](const Packet& p) {
    return std::string_view(reinterpret_cast<const char*>(p.data().data()), p.data().size());
  };
  CHECK(text(sub_packets[0]).ends_with(",Un"));
  CHECK(text(sub_packets[1]).ends_with(",Deux"));
  CHECK(sub_packets[2].pts_us == 900'000);
  CHECK(sub_packets[2].duration_us == 200'000);
}

// Régression du blocage de la vague 1 : 16 threads d'images retiennent la première image
// jusqu'au 16e paquet vidéo, et 30 paquets audio le précèdent dans le fichier. Sans la
// réserve d'amorçage, la source attendait sur la file audio pleine (8 paquets) un Credit
// que le consommateur ne rend qu'après la première image : 10 s puis échec, sur toute
// machine puisque le nombre de threads est explicite.
TEST_CASE("SourceThread : la première image passe sans qu'on retire l'audio, même à 16 threads",
          "[media][source]") {
  SourceInbox* inbox = nullptr;
  auto credit = [&inbox] { inbox->try_push(Credit{}); };
  auto audio = PacketQueue::create({.max_duration_us = 160'000}, credit);  // 8 paquets
  auto subs = PacketQueue::create({.max_duration_us = std::nullopt}, credit);
  REQUIRE(audio.has_value());
  REQUIRE(subs.has_value());
  Running source(seek_clip(), 4, 0, {{.track = 1, .sink = audio->get()}, {.track = 2, .sink = subs->get()}},
                 16);
  inbox = &source.inbox;

  CHECK(source.pop().pts_us == 0);  // l'audio n'est pas retiré
  CHECK((*audio)->size() == 8);
  // Amorcée, l'époque reprend la contre-pression : la file audio pleine retient la
  // lecture, et donc la vidéo, jusqu'à ce que l'audio retire ses paquets.
  CHECK_FALSE(source.sink.pop(source.inbox, 200ms).has_value());

  std::vector<Packet> audio_packets;
  int frames = 1;
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while ((frames < 30 || audio_packets.size() < 60) && std::chrono::steady_clock::now() < deadline) {
    if (auto frame = source.sink.pop(source.inbox, 1ms)) ++frames;
    while (auto packet = (*audio)->try_pop()) audio_packets.push_back(std::move(*packet));
    while ((*subs)->try_pop()) {
    }
  }
  CHECK(frames == 30);
  REQUIRE(audio_packets.size() == 60);
  for (std::size_t k = 0; k < audio_packets.size(); ++k) {
    CHECK(audio_packets[k].pts_us == static_cast<std::int64_t>(k) * 20'000);  // la réserve garde l'ordre
    CHECK(audio_packets[k].epoch == 0);
  }
  CHECK(source.listener.wait_end(0));
  CHECK(source.listener.failure_count() == 0);
}

TEST_CASE("SourceThread : une réserve d'amorçage pleine est un refus PacketBacklog, pas une attente",
          "[media][source]") {
  // 64 Kio : 16 paquets PCM de 20 ms (3840 octets + kPacketOverhead). Avant la première
  // image à 16 threads, 30 paquets audio arrivent : 8 dans la file, 22 pour la réserve.
  auto audio = PacketQueue::create({.max_duration_us = 160'000}, [] {});
  REQUIRE(audio.has_value());
  Running source(seek_clip(), 4, 0, {{.track = 1, .sink = audio->get()}}, 16, kMinReserveBytes);
  const auto failure = source.listener.wait_failure();
  REQUIRE(failure.has_value());
  CHECK(failure->code == RefusalCode::PacketBacklog);
  CHECK(failure->detail.find("piste 1") != std::string::npos);
  CHECK_FALSE(source.sink.pop(source.inbox, 50ms).has_value());
}

TEST_CASE("SourceThread : un Flush qui ne fait pas croître l'époque est refusé, la lecture continue",
          "[media][source]") {
  Running source(full_clip(), 2);
  CHECK(source.pop().pts_us == 0);
  REQUIRE(source.inbox.try_push(Flush{.epoch = 0}));
  const auto failure = source.listener.wait_failure();
  REQUIRE(failure.has_value());
  CHECK(failure->code == RefusalCode::FieldOutOfRange);
  std::int64_t last = 0;
  for (int n = 1; n < 30; ++n) last = source.pop().pts_us;  // rien de perdu : l'époque n'a pas changé
  CHECK(last == 29 * kFrameUs);
}

TEST_CASE("SourceThread : un seek hors du fichier est refusé, la position est gardée", "[media][source]") {
  Running source(full_clip(), 2);
  CHECK(source.pop().pts_us == 0);
  REQUIRE(source.inbox.try_push(Command{cmd::Seek{.target_us = 60'000'000}}));
  const auto failure = source.listener.wait_failure();
  REQUIRE(failure.has_value());
  CHECK(failure->code == RefusalCode::SeekFailed);
  for (int n = 1; n < 30; ++n) CHECK(source.pop().pts_us == n * kFrameUs);
}

TEST_CASE("SourceThread : Quit arrête la boucle", "[media][source]") {
  Running source(full_clip(), 1);
  REQUIRE(source.sink.wait_refusals(1));  // une image livrée, la suivante attend
  REQUIRE(source.inbox.try_push(Command{cmd::Quit{}}));
  std::this_thread::sleep_for(50ms);
  const int attempts = source.sink.attempts();
  CHECK(source.sink.pop(source.inbox, 50ms).has_value());  // rend un Credit
  std::this_thread::sleep_for(50ms);
  CHECK(source.sink.attempts() == attempts);
}

TEST_CASE("SourceThread : des liens incomplets ou faux sont refusés", "[media][source]") {
  SourceInbox inbox(kSourceInboxBounds);
  TestSink sink(2);
  TestListener listener;
  auto queue = PacketQueue::create({}, [] {});
  REQUIRE(queue.has_value());
  const auto refusal = [&](SourceLinks links) {
    auto started = SourceThread::start(open_media(full_clip()), std::move(links), inbox);
    REQUIRE_FALSE(started);
    return started.error().code;
  };
  CHECK(refusal({.frames = nullptr, .listener = &listener}) == RefusalCode::FieldMissing);
  CHECK(refusal({.frames = &sink, .listener = nullptr}) == RefusalCode::FieldMissing);
  CHECK(refusal({.frames = &sink, .listener = &listener, .packets = {{.track = 0, .sink = queue->get()}}}) ==
        RefusalCode::TrackUnavailable);  // la vidéo n'est pas routée en paquets
  CHECK(refusal({.frames = &sink, .listener = &listener, .packets = {{.track = 5, .sink = queue->get()}}}) ==
        RefusalCode::TrackUnavailable);  // une pièce jointe n'a pas de paquets
  CHECK(refusal({.frames = &sink, .listener = &listener, .packets = {{.track = 1, .sink = nullptr}}}) ==
        RefusalCode::TrackUnavailable);
  CHECK(refusal({.frames = &sink,
                 .listener = &listener,
                 .packets = {{.track = 1, .sink = queue->get()}, {.track = 1, .sink = queue->get()}}}) ==
        RefusalCode::TrackUnavailable);
  CHECK(refusal({.frames = &sink, .listener = &listener, .reserve_bytes = kMinReserveBytes - 1}) ==
        RefusalCode::FieldOutOfRange);
  CHECK(refusal({.frames = &sink, .listener = &listener, .reserve_bytes = kMaxReserveBytes + 1}) ==
        RefusalCode::FieldOutOfRange);
  auto not_video = open_source(*Demuxer::open(require_clip(full_clip())), 1, {.attempts = {HwApi::Software}});
  REQUIRE_FALSE(not_video);
  CHECK(not_video.error().code == RefusalCode::TrackUnavailable);
}
