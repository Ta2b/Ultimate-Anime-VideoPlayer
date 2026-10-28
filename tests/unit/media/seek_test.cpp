// Seek exact du thread source sur des cas limites : images B dans un MP4 (index
// en dts), cible sur une frontière d'image, cible à la fin du fichier ; Flush
// qui suspend la livraison jusqu'au Seek ; époque des paquets après Flush.
#include <chrono>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavutil/mathematics.h>
}

#include "clips.h"
#include "media/source.h"
#include "source_harness.h"

using namespace ajp::media;
using namespace ajp::media::test;
using ajp::Command;
using ajp::DecodedFrame;
using ajp::Flush;
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

struct Running {
  SourceInbox inbox{kSourceInboxBounds};
  TestSink sink;
  TestListener listener;
  std::unique_ptr<SourceThread> thread;

  Running(const ClipSpec& spec, std::size_t capacity, std::vector<PacketRoute> routes = {}, int threads = 0)
      : sink(capacity) {
    auto started =
        SourceThread::start(open_media(spec, threads),
                            {.frames = &sink, .listener = &listener, .packets = std::move(routes)}, inbox);
    REQUIRE(started.has_value());
    thread = std::move(*started);
  }

  // Envoie Flush puis Seek, et rend la première image de la nouvelle époque.
  DecodedFrame seek(std::uint32_t epoch, std::int64_t target_us, bool exact = true) {
    REQUIRE(inbox.try_push(Flush{.epoch = epoch}));
    REQUIRE(inbox.try_push(Command{cmd::Seek{.target_us = target_us, .exact = exact}}));
    while (true) {
      auto frame = sink.pop(inbox);
      REQUIRE(frame.has_value());
      if (frame->epoch == epoch) return std::move(*frame);
      REQUIRE(frame->epoch < epoch);
    }
  }
};

int luma_frame(const DecodedFrame& frame) {
  const auto plane = frame.data->host_plane(0);
  return frame_of_flat_luma(static_cast<int>(plane[20 * frame.data->host_stride(0) + 30]));
}

std::int64_t bframes_pts(int n) {
  return av_rescale(static_cast<std::int64_t>(n) * 1001, 1'000'000, 24000);
}

}  // namespace

TEST_CASE("seek exact : images B d'un MP4, chaque image visée est la première livrée", "[media][seek]") {
  Running source(bframes_clip(), 2);
  std::uint32_t epoch = 0;
  for (int n = 0; n < 24; ++n) {
    // Sur la frontière de l'image, puis au milieu de sa durée.
    for (const std::int64_t target : {bframes_pts(n), (bframes_pts(n) + bframes_pts(n + 1)) / 2}) {
      INFO("image " << n << ", cible " << target << " µs");
      const DecodedFrame frame = source.seek(++epoch, target);
      CHECK(frame.pts_us == bframes_pts(n));
      CHECK(luma_frame(frame) == n);
    }
  }
}

TEST_CASE("seek exact : la cible 0 livre la première image", "[media][seek]") {
  Running source(seek_clip(), 2);
  const DecodedFrame frame = source.seek(1, 0);
  CHECK(frame.pts_us == 0);
  CHECK(luma_frame(frame) == 0);
}

TEST_CASE("seek exact : une cible à la fin du fichier livre la dernière image", "[media][seek]") {
  Running source(seek_clip(), 2);
  const DecodedFrame frame = source.seek(1, 30 * kFrameUs);  // durée du fichier
  CHECK(frame.pts_us == 29 * kFrameUs);
  CHECK(luma_frame(frame) == 29);
  CHECK(source.listener.wait_end(1));
  CHECK_FALSE(source.sink.pop(source.inbox, 50ms).has_value());
  CHECK(source.listener.failure_count() == 0);
}

TEST_CASE("seek exact : une cible avant la première image vidéo livre celle-ci", "[media][seek]") {
  Running source(late_video_clip(), 2);  // vidéo à partir de l'image 5, audio dès 0
  std::uint32_t epoch = 0;
  for (const std::int64_t target : {std::int64_t{0}, 3 * kFrameUs, 5 * kFrameUs}) {
    INFO("cible " << target << " µs");
    const DecodedFrame frame = source.seek(++epoch, target);
    CHECK(frame.pts_us == 5 * kFrameUs);
    CHECK(luma_frame(frame) == 0);
  }
  CHECK(source.listener.failure_count() == 0);
}

TEST_CASE("Flush : rien ne part avant le Seek qui suit, pas même avec de la place", "[media][seek]") {
  Running source(seek_clip(), 4);
  REQUIRE(source.sink.pop(source.inbox).has_value());
  REQUIRE(source.inbox.try_push(Flush{.epoch = 1}));
  // Les images de l'époque 0 déjà livrées sont rendues : leurs Credit réveillent
  // la source, qui ne doit rien livrer de l'ancienne position sous l'époque 1.
  while (auto old = source.sink.pop(source.inbox, 100ms)) CHECK(old->epoch == 0);
  CHECK(source.sink.queued() == 0);

  REQUIRE(source.inbox.try_push(Command{cmd::Seek{.target_us = 7 * kFrameUs}}));
  auto frame = source.sink.pop(source.inbox);
  REQUIRE(frame.has_value());
  CHECK(frame->epoch == 1);
  CHECK(luma_frame(*frame) == 7);
}

TEST_CASE("Flush : un Seek refusé ensuite est signalé et la lecture reprend", "[media][seek]") {
  Running source(seek_clip(), 2);
  REQUIRE(source.sink.pop(source.inbox).has_value());
  REQUIRE(source.inbox.try_push(Flush{.epoch = 1}));
  REQUIRE(source.inbox.try_push(Command{cmd::Seek{.target_us = -1}}));
  const auto failure = source.listener.wait_failure();
  REQUIRE(failure.has_value());
  CHECK(failure->code == ajp::RefusalCode::SeekFailed);
  while (true) {  // la lecture reprend sous l'époque 1 jusqu'à la fin
    auto frame = source.sink.pop(source.inbox);
    REQUIRE(frame.has_value());
    if (frame->epoch == 1 && luma_frame(*frame) == 29) break;
  }
  CHECK(source.listener.wait_end(1));
}

TEST_CASE("seek exact : paquets audio et sous-titres de la nouvelle époque, depuis le point clé",
          "[media][seek]") {
  SourceInbox* inbox = nullptr;
  auto credit = [&inbox] { inbox->try_push(ajp::Credit{}); };
  auto audio = PacketQueue::create({.max_duration_us = 160'000}, credit);  // 8 paquets de 20 ms
  auto subs = PacketQueue::create({.max_duration_us = std::nullopt}, credit);
  REQUIRE(audio.has_value());
  REQUIRE(subs.has_value());
  // 16 threads d'images : la première image attend le 16e paquet vidéo, précédé de 30
  // paquets audio. La file audio en prend 8, la réserve d'amorçage le reste, et la
  // source s'arrête dessus une fois l'image livrée.
  Running source(seek_clip(), 2, {{.track = 1, .sink = audio->get()}, {.track = 2, .sink = subs->get()}}, 16);
  inbox = &source.inbox;

  // Laisse la lecture remplir les files, puis seek vers l'image 25 (point clé 20). Le
  // Flush vide la réserve : rien de l'ancienne position ne repart sous l'époque 1.
  REQUIRE(source.sink.pop(source.inbox).has_value());
  REQUIRE((*audio)->size() == 8);
  REQUIRE(source.inbox.try_push(Flush{.epoch = 1}));
  REQUIRE(source.inbox.try_push(Command{cmd::Seek{.target_us = 25 * kFrameUs}}));
  (*audio)->discard_before(1);  // ce que fait le module audio à son Flush
  (*subs)->discard_before(1);

  // Comme audio et subs : un paquet d'une époque passée, encore en route, est jeté.
  std::vector<Packet> audio_packets;
  std::vector<Packet> sub_packets;
  const auto take = [](PacketQueue& queue, std::vector<Packet>& out) {
    while (auto packet = queue.try_pop()) {
      if (packet->epoch >= 1) out.push_back(std::move(*packet));
    }
  };
  bool last_frame = false;
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (!(last_frame && !audio_packets.empty() && audio_packets.back().pts_us == 59 * 20'000) &&
         std::chrono::steady_clock::now() < deadline) {
    if (auto frame = source.sink.pop(source.inbox, 1ms)) {
      if (frame->epoch == 1 && luma_frame(*frame) == 29) last_frame = true;
    }
    take(**audio, audio_packets);
    take(**subs, sub_packets);
  }
  CHECK(source.listener.wait_end(1));
  take(**subs, sub_packets);

  REQUIRE(last_frame);
  REQUIRE_FALSE(audio_packets.empty());
  for (const Packet& packet : audio_packets) CHECK(packet.epoch == 1);
  // Rien d'avant le point clé de l'image 20, puis tout jusqu'à la fin, sans trou.
  CHECK(audio_packets.front().pts_us >= 18 * kFrameUs);
  CHECK(audio_packets.front().pts_us <= 20 * kFrameUs);
  for (std::size_t k = 1; k < audio_packets.size(); ++k) {
    CHECK(audio_packets[k].pts_us == audio_packets[k - 1].pts_us + 20'000);
  }
  REQUIRE(sub_packets.size() == 1);  // « Trois » à 900 ms ; « Deux » (500-800 ms) est fini
  CHECK(sub_packets[0].epoch == 1);
  CHECK(sub_packets[0].pts_us == 900'000);
}
