#include "media/source.h"

#include <filesystem>
#include <fstream>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavutil/error.h>
}

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
SourceMedia open_media(const std::filesystem::path& path, int threads = 0) {
  auto demuxer = Demuxer::open(path);
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
      : Running(require_clip(spec), capacity, refuse_first, std::move(routes), threads, reserve_bytes) {}

  Running(const std::filesystem::path& path, std::size_t capacity, int refuse_first = 0,
          std::vector<PacketRoute> routes = {}, int threads = 0,
          std::size_t reserve_bytes = kDefaultReserveBytes, ReadFault fault = {})
      : sink(capacity, refuse_first) {
    SourceMedia media = open_media(path, threads);
    media.demuxer.fail_reads(fault);
    auto started = SourceThread::start(std::move(media),
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

// Clip de 100 images (4 s), un point clé toutes les 10, assez long pour qu'un dégât au milieu
// laisse des images lisibles de part et d'autre.
ClipSpec long_clip() {
  ClipSpec spec = seek_clip();
  spec.name = "long.mkv";
  spec.frames = 100;
  spec.subtitles.clear();
  return spec;
}

// Copie du clip, tronquée à `keep` de sa taille, ou abîmée par des octets 0xA5 entre `from` et `to`.
std::filesystem::path damaged_copy(const std::string& name, double keep, double from = 0.0, double to = 0.0) {
  const std::filesystem::path clean = require_clip(long_clip());
  const std::filesystem::path damaged = clean.parent_path() / name;
  std::filesystem::copy_file(clean, damaged, std::filesystem::copy_options::overwrite_existing);
  const auto size = std::filesystem::file_size(damaged);
  if (to > from) {
    std::fstream file(damaged, std::ios::in | std::ios::out | std::ios::binary);
    const auto begin = static_cast<std::streamoff>(static_cast<double>(size) * from);
    const auto end = static_cast<std::streamoff>(static_cast<double>(size) * to);
    file.seekp(begin);
    const std::string garbage(static_cast<std::size_t>(end - begin), '\xA5');
    file.write(garbage.data(), static_cast<std::streamsize>(garbage.size()));
  }
  std::filesystem::resize_file(damaged, static_cast<std::uintmax_t>(static_cast<double>(size) * keep));
  return damaged;
}

// Toutes les images livrées jusqu'à la fin du flux.
std::vector<std::int64_t> play_to_end(Running& source) {
  std::vector<std::int64_t> pts;
  while (auto frame = source.sink.pop(source.inbox, 2s)) pts.push_back(frame->pts_us);
  return pts;
}

// Les `count` premières images, chacune attendue jusqu'à kTimeout : une pause de la source (nouvel essai
// après une erreur) sur une machine chargée ne coupe pas la lecture comme le délai court de play_to_end.
std::vector<std::int64_t> play_count(Running& source, std::size_t count) {
  std::vector<std::int64_t> pts;
  while (pts.size() < count) {
    auto frame = source.sink.pop(source.inbox);
    if (!frame) break;
    pts.push_back(frame->pts_us);
  }
  return pts;
}

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
  auto credit = [&inbox] { std::ignore = inbox->try_push(Credit{}); };
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
  auto credit = [&inbox] { std::ignore = inbox->try_push(Credit{}); };
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
    auto started = SourceThread::start(open_media(require_clip(full_clip())), std::move(links), inbox);
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

TEST_CASE("SourceThread : un MKV tronqué à 50 % se lit jusqu'au bout lisible, puis fin de flux",
          "[media][source]") {
  Running source(damaged_copy("truncated.mkv", 0.5), 4);
  const std::vector<std::int64_t> pts = play_to_end(source);
  CHECK(source.listener.wait_end(0));
  REQUIRE(pts.size() >= 40);  // environ la moitié des 100 images
  CHECK(pts.size() < 100);
  for (std::size_t n = 0; n < pts.size(); ++n) CHECK(pts[n] == static_cast<std::int64_t>(n) * kFrameUs);
  for (const auto& failure : source.listener.failures()) CHECK(failure.recoverable);
}

TEST_CASE("SourceThread : un bloc abîmé au milieu n'arrête pas la lecture", "[media][source]") {
  Running source(damaged_copy("corrupt.mkv", 1.0, 0.45, 0.50), 4);
  const std::vector<std::int64_t> pts = play_to_end(source);
  CHECK(source.listener.wait_end(0));
  REQUIRE_FALSE(pts.empty());
  CHECK(pts.front() == 0);
  CHECK(pts.back() >= 90 * kFrameUs);  // les images d'après le dégât arrivent
  for (const auto& failure : source.listener.failures()) CHECK(failure.recoverable);
}

TEST_CASE("SourceThread : une erreur de lecture passagère est retentée, rien n'est perdu",
          "[media][source]") {
  // Trois échecs d'entrée-sortie au milieu du fichier, comme un partage réseau qui hésite.
  Running source(require_clip(long_clip()), 4, 0, {}, 0, kDefaultReserveBytes,
                 ReadFault{.after_packets = 60, .count = 3, .error = AVERROR(EIO)});
  const std::vector<std::int64_t> pts = play_count(source, 100);
  CHECK(source.listener.wait_end(0));
  REQUIRE(pts.size() == 100);
  CHECK(pts.back() == 99 * kFrameUs);
  CHECK(source.listener.failure_count() == 0);
}

TEST_CASE("SourceThread : une erreur de lecture qui dure finit le flux proprement, sans refus fatal",
          "[media][source]") {
  Running source(require_clip(long_clip()), 4, 0, {}, 0, kDefaultReserveBytes,
                 ReadFault{.after_packets = 60, .count = 1000, .error = AVERROR(EIO)});
  const std::vector<std::int64_t> pts = play_to_end(source);
  CHECK(source.listener.wait_end(0));
  // Les images déjà lues sortent du décodeur avant la fin du flux.
  REQUIRE(pts.size() >= 25);
  CHECK(pts.size() < 100);
  for (std::size_t n = 0; n < pts.size(); ++n) CHECK(pts[n] == static_cast<std::int64_t>(n) * kFrameUs);
  const auto failures = source.listener.failures();
  REQUIRE(failures.size() == 1);
  CHECK(failures.front().code == RefusalCode::DecodeFailed);
  CHECK(failures.front().recoverable);
}

TEST_CASE("SourceThread : après une fin sur erreur, un seek relit le fichier", "[media][source]") {
  Running source(require_clip(long_clip()), 4, 0, {}, 0, kDefaultReserveBytes,
                 ReadFault{.after_packets = 60, .count = 70, .error = AVERROR_INVALIDDATA});
  play_to_end(source);
  CHECK(source.listener.wait_end(0));
  REQUIRE(source.inbox.try_push(Flush{.epoch = 1}));
  REQUIRE(source.inbox.try_push(Command{cmd::Seek{.target_us = 0, .exact = true}}));
  const std::vector<std::int64_t> pts = play_to_end(source);
  CHECK(source.listener.wait_end(1));
  REQUIRE(pts.size() == 100);  // les essais ratés sont épuisés : tout le fichier se lit
}

TEST_CASE("SourceThread : une source aux dimensions impaires arrive paire, sans refus", "[media][source]") {
  ClipSpec spec = full_clip();
  spec.name = "odd.mkv";
  spec.width = 65;  // comme 853×480 : la chaîne refuse une largeur ou une hauteur impaire
  spec.height = 49;
  const std::filesystem::path path = require_clip(spec);
  auto demuxer = Demuxer::open(path);
  REQUIRE(demuxer.has_value());
  CHECK(demuxer->tracks().front().width == 64);
  CHECK(demuxer->tracks().front().height == 48);
  Running source(path, 4);
  const DecodedFrame first = source.pop();
  CHECK(first.width == 64);
  CHECK(first.height == 48);
  // Les pixels gardés sont ceux de la source, sans décalage : le coin haut-gauche du motif.
  const auto luma = first.data->host_plane(0);
  CHECK(static_cast<int>(luma[0]) == pattern(0, 0, 0, 0, 8));
  CHECK(static_cast<int>(luma[63]) == pattern(0, 63, 0, 0, 8));
  const std::vector<std::int64_t> rest = play_to_end(source);
  CHECK(rest.size() == 29);
  CHECK(source.listener.failure_count() == 0);
}
