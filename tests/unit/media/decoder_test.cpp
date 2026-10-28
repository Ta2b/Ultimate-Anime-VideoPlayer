#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "media/video_decoder.h"

extern "C" {
#include <libavcodec/packet.h>
#include <libavutil/cpu.h>
#include <libavutil/mathematics.h>
}

#include "clips.h"
#include "media/av_util.h"
#include "media/demuxer.h"

using namespace ajp::media;
using ajp::ChromaSiting;
using ajp::ColorMatrix;
using ajp::ColorRange;
using ajp::DecodedFrame;
using ajp::FrameMemory;
using ajp::PixelFormat;
using ajp::Primaries;
using ajp::RefusalCode;
using ajp::Transfer;

namespace {

DecoderOptions software() {
  return DecoderOptions{.attempts = {HwApi::Software}};
}

// Décode la piste 0 jusqu'à la fin du flux.
std::vector<DecodedFrame> drain(Demuxer& demuxer, VideoDecoder& decoder) {
  std::vector<DecodedFrame> frames;
  while (!decoder.drained()) {
    auto frame = decoder.receive();
    REQUIRE(frame.has_value());
    if (*frame) {
      frames.push_back(std::move(**frame));
      continue;
    }
    if (decoder.drained()) break;
    auto packet = demuxer.read();
    REQUIRE(packet.has_value());
    if (!*packet) {
      REQUIRE(decoder.send_end());
    } else if ((*packet)->track == 0) {
      REQUIRE(decoder.send(**packet));
    }
  }
  return frames;
}

// Décode toute la piste 0 du clip.
std::vector<DecodedFrame> decode_all(const test::ClipSpec& spec, const DecoderOptions& options = software()) {
  auto demuxer = Demuxer::open(test::require_clip(spec));
  REQUIRE(demuxer.has_value());
  auto decoder = VideoDecoder::open(demuxer->tracks()[0], options);
  INFO((decoder ? std::string() : decoder.error().detail));
  REQUIRE(decoder.has_value());
  return drain(*demuxer, *decoder);
}

// Même taille et mêmes horodatages que `model`, octets sans sens.
Packet garbage(const Packet& model) {
  AVPacket* raw = av_packet_alloc();
  REQUIRE(raw != nullptr);
  REQUIRE(av_new_packet(raw, model.av->size) == 0);
  for (int i = 0; i < raw->size; ++i) raw->data[i] = static_cast<std::uint8_t>((i * 37 + 11) & 0xFF);
  REQUIRE(av_packet_copy_props(raw, model.av.get()) == 0);
  Packet damaged = model;
  damaged.av = std::shared_ptr<AVPacket>(raw, PacketFree{});
  return damaged;
}

int sample(const DecodedFrame& frame, int plane, int x, int y, int bits) {
  const auto bytes = frame.data->host_plane(plane);
  const std::size_t offset = static_cast<std::size_t>(y) * frame.data->host_stride(plane) +
                             static_cast<std::size_t>(x) * (bits > 8 ? 2U : 1U);
  if (bits <= 8) return static_cast<int>(bytes[offset]);
  return static_cast<int>(bytes[offset]) | (static_cast<int>(bytes[offset + 1]) << 8);
}

}  // namespace

TEST_CASE("VideoDecoder : pixels FFV1 exacts, pts et durées en microsecondes", "[media][decoder]") {
  const std::vector<DecodedFrame> frames = decode_all(test::full_clip());
  REQUIRE(frames.size() == 30);
  for (std::size_t n = 0; n < frames.size(); ++n) {
    const DecodedFrame& frame = frames[n];
    CHECK(frame.pts_us == static_cast<std::int64_t>(n) * test::kFrameUs);
    CHECK(frame.duration_us == test::kFrameUs);
    CHECK(frame.width == 64);
    CHECK(frame.height == 48);
    CHECK(frame.format == PixelFormat::Yuv420p);
    REQUIRE(frame.data);
    CHECK(frame.data->memory() == FrameMemory::Host);
    CHECK(frame.data->plane_count() == 3);
    CHECK(frame.data->ready_event() == 0);
    CHECK(frame.data->ready_value() == 0);    // ni CUevent ni sémaphore timeline
    CHECK(frame.data->native_pitch(0) == 0);  // mémoire hôte : le pas est host_stride
    CHECK(frame.data->native_plane(0) == 0);
    CHECK(frame.data->host_stride(0) >= 64);
    CHECK(frame.data->host_plane(1).size() >= frame.data->host_stride(1) * 23 + 32);
    CHECK(frame.data->host_plane(3).empty());
    const int i = static_cast<int>(n);
    for (const auto& [x, y] : {std::pair{0, 0}, std::pair{63, 47}, std::pair{17, 29}}) {
      CHECK(sample(frame, 0, x, y, 8) == test::pattern(0, x, y, i, 8));
      CHECK(sample(frame, 1, x / 2, y / 2, 8) == test::pattern(1, x / 2, y / 2, i, 8));
      CHECK(sample(frame, 2, x / 2, y / 2, 8) == test::pattern(2, x / 2, y / 2, i, 8));
    }
  }
}

TEST_CASE("VideoDecoder : une source non étiquetée reçoit la couleur devinée comme mpv", "[media][decoder]") {
  const std::vector<DecodedFrame> frames = decode_all(test::full_clip());
  REQUIRE_FALSE(frames.empty());
  const ajp::ColorInfo& color = frames[0].color;
  CHECK(color.matrix == ColorMatrix::Bt601);   // 64x48 : définition standard
  CHECK(color.primaries == Primaries::Bt709);  // ni 576 ni 480 lignes
  CHECK(color.transfer == Transfer::Bt1886);
  CHECK(color.range == ColorRange::Limited);
  CHECK(color.siting == ChromaSiting::Left);
  CHECK_FALSE(color.hdr.has_value());
}

TEST_CASE("VideoDecoder : ColorInfo HDR10 depuis les side data, pixels 10 bits", "[media][decoder]") {
  const std::vector<DecodedFrame> frames = decode_all(test::hdr_clip());
  REQUIRE(frames.size() == 5);
  const DecodedFrame& frame = frames[2];
  CHECK(frame.format == PixelFormat::Yuv420p10);
  CHECK(sample(frame, 0, 5, 7, 10) == test::pattern(0, 5, 7, 2, 10));
  CHECK(sample(frame, 2, 3, 1, 10) == test::pattern(2, 3, 1, 2, 10));

  const ajp::ColorInfo& color = frame.color;
  CHECK(color.matrix == ColorMatrix::Bt2020Ncl);
  CHECK(color.primaries == Primaries::Bt2020);
  CHECK(color.transfer == Transfer::Pq);
  CHECK(color.range == ColorRange::Limited);
  CHECK(color.siting == ChromaSiting::TopLeft);
  REQUIRE(color.hdr.has_value());
  const ajp::HdrMetadata& hdr = *color.hdr;
  CHECK(hdr.primaries_xy[0] == Catch::Approx(0.708).margin(1e-4));
  CHECK(hdr.primaries_xy[1] == Catch::Approx(0.292).margin(1e-4));
  CHECK(hdr.primaries_xy[2] == Catch::Approx(0.170).margin(1e-4));
  CHECK(hdr.primaries_xy[5] == Catch::Approx(0.046).margin(1e-4));
  CHECK(hdr.white_xy[0] == Catch::Approx(0.3127).margin(1e-4));
  CHECK(hdr.white_xy[1] == Catch::Approx(0.3290).margin(1e-4));
  CHECK(hdr.max_luminance == Catch::Approx(1000.0));
  CHECK(hdr.min_luminance == Catch::Approx(0.0001).margin(1e-6));
  CHECK(hdr.max_cll == Catch::Approx(1000.0));
  CHECK(hdr.max_fall == Catch::Approx(400.0));
}

TEST_CASE("VideoDecoder : images B remises dans l'ordre, pts exacts à 24000/1001", "[media][decoder]") {
  const std::vector<DecodedFrame> frames = decode_all(test::bframes_clip());
  REQUIRE(frames.size() == 24);
  for (std::size_t n = 0; n < frames.size(); ++n) {
    const std::int64_t expected = av_rescale(static_cast<std::int64_t>(n) * 1001, 1'000'000, 24000);
    CHECK(frames[n].pts_us == expected);
    CHECK(test::frame_of_flat_luma(sample(frames[n], 0, 30, 20, 8)) == static_cast<int>(n));
  }
  CHECK(frames[1].duration_us == Catch::Approx(41708).margin(1));
}

TEST_CASE("VideoDecoder : des données abîmées perdent des images, le décodage continue", "[media][decoder]") {
  // Avec des threads d'images, l'erreur sort à la réception, en fin de flux
  // surtout ; sans thread, à l'envoi. Les deux doivent finir le flux.
  const int threads = GENERATE(1, 0);
  const bool at_end = GENERATE(false, true);
  INFO("threads " << threads << (at_end ? ", fin abîmée" : ", milieu abîmé"));
  auto demuxer = Demuxer::open(test::require_clip(test::seek_clip()));
  REQUIRE(demuxer.has_value());
  auto decoder =
      VideoDecoder::open(demuxer->tracks()[0], {.attempts = {HwApi::Software}, .threads = threads});
  REQUIRE(decoder.has_value());
  int video = 0;
  int frames = 0;
  while (!decoder->drained()) {
    auto frame = decoder->receive();
    REQUIRE(frame.has_value());
    if (*frame) {
      ++frames;
      continue;
    }
    if (decoder->drained()) break;
    auto packet = demuxer->read();
    REQUIRE(packet.has_value());
    if (!*packet) {
      REQUIRE(decoder->send_end());
      continue;
    }
    if ((*packet)->track != 0) continue;
    ++video;
    const bool damaged = at_end ? video >= 29 : (video == 5 || video == 6);
    REQUIRE(decoder->send(damaged ? garbage(**packet) : **packet));
  }
  CHECK(decoder->data_errors() >= 1);
  CHECK(frames >= 24);
  CHECK(frames < 30);
}

TEST_CASE("VideoDecoder : options hors plage refusées", "[media][decoder]") {
  auto demuxer = Demuxer::open(test::require_clip(test::full_clip()));
  REQUIRE(demuxer.has_value());
  const TrackInfo& video = demuxer->tracks()[0];
  const auto code = [&](const DecoderOptions& options) {
    auto decoder = VideoDecoder::open(video, options);
    return decoder ? std::optional<RefusalCode>() : std::optional(decoder.error().code);
  };
  CHECK(code(DecoderOptions{}) == RefusalCode::FieldOutOfRange);
  CHECK(code({.attempts = {HwApi::Software, HwApi::Software}}) == RefusalCode::FieldOutOfRange);
  CHECK(code({.attempts = {HwApi::Software}, .extra_hw_frames = 33}) == RefusalCode::FieldOutOfRange);
  CHECK(code({.attempts = {HwApi::Software}, .extra_hw_frames = -1}) == RefusalCode::FieldOutOfRange);
  CHECK(code({.attempts = {HwApi::Software}, .direct_surfaces = 33}) == RefusalCode::FieldOutOfRange);
  CHECK(code({.attempts = {HwApi::Software}, .direct_surfaces = -1}) == RefusalCode::FieldOutOfRange);
  CHECK(code({.attempts = {HwApi::Software}, .threads = 65}) == RefusalCode::FieldOutOfRange);
  CHECK(code(software()) == std::nullopt);
  auto audio = VideoDecoder::open(demuxer->tracks()[1], software());
  REQUIRE_FALSE(audio);
  CHECK(audio.error().code == RefusalCode::TrackUnavailable);
}

TEST_CASE("VideoDecoder : chaque essai matériel écarté est nommé, le logiciel prend le relais",
          "[media][decoder]") {
  auto demuxer = Demuxer::open(test::require_clip(test::full_clip()));
  REQUIRE(demuxer.has_value());
  const TrackInfo& video = demuxer->tracks()[0];
  // Ni NVDEC ni VA-API n'ont de décodeur FFV1 : deux refus avant tout appareil. FFV1
  // n'est pas un codec Vulkan Video : sans vulkan_compute, Vulkan le refuse aussi avant
  // tout appareil, que FFmpeg ait ou non ffv1_vulkan (il dépend du glslc de configure).
  const DecoderOptions options{.attempts = {HwApi::Cuda, HwApi::Vaapi, HwApi::Vulkan, HwApi::Software}};
  auto decoder = VideoDecoder::open(video, options);
  REQUIRE(decoder.has_value());
  CHECK(decoder->api() == HwApi::Software);
  REQUIRE(decoder->rejected().size() == 3);
  CHECK(decoder->rejected()[0].api == HwApi::Cuda);
  CHECK(decoder->rejected()[1].api == HwApi::Vaapi);
  CHECK(decoder->rejected()[2].api == HwApi::Vulkan);
  for (const Attempt& attempt : decoder->rejected()) {
    CHECK(attempt.why.code == RefusalCode::HwDecodeUnavailable);
    CHECK(attempt.why.detail.starts_with(name(attempt.api)));
  }
  CHECK(decoder->rejected()[2].why.detail.find("hors Vulkan Video") != std::string::npos);
}

TEST_CASE("VideoDecoder : FFV1 sous Vulkan sur option, décodé par le GPU ou repli logiciel nommé",
          "[media][decoder][vulkan]") {
  // ffv1_vulkan (compute shaders) s'initialise au premier point clé, dans get_format, pas à
  // l'ouverture : le chemin réel ne se lit qu'une fois le flux décodé, sur le même décodeur.
  auto demuxer = Demuxer::open(test::require_clip(test::full_clip()));
  REQUIRE(demuxer.has_value());
  auto decoder = VideoDecoder::open(demuxer->tracks()[0],
                                    {.attempts = {HwApi::Vulkan, HwApi::Software}, .vulkan_compute = true});
  REQUIRE(decoder.has_value());
  const std::vector<DecodedFrame> frames = drain(*demuxer, *decoder);
  REQUIRE(frames.size() == 30);
  const bool gpu = decoder->api() == HwApi::Vulkan;
  if (gpu) {
    CHECK(decoder->rejected().empty());
    CHECK(decoder->threads() == 1);
  } else {
    REQUIRE(decoder->rejected().size() == 1);
    CHECK(decoder->rejected()[0].api == HwApi::Vulkan);
    CHECK(decoder->rejected()[0].why.detail.starts_with("vulkan"));
    // Repli avant la première image : rouvert en logiciel sur ses threads, paquets rejoués.
    CHECK(decoder->threads() == std::clamp(av_cpu_count(), 1, kAutoDecoderThreads));
  }
  for (const DecodedFrame& frame : frames) {
    CHECK(frame.data->memory() == (gpu ? FrameMemory::Vulkan : FrameMemory::Host));
  }
}

TEST_CASE("VideoDecoder : sans repli logiciel, un refus nommé liste les essais", "[media][decoder]") {
  auto demuxer = Demuxer::open(test::require_clip(test::full_clip()));
  REQUIRE(demuxer.has_value());
  auto decoder = VideoDecoder::open(demuxer->tracks()[0], {.attempts = {HwApi::Cuda, HwApi::VideoToolbox}});
  REQUIRE_FALSE(decoder);
  CHECK(decoder.error().code == RefusalCode::DecoderUnavailable);
  CHECK(decoder.error().detail.find("cuda") != std::string::npos);
  CHECK(decoder.error().detail.find("videotoolbox") != std::string::npos);
}

TEST_CASE("VideoDecoder : l'ordre d'essai par défaut finit par le logiciel", "[media][decoder]") {
  const std::vector<HwApi> attempts = default_attempts();
  REQUIRE_FALSE(attempts.empty());
  CHECK(attempts.back() == HwApi::Software);
  CHECK(std::ranges::count(attempts, HwApi::Software) == 1);
}
