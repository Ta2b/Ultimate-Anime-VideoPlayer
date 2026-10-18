#include "audio/decoder.h"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "sine.h"

using ajp::RefusalCode;
using namespace ajp::audio;
using ajp::audio::test::channel;
using ajp::audio::test::concat;
using ajp::audio::test::encode_sine;
using ajp::audio::test::peak;
using ajp::audio::test::tone_amplitude;

namespace {

std::vector<PcmBlock> decode_all(AudioDecoder& decoder, const std::vector<AudioPacket>& packets) {
  std::vector<PcmBlock> blocks;
  for (const AudioPacket& packet : packets) {
    auto decoded = decoder.decode(packet);
    REQUIRE(decoded);
    for (PcmBlock& block : *decoded) blocks.push_back(std::move(block));
  }
  auto tail = decoder.drain();
  REQUIRE(tail);
  for (PcmBlock& block : *tail) blocks.push_back(std::move(block));
  return blocks;
}

}  // namespace

TEST_CASE("AudioDecoder : PCM s16 stéréo 48 kHz décodé sans perte, horodatage gardé", "[audio][decoder]") {
  const auto clip = encode_sine({.encoder = "pcm_s16le", .seconds = 0.25, .start_us = 5'000'000});
  auto decoder = AudioDecoder::open(clip.codec, {.sample_rate = 48'000, .channels = 2});
  REQUIRE(decoder);
  const std::vector<PcmBlock> blocks = decode_all(*decoder, clip.packets);
  REQUIRE_FALSE(blocks.empty());
  CHECK(blocks.front().pts_us == 5'000'000);
  const std::vector<float> all = concat(blocks);
  CHECK(all.size() == 12'000 * 2);
  // Échantillon i : 0,5 sin(2π 1000 i / 48000), quantifié sur 16 bits.
  for (std::size_t i = 0; i < 480; ++i) {
    const double phase = 2.0 * std::numbers::pi * 1'000.0 * static_cast<double>(i) / 48'000.0;
    const auto quantized = static_cast<double>(std::lround(0.5 * std::sin(phase) * 32767.0));
    CHECK(std::abs(all[2 * i] - quantized / 32768.0) < 1e-6);
    CHECK(all[2 * i] == all[2 * i + 1]);
  }
}

TEST_CASE("AudioDecoder : FLAC mono 44,1 kHz vers stéréo 48 kHz", "[audio][decoder]") {
  const auto clip = encode_sine({.encoder = "flac", .sample_rate = 44'100, .channels = 1, .seconds = 0.5});
  REQUIRE_FALSE(clip.codec.extradata.empty());  // STREAMINFO, exigé par le décodeur
  auto decoder = AudioDecoder::open(clip.codec, {.sample_rate = 48'000, .channels = 2});
  REQUIRE(decoder);
  const std::vector<PcmBlock> blocks = decode_all(*decoder, clip.packets);
  const std::vector<float> all = concat(blocks);
  CHECK(std::abs(static_cast<std::int64_t>(all.size() / 2) - 24'000) <= 2);
  const std::vector<float> left = channel(all, 2, 0);
  const std::vector<float> right = channel(all, 2, 1);
  CHECK(left == right);
  // libswresample répartit le centre mono sur gauche et droite à -3 dB.
  CHECK(std::abs(tone_amplitude(left, 48'000, 1'000.0) - 0.5 / std::numbers::sqrt2) < 0.01);
  CHECK(tone_amplitude(left, 48'000, 3'000.0) < 0.005);  // pas de repliement notable
  // Les blocs se suivent dans le temps média.
  for (std::size_t i = 1; i < blocks.size(); ++i) {
    const std::int64_t end = blocks[i - 1].pts_us + frames_to_us(blocks[i - 1].frames(), 48'000);
    CHECK(std::abs(blocks[i].pts_us - end) <= 21);
  }
}

TEST_CASE("AudioDecoder : AAC stéréo vers 5.1, les canaux avant portent le signal", "[audio][decoder]") {
  const auto clip = encode_sine({.encoder = "aac", .sample_rate = 48'000, .channels = 2, .seconds = 0.5});
  auto decoder = AudioDecoder::open(clip.codec, {.sample_rate = 48'000, .channels = 6});
  REQUIRE(decoder);
  const std::vector<float> all = concat(decode_all(*decoder, clip.packets));
  REQUIRE(all.size() % 6 == 0);
  CHECK(tone_amplitude(channel(all, 6, 0), 48'000, 1'000.0) > 0.4);  // FL
  CHECK(tone_amplitude(channel(all, 6, 1), 48'000, 1'000.0) > 0.4);  // FR
  CHECK(peak(channel(all, 6, 2)) < 1e-6);                            // FC vide : rien n'est inventé
  CHECK(peak(channel(all, 6, 3)) < 1e-6);                            // LFE vide
}

TEST_CASE("AudioDecoder : codec inconnu ou non audio refusé", "[audio][decoder]") {
  const auto unknown = AudioDecoder::open({.codec = "inexistant"}, {});
  REQUIRE_FALSE(unknown);
  CHECK(unknown.error().code == RefusalCode::AudioCodecUnavailable);
  const auto video = AudioDecoder::open({.codec = "h264"}, {});
  REQUIRE_FALSE(video);
  CHECK(video.error().code == RefusalCode::AudioCodecUnavailable);
}

TEST_CASE("AudioDecoder : format de sortie hors bornes refusé", "[audio][decoder]") {
  const auto refused = AudioDecoder::open({.codec = "pcm_s16le", .sample_rate = 48'000, .channels = 2},
                                          {.sample_rate = 48'000, .channels = 12});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("AudioDecoder : un paquet corrompu est refusé, le décodeur continue", "[audio][decoder]") {
  const auto clip = encode_sine({.encoder = "flac", .sample_rate = 48'000, .channels = 2, .seconds = 0.5});
  auto decoder = AudioDecoder::open(clip.codec, {.sample_rate = 48'000, .channels = 2});
  REQUIRE(decoder);
  AudioPacket garbage{.data = std::vector<std::uint8_t>(512, 0x5A), .pts_us = 0};
  const auto refused = decoder->decode(garbage);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::AudioDecodeFailed);
  const auto good = decoder->decode(clip.packets.at(1));
  REQUIRE(good);
  CHECK_FALSE(good->empty());
}

TEST_CASE("AudioDecoder : flush oublie le flux, le décodage reprend au nouvel instant", "[audio][decoder]") {
  const auto clip = encode_sine({.encoder = "pcm_s16le", .seconds = 0.2});
  const auto later = encode_sine({.encoder = "pcm_s16le", .seconds = 0.2, .start_us = 60'000'000});
  auto decoder = AudioDecoder::open(clip.codec, {.sample_rate = 44'100, .channels = 2});
  REQUIRE(decoder);
  REQUIRE(decoder->decode(clip.packets.at(0)));
  decoder->flush();
  const auto resumed = decoder->decode(later.packets.at(0));
  REQUIRE(resumed);
  REQUIRE_FALSE(resumed->empty());
  CHECK(resumed->front().pts_us == 60'000'000);
}
