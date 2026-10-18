#include "audio/converter.h"

#include <array>
#include <cmath>
#include <vector>

extern "C" {
#include <libavutil/channel_layout.h>
}

#include <catch2/catch_test_macros.hpp>

#include "sine.h"

using namespace ajp::audio;
using ajp::audio::test::channel;
using ajp::audio::test::concat;
using ajp::audio::test::sine_frame;
using ajp::audio::test::tone_amplitude;

namespace {

// Canaux de la disposition dans l'ordre où ils sont rangés en mémoire.
std::vector<AVChannel> order(std::uint64_t mask) {
  AVChannelLayout layout{};
  av_channel_layout_from_mask(&layout, mask);
  std::vector<AVChannel> channels;
  for (int i = 0; i < layout.nb_channels; ++i) {
    channels.push_back(av_channel_layout_channel_from_index(&layout, static_cast<unsigned>(i)));
  }
  av_channel_layout_uninit(&layout);
  return channels;
}

}  // namespace

TEST_CASE("PcmConverter : les canaux sortent dans l'ordre documenté par SDL3", "[audio][converter]") {
  CHECK(order(sdl_channel_mask(1)) == std::vector{AV_CHAN_FRONT_CENTER});
  CHECK(order(sdl_channel_mask(2)) == std::vector{AV_CHAN_FRONT_LEFT, AV_CHAN_FRONT_RIGHT});
  CHECK(order(sdl_channel_mask(3)) ==
        std::vector{AV_CHAN_FRONT_LEFT, AV_CHAN_FRONT_RIGHT, AV_CHAN_LOW_FREQUENCY});
  CHECK(order(sdl_channel_mask(4)) ==
        std::vector{AV_CHAN_FRONT_LEFT, AV_CHAN_FRONT_RIGHT, AV_CHAN_BACK_LEFT, AV_CHAN_BACK_RIGHT});
  CHECK(order(sdl_channel_mask(5)) == std::vector{AV_CHAN_FRONT_LEFT, AV_CHAN_FRONT_RIGHT,
                                                  AV_CHAN_LOW_FREQUENCY, AV_CHAN_BACK_LEFT,
                                                  AV_CHAN_BACK_RIGHT});
  CHECK(order(sdl_channel_mask(6)) == std::vector{AV_CHAN_FRONT_LEFT, AV_CHAN_FRONT_RIGHT,
                                                  AV_CHAN_FRONT_CENTER, AV_CHAN_LOW_FREQUENCY,
                                                  AV_CHAN_SIDE_LEFT, AV_CHAN_SIDE_RIGHT});
  CHECK(order(sdl_channel_mask(7)) ==
        std::vector{AV_CHAN_FRONT_LEFT, AV_CHAN_FRONT_RIGHT, AV_CHAN_FRONT_CENTER, AV_CHAN_LOW_FREQUENCY,
                    AV_CHAN_BACK_CENTER, AV_CHAN_SIDE_LEFT, AV_CHAN_SIDE_RIGHT});
  CHECK(order(sdl_channel_mask(8)) ==
        std::vector{AV_CHAN_FRONT_LEFT, AV_CHAN_FRONT_RIGHT, AV_CHAN_FRONT_CENTER, AV_CHAN_LOW_FREQUENCY,
                    AV_CHAN_BACK_LEFT, AV_CHAN_BACK_RIGHT, AV_CHAN_SIDE_LEFT, AV_CHAN_SIDE_RIGHT});
  CHECK(sdl_channel_mask(0) == 0);
  CHECK(sdl_channel_mask(9) == 0);
}

TEST_CASE("PcmConverter : s16 entrelacé vers float32, même fréquence, valeurs exactes",
          "[audio][converter]") {
  PcmConverter converter({.sample_rate = 48'000, .channels = 2});
  const auto frame = sine_frame(AV_SAMPLE_FMT_S16, 48'000, 2, 480, 1'000.0, 0.5, 0);
  const auto block = converter.convert(*frame, 3'000'000);
  REQUIRE(block);
  CHECK(block->pts_us == 3'000'000);
  REQUIRE(block->frames() == 480);
  const auto* source = reinterpret_cast<const std::int16_t*>(frame->data[0]);
  for (std::size_t i = 0; i < block->samples.size(); ++i) {
    CHECK(std::abs(block->samples[i] - static_cast<float>(source[i]) / 32768.0F) < 1e-6F);
  }
}

TEST_CASE("PcmConverter : un changement de format en cours de flux ne perd ni trame ni temps",
          "[audio][converter]") {
  constexpr double kHz = 1'000.0;
  PcmConverter converter({.sample_rate = 48'000, .channels = 2});
  struct Piece {
    AVSampleFormat format;
    int rate;
    int channels;
  };
  // 0,1 s dans chaque format : stéréo 44,1 kHz, mono planaire 48 kHz, 5.1 s32 22,05 kHz.
  const std::array pieces{Piece{AV_SAMPLE_FMT_S16, 44'100, 2}, Piece{AV_SAMPLE_FMT_FLTP, 48'000, 1},
                          Piece{AV_SAMPLE_FMT_S32, 22'050, 6}};
  std::vector<PcmBlock> blocks;
  std::int64_t pts_us = 0;
  for (const Piece& piece : pieces) {
    const int frames = piece.rate / 10;
    for (int start = 0; start < frames; start += 1'000) {
      const int count = std::min(1'000, frames - start);
      const auto frame = sine_frame(piece.format, piece.rate, piece.channels, count, kHz, 0.5, start);
      auto block = converter.convert(*frame, pts_us + start * 1'000'000LL / piece.rate);
      REQUIRE(block);
      if (block->frames() > 0) blocks.push_back(std::move(*block));
    }
    pts_us += 100'000;
  }
  blocks.push_back(converter.drain());

  // Chaque bloc commence où le précédent finit, à une trame près.
  for (std::size_t i = 1; i < blocks.size(); ++i) {
    const std::int64_t end = blocks[i - 1].pts_us + frames_to_us(blocks[i - 1].frames(), 48'000);
    CHECK(std::abs(blocks[i].pts_us - end) <= 21);
  }
  const std::vector<float> all = concat(blocks);
  const auto total = static_cast<std::int64_t>(all.size() / 2);
  CHECK(std::abs(total - 14'400) <= 64);  // 0,3 s à 48 kHz
  // Le sinus traverse les trois formats sur les deux canaux.
  for (int c = 0; c < 2; ++c) CHECK(tone_amplitude(channel(all, 2, c), 48'000, kHz) > 0.3);
}

TEST_CASE("PcmConverter : sans horodatage, le bloc suit le précédent ; reset l'oublie",
          "[audio][converter]") {
  PcmConverter converter({.sample_rate = 48'000, .channels = 2});
  const auto frame = sine_frame(AV_SAMPLE_FMT_FLT, 48'000, 2, 480, 1'000.0, 0.5, 0);
  REQUIRE(converter.convert(*frame, 1'000'000));
  const auto next = converter.convert(*frame, std::nullopt);
  REQUIRE(next);
  CHECK(next->pts_us == 1'010'000);
  converter.reset();
  const auto after_reset = converter.convert(*frame, std::nullopt);
  REQUIRE(after_reset);
  CHECK(after_reset->pts_us == 0);
}

TEST_CASE("PcmConverter : le rééchantillonnage rend sa queue au drain", "[audio][converter]") {
  PcmConverter converter({.sample_rate = 48'000, .channels = 2});
  const auto frame = sine_frame(AV_SAMPLE_FMT_S16, 44'100, 2, 4'410, 1'000.0, 0.5, 0);
  const auto block = converter.convert(*frame, 0);
  REQUIRE(block);
  const PcmBlock tail = converter.drain();
  CHECK(tail.pts_us == block->pts_us + frames_to_us(block->frames(), 48'000));
  CHECK(std::abs(block->frames() + tail.frames() - 4'800) <= 2);
}
