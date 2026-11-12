#include "media/telecine.h"

#include <array>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "clips.h"
#include "media/demuxer.h"

using ajp::media::TelecineMeter;

namespace {

constexpr int kWidth = 96;
constexpr int kHeight = 64;

// Le film au temps t : des motifs qui avancent de 5 pixels par pas, avec du relief sur chaque ligne.
int film(int x, int y, int t) {
  return 20 + ((x + 5 * t) * 11 + y * 3) % 200;
}

struct Image {
  std::vector<std::uint8_t> bytes;
  int sample_bytes;
  int bits;
};

// Un champ de chaque parité pris au temps `top_time` et `bottom_time`.
Image interlaced(int top_time, int bottom_time, int bits) {
  Image image{.bytes = {}, .sample_bytes = bits > 8 ? 2 : 1, .bits = bits};
  image.bytes.resize(static_cast<std::size_t>(kWidth * kHeight * image.sample_bytes));
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const int value = film(x, y, y % 2 == 0 ? top_time : bottom_time) << (bits - 8);
      const auto at = static_cast<std::size_t>((y * kWidth + x) * image.sample_bytes);
      image.bytes[at] = static_cast<std::uint8_t>(value & 0xFF);
      if (image.sample_bytes == 2) image.bytes[at + 1] = static_cast<std::uint8_t>(value >> 8);
    }
  }
  return image;
}

void push(TelecineMeter& meter, const Image& image) {
  meter.push(image.bytes.data(), static_cast<std::ptrdiff_t>(kWidth * image.sample_bytes), kWidth, kHeight,
             image.sample_bytes, image.bits);
}

// 2:3 : les images du film A B C D donnent (A A) (B B) (B C) (C D) (D D).
void telecined(TelecineMeter& meter, int cycles, int bits) {
  constexpr std::array<std::array<int, 2>, 5> kPattern{{{0, 0}, {1, 1}, {1, 2}, {2, 3}, {3, 3}}};
  for (int cycle = 0; cycle < cycles; ++cycle)
    for (const auto& [top, bottom] : kPattern)
      push(meter, interlaced(4 * cycle + top, 4 * cycle + bottom, bits));
}

}  // namespace

TEST_CASE("TelecineMeter : un 2:3 est reconnu", "[media][telecine]") {
  TelecineMeter meter;
  telecined(meter, 20, 8);
  CHECK(meter.verdict() == TelecineMeter::Verdict::Telecine);
  CHECK(meter.repeats() >= meter.moving() / 3);
}

TEST_CASE("TelecineMeter : un 2:3 sur 10 bits aussi", "[media][telecine]") {
  TelecineMeter meter;
  telecined(meter, 20, 10);
  CHECK(meter.verdict() == TelecineMeter::Verdict::Telecine);
}

// B15-01 : du vrai 30i, deux champs de deux instants, qui bougent tous les deux d'une image à l'autre.
TEST_CASE("TelecineMeter : un vrai 30i n'est pas du télécinéma", "[media][telecine]") {
  TelecineMeter meter;
  for (int frame = 0; frame < 100; ++frame) push(meter, interlaced(2 * frame, 2 * frame + 1, 8));
  CHECK(meter.verdict() == TelecineMeter::Verdict::Native);
  CHECK(meter.repeats() == 0);
}

TEST_CASE("TelecineMeter : une image fixe ne conclut pas", "[media][telecine]") {
  TelecineMeter meter;
  for (int frame = 0; frame < 100; ++frame) push(meter, interlaced(0, 0, 8));
  CHECK(meter.verdict() == TelecineMeter::Verdict::Unknown);
}

TEST_CASE("TelecineMeter : trop peu d'images ne concluent pas", "[media][telecine]") {
  TelecineMeter meter;
  telecined(meter, 3, 8);
  CHECK(meter.verdict() == TelecineMeter::Verdict::Unknown);
}

// La mesure sur un fichier : trop court pour conclure, elle ne change rien, et la lecture repart du début.
TEST_CASE("measure_telecine : un clip court ne conclut pas et rend le demuxer au début",
          "[media][telecine]") {
  auto opened = ajp::media::Demuxer::open(ajp::media::test::require_clip(ajp::media::test::seek_clip()));
  REQUIRE(opened.has_value());
  const ajp::media::TrackInfo video = *opened->track(0);
  CHECK_FALSE(ajp::media::measure_telecine(*opened, video).has_value());
  auto first = opened->read();
  REQUIRE(first.has_value());
  REQUIRE(first->has_value());
  CHECK((*first)->pts_us == 0);
}

TEST_CASE("Demuxer : set_inverse_telecine corrige la cadence supposée", "[media][telecine]") {
  auto opened = ajp::media::Demuxer::open(ajp::media::test::require_clip(ajp::media::test::seek_clip()));
  REQUIRE(opened.has_value());
  opened->set_inverse_telecine(0, true);
  CHECK(opened->track(0)->frame_rate.num == 24000);
  opened->set_inverse_telecine(0, false);
  CHECK(opened->track(0)->frame_rate.num == 30000);
  CHECK_FALSE(opened->track(0)->inverse_telecine);
}
