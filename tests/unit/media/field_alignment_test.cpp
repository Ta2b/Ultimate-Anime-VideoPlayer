// C15-01 : des champs du même instant (2:2) se distinguent de champs de deux instants (vrai entrelacé).
#include "media/field_alignment.h"

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::media::FieldAlignmentMeter;

namespace {

constexpr int kWidth = 128;
constexpr int kHeight = 96;

// Des bandes obliques douces qui avancent de 3 pixels par instant, avec un relief lent en ligne : un dessin
// d'anime, sans bruit d'une ligne à l'autre.
int scene(int x, int y, int t) {
  const int u = (x + y + 3 * t) % 32;
  return 30 + (u < 16 ? u * 12 : (32 - u) * 12) + (y / 8) % 4 * 5;
}

std::vector<std::uint8_t> picture(int top_time, int bottom_time, int bits) {
  const int bytes = bits > 8 ? 2 : 1;
  std::vector<std::uint8_t> out(static_cast<std::size_t>(kWidth * kHeight * bytes));
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const int value = scene(x, y, y % 2 == 0 ? top_time : bottom_time) << (bits - 8);
      const auto at = static_cast<std::size_t>((y * kWidth + x) * bytes);
      out[at] = static_cast<std::uint8_t>(value & 0xFF);
      if (bytes == 2) out[at + 1] = static_cast<std::uint8_t>(value >> 8);
    }
  }
  return out;
}

void push(FieldAlignmentMeter& meter, const std::vector<std::uint8_t>& image, int bits) {
  const int bytes = bits > 8 ? 2 : 1;
  meter.push(image.data(), static_cast<std::ptrdiff_t>(kWidth * bytes), kWidth, kHeight, bytes, bits);
}

}  // namespace

TEST_CASE("FieldAlignmentMeter : deux champs du même instant sont alignés", "[media][interlace]") {
  for (const int bits : {8, 10}) {
    FieldAlignmentMeter meter;
    for (int n = 0; n < 60; ++n) push(meter, picture(n, n, bits), bits);  // 2:2 : (haut n, bas n)
    CAPTURE(bits, meter.moving(), meter.combed());
    CHECK(meter.verdict() == FieldAlignmentMeter::Verdict::Aligned);
  }
}

TEST_CASE("FieldAlignmentMeter : du vrai entrelacé est peigné", "[media][interlace]") {
  FieldAlignmentMeter meter;
  for (int n = 0; n < 60; ++n) push(meter, picture(2 * n, 2 * n + 1, 8), 8);  // haut à t, bas à t + 1
  CAPTURE(meter.moving(), meter.combed());
  CHECK(meter.verdict() == FieldAlignmentMeter::Verdict::Combed);
}

TEST_CASE("FieldAlignmentMeter : une image fixe ou trop peu d'images ne concluent pas",
          "[media][interlace]") {
  FieldAlignmentMeter still;
  for (int n = 0; n < 60; ++n) push(still, picture(0, 0, 8), 8);
  CHECK(still.verdict() == FieldAlignmentMeter::Verdict::Unknown);
  FieldAlignmentMeter few;
  for (int n = 0; n < 10; ++n) push(few, picture(n, n, 8), 8);
  CHECK(few.verdict() == FieldAlignmentMeter::Verdict::Unknown);
}
