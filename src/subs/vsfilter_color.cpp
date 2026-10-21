#include "subs/vsfilter_color.h"

#include <algorithm>
#include <cmath>

namespace ajp::subs {
namespace {

// Coefficients de luminance Kr et Kb ; Kg = 1 − Kr − Kb.
struct Luma {
  double kr = 0.0;
  double kb = 0.0;
  double kg() const { return 1.0 - kr - kb; }
};

constexpr Luma kBt601{.kr = 0.299, .kb = 0.114};
constexpr Luma kBt709{.kr = 0.2126, .kb = 0.0722};

struct Ycbcr8 {
  std::uint8_t y = 0;
  std::uint8_t cb = 0;
  std::uint8_t cr = 0;
};

// Valeur sur l'échelle 0-255, arrondie au plus proche puis bornée.
std::uint8_t to_8bit(double value) {
  return static_cast<std::uint8_t>(std::clamp(std::lrint(value), 0L, 255L));
}

double unit(std::uint8_t value) {
  return static_cast<double>(value) / 255.0;
}

// Plage limitée : Y sur [16, 235], Cb et Cr sur 128 ± 112.
Ycbcr8 to_limited_ycbcr(Rgba8 color, Luma k) {
  const double r = unit(color.r);
  const double g = unit(color.g);
  const double b = unit(color.b);
  const double y = k.kr * r + k.kg() * g + k.kb * b;
  const double cb = (b - y) / (2.0 * (1.0 - k.kb));
  const double cr = (r - y) / (2.0 * (1.0 - k.kr));
  return {
      .y = to_8bit(16.0 + 219.0 * y), .cb = to_8bit(128.0 + 224.0 * cb), .cr = to_8bit(128.0 + 224.0 * cr)};
}

Rgba8 from_limited_ycbcr(Ycbcr8 value, Luma k, std::uint8_t alpha) {
  const double y = (static_cast<double>(value.y) - 16.0) / 219.0;
  const double cb = (static_cast<double>(value.cb) - 128.0) / 224.0;
  const double cr = (static_cast<double>(value.cr) - 128.0) / 224.0;
  const double r = y + 2.0 * (1.0 - k.kr) * cr;
  const double b = y + 2.0 * (1.0 - k.kb) * cb;
  const double g = (y - k.kr * r - k.kb * b) / k.kg();
  return {.r = to_8bit(255.0 * r), .g = to_8bit(255.0 * g), .b = to_8bit(255.0 * b), .a = alpha};
}

}  // namespace

bool vsfilter_converts(ScriptMatrix script, const std::optional<VideoMatrix>& video) {
  const bool vsfilter_601 = script == ScriptMatrix::Absent || script == ScriptMatrix::Bt601Tv;
  return vsfilter_601 && video == VideoMatrix{.matrix = ColorMatrix::Bt709, .range = ColorRange::Limited};
}

Rgba8 vsfilter_bt601_on_bt709(Rgba8 color) {
  return from_limited_ycbcr(to_limited_ycbcr(color, kBt601), kBt709, color.a);
}

}  // namespace ajp::subs
