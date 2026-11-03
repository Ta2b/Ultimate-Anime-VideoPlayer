#include "gpu/span_fp8/hard_cases.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>

namespace ajp::gpu::span_fp8 {
namespace {

using Pixel = std::function<void(int x, int y, std::uint8_t (&rgb)[3])>;

HardCase draw(std::string name, int width, int height, const Pixel& pixel) {
  HardCase out{.name = std::move(name), .width = width, .height = height, .rgb = {}};
  const std::size_t plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  out.rgb.resize(3 * plane);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      std::uint8_t rgb[3] = {0, 0, 0};
      pixel(x, y, rgb);
      const std::size_t at =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
      for (std::size_t c = 0; c < 3; ++c) out.rgb[c * plane + at] = rgb[c];
    }
  return out;
}

void gray(std::uint8_t (&rgb)[3], int value) {
  const auto v = static_cast<std::uint8_t>(std::clamp(value, 0, 255));
  rgb[0] = rgb[1] = rgb[2] = v;
}

// Traits blancs d'un pixel sur un fond moyen, à l'angle donné (degrés), tous les period pixels.
Pixel lines(double degrees, int period) {
  const double rad = degrees * 3.14159265358979323846 / 180.0;
  const double nx = -std::sin(rad), ny = std::cos(rad);  // normale au trait
  return [=](int x, int y, std::uint8_t (&rgb)[3]) {
    const double d = nx * x + ny * y;
    const double m = d - std::floor(d / period) * period;
    gray(rgb, m < 1.0 ? 235 : 72);
  };
}

}  // namespace

std::vector<HardCase> hard_cases(int width, int height) {
  if (width < 16 || height < 16 || width % 2 != 0 || height % 2 != 0) return {};
  std::vector<HardCase> out;
  out.push_back(draw("traits 0°", width, height, lines(0.0, 9)));
  out.push_back(draw("traits 30°", width, height, lines(30.0, 11)));
  out.push_back(draw("traits 45°", width, height, lines(45.0, 10)));
  out.push_back(draw("traits 90°", width, height, lines(90.0, 9)));
  // Rampes sombres : 32 paliers de 8 bits (0 à 31) en bandes verticales, une teinte par tiers de hauteur.
  out.push_back(draw("rampes sombres", width, height, [=](int x, int y, std::uint8_t (&rgb)[3]) {
    const int step = x * 32 / width;
    const int third = y * 3 / height;
    rgb[0] = static_cast<std::uint8_t>(step + (third == 0 ? 4 : 0));
    rgb[1] = static_cast<std::uint8_t>(step + (third == 1 ? 4 : 0));
    rgb[2] = static_cast<std::uint8_t>(step + (third == 2 ? 4 : 0));
  }));
  // Bords saturés : damier de 24 pixels des huit coins du cube RGB.
  out.push_back(draw("bords saturés", width, height, [](int x, int y, std::uint8_t (&rgb)[3]) {
    const int corner = (x / 24 + 3 * (y / 24)) % 8;
    for (int c = 0; c < 3; ++c) rgb[c] = static_cast<std::uint8_t>(((corner >> c) & 1) != 0 ? 255 : 0);
  }));
  // Motifs fins : damier d'un pixel à gauche, rayures de deux pixels à droite.
  out.push_back(draw("motifs fins", width, height, [=](int x, int y, std::uint8_t (&rgb)[3]) {
    const bool on = x < width / 2 ? ((x + y) % 2 == 0) : ((x / 2) % 2 == 0);
    gray(rgb, on ? 200 : 40);
  }));
  // Hautes lumières : rampe de 224 à 255 et disque blanc au centre.
  out.push_back(draw("hautes lumières", width, height, [=](int x, int y, std::uint8_t (&rgb)[3]) {
    const double dx = x - width / 2.0, dy = y - height / 2.0;
    const double r = std::min(width, height) / 4.0;
    gray(rgb, dx * dx + dy * dy < r * r ? 255 : 224 + x * 32 / width);
  }));
  // Grain : gris 96 et bruit uniforme de ±12 niveaux, d'un générateur congruentiel fixe.
  out.push_back(draw("grain", width, height,
                     [state = std::uint32_t{0x9E3779B9U}](int, int, std::uint8_t (&rgb)[3]) mutable {
                       for (int c = 0; c < 3; ++c) {
                         state = state * 1664525U + 1013904223U;
                         rgb[c] = static_cast<std::uint8_t>(96 + static_cast<int>(state >> 24U) % 25 - 12);
                       }
                     }));
  return out;
}

}  // namespace ajp::gpu::span_fp8
