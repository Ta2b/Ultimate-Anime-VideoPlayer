#include "kernels/ref/deband.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "kernels/ref/check.h"

namespace ajp::kernels::ref {

DebandOffset deband_offset(int x, int y) {
  const auto ux = static_cast<std::uint32_t>(x);
  const auto uy = static_cast<std::uint32_t>(y);
  const float distance = kDebandRadius * deband_uniform(ux, uy, 1U);
  const float angle = 6.2831853F * deband_uniform(ux, uy, 2U);
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  return {.dx = static_cast<int>(std::nearbyint(distance * c)),
          .dy = static_cast<int>(std::nearbyint(distance * s))};
}

float deband_rule(float value, float a, float b, float d, float e) {
  const float average = 0.25F * (a + b + d + e);
  return std::fabs(average - value) > kDebandThreshold ? value : average;
}

Result<RgbImage> deband(const RgbImage& image) {
  if (auto ok = check_rgb(image, "débandage"); !ok) return std::unexpected(ok.error());
  const int w = image.width();
  const int h = image.height();
  RgbImage out(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const DebandOffset o = deband_offset(x, y);
      const auto at = [&](const PlaneF& p, int ox, int oy) {
        return p.at(std::clamp(x + ox, 0, w - 1), std::clamp(y + oy, 0, h - 1));
      };
      for (std::size_t c = 0; c < 3; ++c) {
        const PlaneF& p = image.planes[c];
        out.planes[c].at(x, y) = deband_rule(p.at(x, y), at(p, o.dx, o.dy), at(p, -o.dx, -o.dy),
                                             at(p, -o.dy, o.dx), at(p, o.dy, -o.dx));
      }
    }
  }
  return out;
}

}  // namespace ajp::kernels::ref
