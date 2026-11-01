#include "app/play/calibration_images.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ajp::app::play {

namespace {

constexpr int kW = kCalibrationWidth;
constexpr int kH = kCalibrationHeight;

struct Rgb {
  float r = 0.0F;
  float g = 0.0F;
  float b = 0.0F;
};

Rgb mix(Rgb a, Rgb b, float t) {
  return {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t};
}

float clamp01(float v) {
  return std::clamp(v, 0.0F, 1.0F);
}

// Une toile en codes BT.1886 (0 : noir, 1 : blanc), en pixels de l'image ; chaque forme a un bord adouci
// d'un pixel, comme le trait d'un anime.
class Canvas {
 public:
  Canvas() : px_(static_cast<std::size_t>(kW) * kH) {}

  void vertical(Rgb top, Rgb bottom) {
    for (int y = 0; y < kH; ++y) {
      const Rgb c = mix(top, bottom, static_cast<float>(y) / (kH - 1));
      for (int x = 0; x < kW; ++x) at(x, y) = c;
    }
  }
  void rect(float x0, float y0, float x1, float y1, Rgb c) {
    paint(x0, y0, x1, y1, c,
          [&](float x, float y) { return std::min({x - x0, x1 - x, y - y0, y1 - y}) + 0.5F; });
  }
  void ellipse(float cx, float cy, float rx, float ry, Rgb c) {
    paint(cx - rx, cy - ry, cx + rx, cy + ry, c, [&](float x, float y) {
      const float dx = (x - cx) / rx;
      const float dy = (y - cy) / ry;
      return (1.0F - std::sqrt(dx * dx + dy * dy)) * std::min(rx, ry) + 0.5F;
    });
  }
  void disc(float cx, float cy, float r, Rgb c) { ellipse(cx, cy, r, r, c); }
  // Un triangle pointe en haut (un sapin, une dune vue de loin).
  void peak(float cx, float top, float half, float bottom, Rgb c) {
    paint(cx - half, top, cx + half, bottom, c, [&](float x, float y) {
      const float t = (y - top) / (bottom - top);
      return std::min(half * t - std::abs(x - cx), bottom - y) + 0.5F;
    });
  }
  // Un anneau (un contour) d'épaisseur width.
  void ring(float cx, float cy, float r, float width, Rgb c) {
    paint(cx - r - width, cy - r - width, cx + r + width, cy + r + width, c,
          [&](float x, float y) { return width * 0.5F - std::abs(std::hypot(x - cx, y - cy) - r) + 0.5F; });
  }
  // Une lumière qui s'ajoute : décroissance gaussienne de rayon r.
  void glow(float cx, float cy, float r, Rgb c, float strength) {
    const int x0 = std::max(0, static_cast<int>(cx - 3 * r));
    const int x1 = std::min(kW - 1, static_cast<int>(cx + 3 * r));
    const int y0 = std::max(0, static_cast<int>(cy - 3 * r));
    const int y1 = std::min(kH - 1, static_cast<int>(cy + 3 * r));
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const float d = std::hypot(static_cast<float>(x) - cx, static_cast<float>(y) - cy) / r;
        const float k = strength * std::exp(-d * d);
        Rgb& p = at(x, y);
        p = {p.r + c.r * k, p.g + c.g * k, p.b + c.b * k};
      }
    }
  }
  // Six marches de gris (start, start + step…) côte à côte, posées sur un bandeau de fond.
  void steps(float y0, float y1, float start, float step, Rgb strip) {
    rect(0, y0 - 14, kW, y1 + 14, strip);
    const float w = (kW - 2 * 60.0F) / 6.0F;
    for (int i = 0; i < 6; ++i) {
      const float v = start + step * static_cast<float>(i);
      rect(60 + w * static_cast<float>(i) + 6, y0, 60 + w * static_cast<float>(i + 1) - 6, y1, {v, v, v});
    }
  }

  std::vector<std::uint16_t> encode() const {
    std::vector<std::uint16_t> out(px_.size() * 4);
    for (std::size_t i = 0; i < px_.size(); ++i) {
      out[i * 4 + 0] = static_cast<std::uint16_t>(std::lround(clamp01(px_[i].r) * 65535.0F));
      out[i * 4 + 1] = static_cast<std::uint16_t>(std::lround(clamp01(px_[i].g) * 65535.0F));
      out[i * 4 + 2] = static_cast<std::uint16_t>(std::lround(clamp01(px_[i].b) * 65535.0F));
      out[i * 4 + 3] = 65535;
    }
    return out;
  }

 private:
  Rgb& at(int x, int y) { return px_[static_cast<std::size_t>(y) * kW + static_cast<std::size_t>(x)]; }

  // Couvre la boîte, chaque pixel selon sa distance signée au bord (positive dedans, en pixels).
  template <typename Inside>
  void paint(float x0, float y0, float x1, float y1, Rgb c, Inside inside) {
    const int ix0 = std::max(0, static_cast<int>(std::floor(x0)) - 1);
    const int ix1 = std::min(kW - 1, static_cast<int>(std::ceil(x1)) + 1);
    const int iy0 = std::max(0, static_cast<int>(std::floor(y0)) - 1);
    const int iy1 = std::min(kH - 1, static_cast<int>(std::ceil(y1)) + 1);
    for (int y = iy0; y <= iy1; ++y) {
      for (int x = ix0; x <= ix1; ++x) {
        const float cover = clamp01(inside(static_cast<float>(x) + 0.5F, static_cast<float>(y) + 0.5F));
        if (cover > 0.0F) at(x, y) = mix(at(x, y), c, cover);
      }
    }
  }

  std::vector<Rgb> px_;
};

// Un visage d'anime de face : cheveux, peau avec son côté d'ombre, yeux et reflets.
void face(Canvas& c, float cx, float cy, float s, Rgb hair, Rgb skin, Rgb shade, Rgb iris, float light) {
  c.ellipse(cx, cy - 20 * s, 150 * s, 160 * s, hair);
  c.rect(cx - 150 * s, cy - 20 * s, cx + 150 * s, cy + 170 * s, hair);
  c.ellipse(cx, cy + 25 * s, 110 * s, 135 * s, skin);
  c.ellipse(cx + 64 * s, cy + 40 * s, 44 * s, 104 * s, shade);
  c.ellipse(cx, cy - 95 * s, 125 * s, 55 * s, hair);  // la frange
  for (const float side : {-1.0F, 1.0F}) {
    const float ex = cx + side * 45 * s;
    c.ellipse(ex, cy + 20 * s, 24 * s, 30 * s, {light * 0.95F, light * 0.95F, light});
    c.ellipse(ex, cy + 24 * s, 17 * s, 25 * s, iris);
    c.ellipse(ex, cy + 26 * s, 8 * s, 12 * s, {0.01F, 0.01F, 0.02F});
    c.disc(ex - 7 * s, cy + 13 * s, 5 * s, {light, light, light});
  }
  c.rect(cx - 18 * s, cy + 100 * s, cx + 18 * s, cy + 104 * s, mix(shade, {0, 0, 0}, 0.5F));
}

// Scènes sombres : le gris moyen, les noirs profonds et les couleurs d'une scène tenue pour sombre.
std::vector<std::uint16_t> night_alley() {
  Canvas c;
  c.vertical({0.03F, 0.035F, 0.09F}, {0.015F, 0.015F, 0.03F});
  c.rect(0, 40, 300, 430, {0.045F, 0.040F, 0.055F});
  c.rect(660, 70, 960, 430, {0.050F, 0.045F, 0.060F});
  c.rect(300, 330, 660, 430, {0.030F, 0.030F, 0.040F});
  for (int i = 0; i < 4; ++i) {
    const auto f = static_cast<float>(i);
    c.rect(40 + 60 * f, 120 + 70 * f, 80 + 60 * f, 160 + 70 * f, {0.55F, 0.40F, 0.16F});
    c.rect(700 + 55 * f, 140 + 50 * f, 735 + 55 * f, 175 + 50 * f, {0.12F, 0.10F, 0.08F});
  }
  c.glow(470, 250, 70, {0.30F, 0.22F, 0.12F}, 0.6F);
  c.disc(470, 250, 10, {0.95F, 0.85F, 0.60F});
  c.steps(450, 520, 0.0F, 0.02F, {0.0F, 0.0F, 0.0F});
  return c.encode();
}

std::vector<std::uint16_t> face_in_shadow() {
  Canvas c;
  c.vertical({0.07F, 0.055F, 0.07F}, {0.03F, 0.025F, 0.035F});
  c.rect(250, 400, 710, 540, {0.10F, 0.09F, 0.14F});  // les épaules
  face(c, 480, 230, 1.0F, {0.07F, 0.05F, 0.10F}, {0.30F, 0.22F, 0.19F}, {0.17F, 0.12F, 0.11F},
       {0.16F, 0.11F, 0.26F}, 0.45F);
  c.glow(150, 120, 60, {0.10F, 0.12F, 0.20F}, 0.5F);  // la fenêtre de nuit, au fond
  return c.encode();
}

std::vector<std::uint16_t> moonlit_forest() {
  Canvas c;
  c.vertical({0.03F, 0.05F, 0.11F}, {0.07F, 0.09F, 0.17F});
  c.glow(780, 90, 90, {0.20F, 0.22F, 0.30F}, 0.6F);
  c.disc(780, 90, 38, {0.88F, 0.88F, 0.82F});
  c.rect(0, 420, kW, kH, {0.03F, 0.07F, 0.04F});
  for (int i = 0; i < 9; ++i) {
    const auto f = static_cast<float>(i);
    const float g = 0.05F + 0.012F * static_cast<float>(i % 3);
    c.peak(40 + 110 * f, 170 + 25 * static_cast<float>(i % 2), 70, 440, {0.02F, g, 0.04F});
  }
  c.rect(463, 300, 467, 380, {0.05F, 0.04F, 0.03F});
  c.glow(465, 400, 55, {0.55F, 0.12F, 0.03F}, 0.8F);
  c.ellipse(465, 400, 26, 34, {0.80F, 0.14F, 0.06F});  // la lanterne
  c.ellipse(465, 402, 11, 16, {1.0F, 0.78F, 0.40F});   // son cœur
  return c.encode();
}

std::vector<std::uint16_t> candle_room() {
  Canvas c;
  c.vertical({0.035F, 0.022F, 0.018F}, {0.02F, 0.012F, 0.010F});
  c.glow(480, 230, 230, {0.50F, 0.27F, 0.09F}, 0.9F);
  c.rect(0, 390, kW, kH, {0.11F, 0.055F, 0.03F});     // la table
  c.rect(160, 340, 330, 392, {0.20F, 0.06F, 0.05F});  // un livre
  c.rect(165, 345, 325, 352, {0.55F, 0.50F, 0.40F});
  c.rect(455, 250, 505, 392, {0.80F, 0.74F, 0.62F});  // la bougie
  c.rect(490, 250, 505, 392, {0.55F, 0.46F, 0.36F});
  c.glow(480, 205, 40, {0.70F, 0.50F, 0.20F}, 0.8F);
  c.ellipse(480, 210, 12, 30, {1.0F, 0.85F, 0.50F});  // la flamme
  c.ellipse(480, 218, 6, 14, {1.0F, 0.97F, 0.85F});
  return c.encode();
}

// Scènes claires : le gris moyen, les noirs profonds et la réduction des couleurs d'une scène claire.
std::vector<std::uint16_t> summer_sky() {
  Canvas c;
  c.vertical({0.22F, 0.48F, 0.90F}, {0.66F, 0.80F, 0.97F});
  c.glow(860, 60, 90, {0.4F, 0.4F, 0.3F}, 0.8F);
  for (const auto& [x, y, s] : {std::array{250.0F, 160.0F, 1.0F}, std::array{640.0F, 260.0F, 0.8F}}) {
    for (int i = 0; i < 5; ++i) {
      const auto f = static_cast<float>(i);
      c.disc(x + (f - 2) * 55 * s, y + 25 * s, 50 * s, {0.76F, 0.79F, 0.86F});  // le dessous gris
    }
    for (int i = 0; i < 5; ++i) {
      const auto f = static_cast<float>(i);
      c.disc(x + (f - 2) * 55 * s, y - std::abs(f - 2) * -12 * s, (60 - std::abs(f - 2) * 12) * s,
             {0.97F, 0.97F, 0.98F});
    }
  }
  c.ellipse(300, 600, 520, 170, {0.36F, 0.64F, 0.25F});
  c.ellipse(800, 620, 420, 160, {0.30F, 0.56F, 0.20F});
  return c.encode();
}

std::vector<std::uint16_t> desert_sun() {
  Canvas c;
  c.vertical({0.50F, 0.68F, 0.92F}, {0.96F, 0.86F, 0.66F});
  c.glow(720, 110, 80, {0.5F, 0.45F, 0.3F}, 0.9F);
  c.disc(720, 110, 40, {1.0F, 0.97F, 0.88F});
  c.ellipse(250, 520, 520, 200, {0.88F, 0.60F, 0.30F});
  c.ellipse(760, 560, 460, 230, {0.92F, 0.66F, 0.34F});
  c.ellipse(860, 560, 220, 200, {0.66F, 0.40F, 0.18F});  // la pente à l'ombre
  c.rect(318, 260, 322, 360, {0.25F, 0.18F, 0.12F});
  c.rect(322, 262, 372, 292, {0.86F, 0.12F, 0.10F});  // le seul vrai rouge : le drapeau
  return c.encode();
}

std::vector<std::uint16_t> daylight_portrait() {
  Canvas c;
  c.vertical({0.66F, 0.84F, 0.96F}, {0.85F, 0.92F, 0.98F});
  c.rect(250, 410, 710, 540, {0.95F, 0.95F, 0.96F});  // la chemise blanche
  for (int i = 0; i < 4; ++i) {
    const auto f = static_cast<float>(i);
    c.rect(300 + 110 * f, 440, 306 + 110 * f, 540, {0.80F, 0.82F, 0.87F});  // ses plis
  }
  face(c, 480, 220, 1.0F, {0.36F, 0.22F, 0.14F}, {0.98F, 0.83F, 0.71F}, {0.89F, 0.69F, 0.59F},
       {0.25F, 0.45F, 0.30F}, 1.0F);
  for (const float side : {-1.0F, 1.0F}) c.ellipse(480 + side * 62, 290, 22, 10, {0.96F, 0.68F, 0.64F});
  return c.encode();
}

std::vector<std::uint16_t> snow_outlines() {
  Canvas c;
  c.vertical({0.78F, 0.85F, 0.95F}, {0.92F, 0.94F, 0.98F});
  c.ellipse(480, 470, 700, 150, {0.95F, 0.96F, 1.0F});
  c.ellipse(700, 430, 200, 40, {0.80F, 0.85F, 0.94F});  // l'ombre bleue de la neige
  c.rect(250, 200, 470, 380, {0.0F, 0.0F, 0.0F});       // la maison et son contour noir
  c.rect(254, 204, 466, 376, {0.72F, 0.52F, 0.40F});
  c.peak(360, 120, 135, 204, {0.0F, 0.0F, 0.0F});
  c.peak(360, 126, 128, 200, {0.93F, 0.95F, 1.0F});
  c.rect(290, 240, 340, 290, {0.0F, 0.0F, 0.0F});
  c.rect(294, 244, 336, 286, {0.06F, 0.05F, 0.08F});
  for (const float r : {40.0F, 28.0F}) {
    c.disc(640, 330 - (40 - r) * 2.6F, r, {0.0F, 0.0F, 0.0F});  // le bonhomme de neige
    c.disc(640, 330 - (40 - r) * 2.6F, r - 3, {0.97F, 0.97F, 1.0F});
  }
  c.ring(640, 330, 40, 3, {0.0F, 0.0F, 0.0F});
  c.steps(440, 510, 0.0F, 0.02F, {0.90F, 0.92F, 0.96F});
  return c.encode();
}

struct Scene {
  std::string_view title;
  std::string_view guidance;
  std::vector<std::uint16_t> (*draw)();
};

const std::array<Scene, 2 * kCalibrationPerStep> kScenes{{
    {"Night alley",
     "The six squares at the bottom: the first one stays black like the strip around it, the second is "
     "barely visible, the other four are clearly apart. The walls of the buildings stand out from the sky. "
     "Second square gone: lower Deep blacks. First square grey: raise them.",
     night_alley},
    {"Face in the shadow",
     "The face reads without effort: both eyes, their small highlights and the line between the lit cheek "
     "and the shadowed one. The room must still feel like night. Face hard to read: raise Mid gray. The "
     "room looks lit or greyish: lower it.",
     face_in_shadow},
    {"Moonlit forest",
     "The lantern stays an orange red with a lighter yellow core and a soft halo. The trees stay a deep, "
     "dark green and the sky a night blue. Flat bright red lantern or trees turning neon green: lower "
     "Colors.",
     moonlit_forest},
    {"Candlelit room",
     "The halo around the flame fades smoothly from yellow to orange then brown, without a red ring or "
     "bands. The book and the edge of the table stay readable, the corners of the room stay dark. Red "
     "ring: lower Colors. Corners crushed into one black: lower Deep blacks.",
     candle_room},
    {"Summer sky",
     "The clouds keep their relief: grey shading underneath, bright tops, and the sky stays blue up to the "
     "horizon. Clouds turned into flat white blobs or a dazzling picture: lower Middle gray, bright scene. "
     "Dull picture: raise it.",
     summer_sky},
    {"Desert in the sun",
     "The sand stays ochre and orange, the shaded slope brown; only the small flag is truly red. Sand or "
     "sky near the horizon turning red or pink: raise Colors cut, bright scene.",
     desert_sun},
    {"Daylight portrait",
     "The skin stays a natural peach, with a light blush on the cheeks only; the white shirt keeps its "
     "grey folds. Red or orange skin: raise Colors cut, bright scene. Folds gone in the white: lower "
     "Middle gray, bright scene.",
     daylight_portrait},
    {"Snow and outlines",
     "The black outlines of the house and the snowman stay pure black and sharp. In the bottom strip, the "
     "first square is black and the next five are distinct dark greys. Squares merged together: lower "
     "Deep blacks, bright scene. Grey outlines: raise them.",
     snow_outlines},
}};

}  // namespace

CalibrationImage calibration_image(bool dark, int index) {
  static std::array<std::vector<std::uint16_t>, 2 * kCalibrationPerStep> drawn;
  const int id = (dark ? 0 : kCalibrationPerStep) + std::clamp(index, 0, kCalibrationPerStep - 1);
  const Scene& scene = kScenes[static_cast<std::size_t>(id)];
  std::vector<std::uint16_t>& pixels = drawn[static_cast<std::size_t>(id)];
  if (pixels.empty()) pixels = scene.draw();
  return {.id = id, .title = scene.title, .guidance = scene.guidance, .rgba = pixels};
}

}  // namespace ajp::app::play
