// La mire de calibration du HDR (Paramètres > HDR > « Calibrer le pic ») : un cadre au maximum du PQ
// (10 000 nits) et, au milieu, un carré à N nits, sur fond noir. L'écran écrête tout ce qui dépasse son pic :
// on monte N jusqu'à ce que le carré se fonde dans le cadre, et N est alors le pic réel de l'écran. La mire
// s'écrit telle quelle dans la swapchain PQ, sans grade, sans adaptation de tons ni interface entre les deux
// (placebo_pattern.cpp) : ce qui sort est exactement le PQ de N.
#pragma once

#include <algorithm>
#include <cmath>

#include "app/play/links.h"

namespace ajp::app::play {

struct HdrPattern {
  int nits = 600;          // le carré du milieu
  int frame_nits = 10000;  // le cadre autour
  double area = 0.10;      // part de l'écran prise par le cadre : 10 % (pic), 50 % (baisse de l'ABL)
  double center_x = 0.36;  // centre du cadre, en part de la largeur : à gauche du panneau des paramètres
  friend bool operator==(const HdrPattern&, const HdrPattern&) = default;
};

// Le signal PQ (SMPTE ST 2084, 0 à 1) de ces nits.
inline double pq_of_nits(double nits) {
  constexpr double m1 = 2610.0 / 16384.0;
  constexpr double m2 = 2523.0 / 4096.0 * 128.0;
  constexpr double c1 = 3424.0 / 4096.0;
  constexpr double c2 = 2413.0 / 4096.0 * 32.0;
  constexpr double c3 = 2392.0 / 4096.0 * 32.0;
  const double y = std::pow(std::clamp(nits, 0.0, 10000.0) / 10000.0, m1);
  return std::pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}

struct PatternRect {
  int x0 = 0;
  int y0 = 0;
  int x1 = 0;
  int y1 = 0;
  friend bool operator==(const PatternRect&, const PatternRect&) = default;
};
struct PatternLayout {
  PatternRect frame;   // le cadre (10 000 nits)
  PatternRect square;  // le carré (N nits), au milieu du cadre, un tiers de son côté
};

// Le cadre garde la forme de l'écran et couvre `area` de sa surface ; il reste dans l'écran.
inline PatternLayout pattern_layout(OutputSize out, const HdrPattern& p) {
  const double side = std::sqrt(std::clamp(p.area, 0.01, 1.0));
  const double fw = side * out.width;
  const double fh = side * out.height;
  const double cx = std::clamp(p.center_x * out.width, fw / 2.0, out.width - fw / 2.0);
  const double cy = out.height / 2.0;
  const auto rect = [cx, cy](double w, double h) {
    const auto x0 = static_cast<int>(std::lround(cx - w / 2.0));
    const auto y0 = static_cast<int>(std::lround(cy - h / 2.0));
    return PatternRect{x0, y0, x0 + static_cast<int>(std::lround(w)), y0 + static_cast<int>(std::lround(h))};
  };
  const double sq = std::min(fw, fh) / 3.0;
  return PatternLayout{.frame = rect(fw, fh), .square = rect(sq, sq)};
}

}  // namespace ajp::app::play
