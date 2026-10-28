// La taille de la fenêtre à sa création. SDL la compte en points sous macOS et Wayland (la densité
// des pixels fait le reste), en pixels de l'écran sous Windows et X11 : là, 1280 × 720 deviendrait
// une petite fenêtre sur un écran à 150 % ou 200 %, dont l'interface, mise à l'échelle du bureau,
// déborderait.
#pragma once

#include <algorithm>
#include <cmath>

namespace ajp::app::play {

struct WindowSize {
  int width = 0;
  int height = 0;
  bool operator==(const WindowSize&) const = default;
};

// width × height points pour une fenêtre dont l'échelle du bureau, rapportée aux coordonnées de
// SDL, est content_scale (SDL_GetWindowDisplayScale / SDL_GetWindowPixelDensity) : multipliés
// quand elle dépasse 1, puis ramenés au besoin aux 9/10 de la zone utile de l'écran
// (usable_width × usable_height, 0 : inconnue), proportions gardées. Sans échelle, inchangés.
inline WindowSize window_size_for(int width, int height, float content_scale, int usable_width,
                                  int usable_height) {
  if (!(content_scale > 1.01F)) return {.width = width, .height = height};
  double w = static_cast<double>(width) * content_scale;
  double h = static_cast<double>(height) * content_scale;
  if (usable_width > 0 && usable_height > 0) {
    const double fit = std::min({1.0, usable_width * 0.9 / w, usable_height * 0.9 / h});
    w *= fit;
    h *= fit;
  }
  return {.width = std::max(width, static_cast<int>(std::lround(w))),
          .height = std::max(height, static_cast<int>(std::lround(h)))};
}

}  // namespace ajp::app::play
