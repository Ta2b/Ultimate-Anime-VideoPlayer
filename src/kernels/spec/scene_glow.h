#pragma once

#include <cstdint>
#include <vector>

namespace kirari::kernels {

inline constexpr int kGlowAnchorBlock = 16;
inline constexpr int kGlowGuideBlock = 32;

// Carte d'éclats confirmés pour la source droite du ticket, sans lissage temporel.
struct SceneGlowTiles {
  std::uint64_t ticket = 0;
  int cols = 0;
  int rows = 0;
  double cover_x = 1.0;
  double cover_y = 1.0;
  std::vector<float> scores;
};

}  // namespace kirari::kernels
