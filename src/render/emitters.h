// Méga plan A5 : les mesures d'émetteurs v2, référence CPU de la carte des émetteurs de la méthode « HDR
// anime ». Trois échelles (le pixel, le bloc 8 × 8, une couronne de blocs à 2 à 4 blocs de lui), un rapport
// centre-couronne en diaphragmes, une porte sur le niveau qui admet aussi une lumière colorée (néon, écran)
// moins claire qu'un blanc, et la forme d'une étincelle (le pixel qui domine nettement son bloc). Elle sert
// de juge aux essais du shader et d'étiquette au jeu de données du réseau appris (A6, tools/grade/emitters/).
#pragma once

#include <cstddef>
#include <vector>

namespace ajp::render {

inline constexpr int kEmitterBlock = 8;
inline constexpr int kEmitterRingIn = 2;  // la couronne : les blocs à une distance de 2 à 4 (Tchebychev)
inline constexpr int kEmitterRingOut = 4;

// Une image linéaire : la luminance en nits et la saturation (max − min) / max du RGB linéaire, par pixel.
struct EmitterImage {
  int width = 0;
  int height = 0;
  std::vector<float> nits;
  std::vector<float> chroma;
  std::size_t at(int x, int y) const {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
  }
};

struct EmitterParams {
  float white_nits = 203.0F;  // le blanc de la source
};

// La part d'émetteur de chaque pixel, de 0 à 1 ; vide si l'image est mal formée.
std::vector<float> emitter_map_v2(const EmitterImage& image, const EmitterParams& params = {});

}  // namespace ajp::render
