// Empaquetage de rectangles en étagères, borné par un côté maximal.
// Fonction pure, sans état : mêmes tailles, mêmes positions.
#pragma once

#include <expected>
#include <span>
#include <vector>

#include "core/types.h"

namespace ajp::subs {

// Écart minimal entre deux rectangles : un échantillonnage bilinéaire au bord
// d'un rectangle ne lit jamais son voisin.
inline constexpr int kAtlasPadding = 1;

struct PackSize {
  int width = 0;
  int height = 0;
};

struct PackPoint {
  int x = 0;
  int y = 0;
  friend bool operator==(const PackPoint&, const PackPoint&) = default;
};

struct Packing {
  int width = 0;  // étendue utilisée, au plus max_side
  int height = 0;
  std::vector<PackPoint> positions;  // positions[i] : coin haut gauche de sizes[i]
};

// Place chaque rectangle dans un atlas d'au plus max_side × max_side, sans
// recouvrement et avec kAtlasPadding pixels d'écart. Résultat déterministe.
// Refus : taille nulle ou négative, max_side hors de [1, 65536], ou contenu
// qui ne tient pas (SubtitleAtlasFull).
std::expected<Packing, Refusal> pack(std::span<const PackSize> sizes, int max_side);

}  // namespace ajp::subs
