// Poids déjà arrondis à la grille FP8 E4M3 (import des Q/DQ, onnx_fp8.h) : retrouver leurs octets
// E4M3 et l'échelle de chaque canal de sortie, pour les noyaux à matrices FP8. L'IR ne garde que
// les valeurs q × échelle ; l'échelle d'un canal est son maximum absolu / 448 (l'étalonnage par
// amax place le plus grand poids sur 448), celle du tenseur entier d'abord quand elle suffit
// (échelle par tenseur). De même pour la grille INT8 (import des Q/DQ INT8) : entiers de [-128, 127] et
// une échelle par canal.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "gpu/ir/ir.h"

namespace kirari::gpu::ir {

struct Fp8Weights {
  std::vector<std::uint8_t> bytes;  // E4M3, dans l'ordre des poids : [cout][cin][kh][kw]
  std::vector<float> scale;         // une échelle par canal de sortie
};

// nullopt si les poids ne sont pas rangés sur une grille E4M3 à échelle par tenseur ou par canal
// (à un arrondi float16 près), ou s'ils ne sont pas de rang 4.
std::optional<Fp8Weights> recover_fp8_weights(const Value& weights);

struct Int8Weights {
  std::vector<std::uint8_t> bytes;  // entiers INT8 en complément à deux, dans l'ordre des poids
  std::vector<float> scale;         // une échelle par canal de sortie
};

// nullopt si les poids ne sont pas rangés sur une grille INT8 à échelle par canal (à un arrondi float16
// près), ou s'ils ne sont pas de rang 4. L'échelle d'un canal vaut son maximum absolu / k, k entier de 1
// à 128 : 127 d'ordinaire (étalonnage symétrique par amax), plus petit quand le plus grand entier du
// canal l'est (échelle par tenseur, ou canal de petits poids).
std::optional<Int8Weights> recover_int8_weights(const Value& weights);

}  // namespace kirari::gpu::ir
