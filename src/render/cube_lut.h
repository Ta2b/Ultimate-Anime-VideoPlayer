// LUT 3D au format .cube (Adobe/Resolve) : lecture, validation et texte de la texture que le hook du look
// lit. Une LUT est une table de size³ couleurs RVB ; la première composante (R) varie le plus vite, puis G,
// puis B.
#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::render {

inline constexpr int kLutMinSize = 17;
inline constexpr int kLutMaxSize = 65;
inline constexpr std::size_t kLookSlots = 15;  // LUT enchaînées au plus dans le menu Couleurs
inline constexpr std::size_t kLutMaxBytes = 16U * 1024U * 1024U;  // un .cube de 65³ pèse ~5 Mo

struct CubeLut {
  int size = 0;            // points par axe (17 à 65)
  std::vector<float> rgb;  // size³ × 3, bornés à [domain_min, domain_max] puis ramenés à [0, 1]
  std::string title;       // TITLE du fichier, vide si absent
  friend bool operator==(const CubeLut&, const CubeLut&) = default;
};

// Refus nommé : fichier trop gros, LUT 1D (LUT_1D_SIZE), taille hors de 17 à 65, nombre de lignes qui ne
// vaut pas size³, valeur illisible ou non finie, DOMAIN_MIN/MAX incohérent.
std::expected<CubeLut, Refusal> parse_cube(std::string_view text);

// Le texte d'un bloc //!TEXTURE libplacebo (rgba16hf, linéaire, bords tenus) nommé `name`, données comprises.
std::string lut_texture_block(const CubeLut& lut, std::string_view name);

// Les entrées d'une LUT qui ne servent jamais : certaines LUT d'HDR rendent (0, 0, 0) pour les couleurs hors
// de leur gamut d'entrée (un tiers du cube pour « HDR grading base »). L'interpolation entre une zone valide
// et ce noir assombrit les pixels proches du bord ; ces entrées (sortie noire pour une entrée qui ne l'est
// pas) sont recopiées du voisin valide le plus proche. Rend le nombre d'entrées remplacées.
std::size_t fill_unreachable(CubeLut& lut);

// La LUT neutre (identité) de `size` points.
CubeLut identity_lut(int size);

// Un demi-flottant IEEE 754 (arrondi au plus proche), pour les textures rgba16hf.
std::uint16_t to_half(float value);

}  // namespace ajp::render
