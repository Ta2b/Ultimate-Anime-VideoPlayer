// LUT 3D au format .cube (Adobe/Resolve) : lecture, validation et texte de la texture que le hook du look
// lit. Une LUT est une table de size³ couleurs RVB ; la première composante (R) varie le plus vite, puis G,
// puis B.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <span>
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
  std::vector<float> rgb;  // size³ × 3, les sorties telles que le fichier les donne (P3.69)
  // Le domaine d'entrée (DOMAIN_MIN, DOMAIN_MAX, LUT_3D_INPUT_RANGE) : une entrée c lit la grille en
  // (c − domain_min) / (domain_max − domain_min). Les sorties n'en dépendent pas.
  std::array<float, 3> domain_min{0.0F, 0.0F, 0.0F};
  std::array<float, 3> domain_max{1.0F, 1.0F, 1.0F};
  std::string title;  // TITLE du fichier, vide si absent
  friend bool operator==(const CubeLut&, const CubeLut&) = default;
};

// Refus nommé : fichier trop gros, LUT 1D (LUT_1D_SIZE), taille hors de 17 à 65, nombre de lignes qui ne
// vaut pas size³, valeur illisible ou non finie, DOMAIN_MIN/MAX incohérent.
std::expected<CubeLut, Refusal> parse_cube(std::string_view text);

// Une pile de LUT réduite à une seule : en chaque point d'une grille size³, la couleur traverse les LUT dans
// l'ordre comme dans le hook du look (entrée bornée à [0, 1], lecture trilinéaire, mélange à son intensité
// `mixes[i]`, une LUT absente ou d'intensité 0 sautée). Une pile d'une LUT de 17 ou 33 points se refait à
// 65 points sans écart : la grille fine tombe sur la grosse. Le hook lit ensuite une texture au lieu de
// quinze.
CubeLut compose_luts(std::span<const CubeLut* const> luts, std::span<const double> mixes,
                     int size = kLutMaxSize);

// Le texte .cube d'une LUT (TITLE, LUT_3D_SIZE, puis les points, rouge le plus rapide), relisible par
// parse_cube à 1e-6 près.
std::string write_cube(const CubeLut& lut);

// Le texte d'un bloc //!TEXTURE libplacebo (rgba16hf, linéaire, bords tenus) nommé `name`, données comprises.
std::string lut_texture_block(const CubeLut& lut, std::string_view name);

// Les entrées d'une LUT qui ne servent jamais : certaines LUT d'HDR rendent (0, 0, 0) pour les couleurs hors
// de leur gamut d'entrée (un tiers du cube pour « HDR grading base »). L'interpolation entre une zone valide
// et ce noir assombrit les pixels proches du bord ; ces entrées (sortie noire pour une entrée qui ne l'est
// pas) sont recopiées du voisin valide le plus proche. Rend le nombre d'entrées remplacées.
std::size_t fill_unreachable(CubeLut& lut);

// La lecture trilinéaire de la texture (comme le hook du look) : le point c, borné à [0, 1]³, tombe à
// c × (size − 1) dans la grille.
std::array<float, 3> sample_lut(const CubeLut& lut, const std::array<float, 3>& c);

// La LUT neutre (identité) de `size` points.
CubeLut identity_lut(int size);

// Une LUT dont le nom ou le titre dit « pq », « hdr » ou « 2020 » attend du PQ BT.2020 (étalonnée pour l'HDR)
// ; toute autre est lue comme une LUT SDR BT.709.
bool look_is_pq(std::string_view name, std::string_view title);

}  // namespace ajp::render
