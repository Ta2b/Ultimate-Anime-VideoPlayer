// Contrat des tuiles de SPAN (« tuiles qui bougent », phase 3 du plan du flux optique NVIDIA, suite du lot A)
// : seules les tuiles qui ont changé repassent dans SPAN, le reste de la sortie reprend celle de l'image
// d'avant.
//
// Grille. Le moteur la donne (gpu::Engine::span_tile_grid) dans les pixels de son entrée (l'image de travail)
// : tile_w × tile_h, depuis (0, 0), ligne par ligne, dernière ligne et colonne tronquées. C'est la tuile de
// ses noyaux, pour qu'une tuile se calcule ou se saute d'un bloc entier.
//
// Halo. Une sortie dépend de l'entrée jusqu'à la portée du réseau (son champ réceptif) : la somme des rayons
// de ses convolutions sur le plus long chemin, calculée par le moteur sur ses couches (SPAN 2× : conv_1,
// trois 3×3 par bloc et to_pixel, soit 3 · blocs + 2 pixels de demi-résolution), arrondie en tuiles
// (reach_cols, reach_rows).
//
// Pourquoi c'est exact. Le moteur calcule TOUTES ses couches sur la même région R : les tuiles à recalculer
// (le cœur C) dilatées du halo. Hors de R, ses tampons intermédiaires tiennent n'importe quoi (une autre
// image, un autre moteur de la voie) ; une valeur fausse ne se propage que d'un rayon par couche, donc d'au
// plus la portée à la sortie. Toute sortie à plus de la portée du bord de R ne voit que des valeurs justes :
// c'est le cas de tout C. La sortie de C est écrite ; le reste de la sortie n'est pas touché. Les bords de
// l'image gardent leur bourrage de zéros (les noyaux le font hors de l'image, pas hors de R).
//
// Décalage. Sur un panoramique d'un nombre entier de pas (shift_step, le pas du dégroupage de l'entrée : 2
// pixels de travail), sans zoom ni rotation, la sortie d'avant se décale de (dx, dy) · scale pixels de sortie
// telle quelle, avant le calcul : seule la bande qui entre dans l'image (et les dessins qui bougent sur le
// décor) est à recalculer. Le décalage ne lit que l'image d'avant : un pixel dont la source sort de l'image
// n'est pas écrit (sa tuile est dans C).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace kirari::kernels {

// Au plus ce nombre de tuiles (128 mots de 32 bits par masque) : au-delà, le moteur calcule l'image entière.
inline constexpr int kSpanTileMaxCount = 4096;
inline constexpr int kSpanTileMaskWords = kSpanTileMaxCount / 32;

struct SpanTileGrid {
  int tile_w = 0;  // pixels de l'entrée du moteur ; 0 : le moteur ne calcule pas par tuiles
  int tile_h = 0;
  int cols = 0;
  int rows = 0;
  int reach_cols = 0;  // halo, en tuiles
  int reach_rows = 0;
  int shift_step = 0;  // pas d'un décalage entier, pixels d'entrée
  int scale = 0;       // pixels de sortie par pixel d'entrée

  constexpr bool usable() const {
    return tile_w > 0 && tile_h > 0 && cols > 0 && rows > 0 && cols * rows <= kSpanTileMaxCount &&
           shift_step > 0 && scale > 0;
  }
  constexpr int count() const { return cols * rows; }
};

constexpr int span_tiles_along(int side, int tile) {
  return tile > 0 ? (side + tile - 1) / tile : 0;
}

// La portée (pixels) en tuiles de côté tile, arrondie vers le haut.
constexpr int span_reach_tiles(int reach_px, int tile) {
  return tile > 0 ? (reach_px + tile - 1) / tile : 0;
}

// Le cœur dilaté d'un rectangle de reach_cols × reach_rows tuiles de chaque côté (séparable : lignes puis
// colonnes). core : cols × rows octets, non nul = tuile à recalculer.
inline std::vector<std::uint8_t> dilate_span_tiles(std::span<const std::uint8_t> core, int cols, int rows,
                                                   int reach_cols, int reach_rows) {
  const auto at = [cols](int c, int r) {
    return static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(c);
  };
  std::vector<std::uint8_t> wide(core.size(), 0);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (core[at(c, r)] == 0) continue;
      const int c0 = std::max(0, c - reach_cols), c1 = std::min(cols - 1, c + reach_cols);
      for (int x = c0; x <= c1; ++x) wide[at(x, r)] = 1;
    }
  }
  std::vector<std::uint8_t> out(core.size(), 0);
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows; ++r) {
      if (wide[at(c, r)] == 0) continue;
      const int r0 = std::max(0, r - reach_rows), r1 = std::min(rows - 1, r + reach_rows);
      for (int y = r0; y <= r1; ++y) out[at(c, y)] = 1;
    }
  }
  return out;
}

// Le masque en bits pour les noyaux : la tuile t (ligne par ligne) au bit t % 32 du mot t / 32. words doit
// tenir kSpanTileMaskWords mots ; une tuile au-delà de kSpanTileMaxCount n'est pas écrite.
inline void pack_span_tiles(std::span<const std::uint8_t> tiles, std::span<std::uint32_t> words) {
  std::ranges::fill(words, 0U);
  const std::size_t count = std::min(tiles.size(), words.size() * 32U);
  for (std::size_t t = 0; t < count; ++t)
    if (tiles[t] != 0) words[t / 32U] |= 1U << (t % 32U);
}

// Le nombre de tuiles non nulles.
inline int count_span_tiles(std::span<const std::uint8_t> tiles) {
  return static_cast<int>(std::ranges::count_if(tiles, [](std::uint8_t v) { return v != 0; }));
}

}  // namespace kirari::kernels
