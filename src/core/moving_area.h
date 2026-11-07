// La zone qui bouge entre deux images (P6.6, P6.18) : d'après l'écart de chaque bloc 16 × 16 que la mesure
// des dessins tenus relit déjà, les blocs qui changent et la boîte qui les tient. Le lecteur s'en sert pour
// mesurer ce qu'une passe de SPAN par tuiles ou de RIFE sur la seule boîte épargnerait, avant de la coder
// dans les moteurs (formes dynamiques, recadrage, recollage) ; rien n'en dépend encore à l'image.
#pragma once

#include <cstdint>
#include <span>

namespace ajp::core {

struct MovingArea {
  std::uint32_t blocks = 0;  // blocs dont l'écart dépasse le seuil
  std::uint32_t total = 0;   // blocs de l'image
  // La boîte des blocs qui changent, en blocs, [x0, x1) × [y0, y1) ; vide si blocks est nul.
  std::uint32_t x0 = 0;
  std::uint32_t y0 = 0;
  std::uint32_t x1 = 0;
  std::uint32_t y1 = 0;

  // Part des blocs qui changent : ce qu'un SPAN par tuiles calculerait encore (P6.6), de 0 à 1.
  double block_share() const {
    return total == 0 ? 0.0 : static_cast<double>(blocks) / static_cast<double>(total);
  }
  // Part de l'image dans la boîte : ce qu'un RIFE sur la boîte calculerait encore (P6.18), de 0 à 1.
  double box_share() const {
    return total == 0
               ? 0.0
               : static_cast<double>(x1 - x0) * static_cast<double>(y1 - y0) / static_cast<double>(total);
  }
};

// deviation : l'écart de chaque bloc, ligne par ligne, blocks_x par ligne (en 1/256 de niveau 8 bits, comme
// PairScores::worst_block) ; un bloc change si son écart dépasse limit (même unité).
MovingArea moving_area(std::span<const std::uint32_t> deviation, std::uint32_t blocks_x, std::uint32_t limit);

// Le cumul d'une lecture, pour son journal : la part moyenne calculée encore par chaque piste, et les paires
// où la boîte tient moins de la moitié de l'image (là où RIFE sur la boîte gagnerait).
struct MovingTally {
  std::uint64_t pairs = 0;
  std::uint64_t small_boxes = 0;
  double box_sum = 0.0;
  double block_sum = 0.0;
  void add(const MovingArea& m) {
    if (m.total == 0) return;
    ++pairs;
    box_sum += m.box_share();
    block_sum += m.block_share();
    if (m.box_share() < 0.5) ++small_boxes;
  }
};

// Le seuil des dessins tenus (kernels::kHeldThreshold, en centièmes de niveau) dans l'unité de deviation : un
// bloc tenu ne bouge pas.
constexpr std::uint32_t moving_limit(std::uint32_t held_threshold) {
  return held_threshold * 256U / 100U;
}

}  // namespace ajp::core
