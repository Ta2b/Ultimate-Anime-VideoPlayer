// Le modèle de coût d'une paire interpolée : t(n) = a + b·n, a le mouvement (une fois par paire), b le
// travail par image créée (warp, SPAN, composition). Moyennes glissantes par nombre d'images, poids qui
// s'estompent (demi-vie de 6 s), p90 de la scène en cours. Pur ; extrait d'AutoRate (étape 7.2) pour que
// le Rapide à pas constant s'en serve aussi.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ajp::core {

struct CostLine {
  double a = 0.0;
  double b = 0.0;
};

class CostModel {
 public:
  static constexpr int kMaxImages = 9;  // kAutoMaxStep + 1
  static constexpr std::int64_t kHalfLifeUs = 6'000'000;

  // Une paire terminée : `images` créées (source comprise), `cost_us` de service.
  void add(int images, std::int64_t cost_us, std::int64_t now_us);
  // La droite ajustée ; rien sans assez de mesures.
  std::optional<CostLine> fit(std::int64_t now_us) const;
  // Rapport p90 / moyenne de la scène en cours, entre 1 et 1,5.
  double spike(std::int64_t now_us) const;
  // Coupe de scène : les poids perdent la moitié de leur valeur.
  void fade(std::int64_t now_us);
  void clear();
  // Mesures gardées pour ce nombre d'images (au plus 16).
  int samples(int images) const;

 private:
  struct ImageCost {
    double ewma_us = 0.0;
    double weight = 0.0;
    std::int64_t at_us = 0;
    std::array<float, 16> ring{};
    std::size_t filled = 0;
    std::size_t next = 0;
  };
  double weight_of(const ImageCost& c, std::int64_t now_us) const;
  std::array<ImageCost, kMaxImages + 1> costs_{};
};

}  // namespace ajp::core
