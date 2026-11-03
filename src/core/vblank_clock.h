// L'horloge des vblanks de l'écran, apprise des dates où les images sont devenues visibles
// (retour de présentation Vulkan, ou wp_presentation). Pure : aucune E/S, aucune horloge propre.
//
// Chaque observation donne la date d'affichage d'une image ; on lui attribue un numéro de vblank
// entier par rapport au premier, et une régression linéaire sur les 256 dernières donne la phase
// et la période de la grille. De là : la fréquence réelle (164,78 Hz et non 165), et la latence
// de sortie, c'est-à-dire l'écart médian entre l'instant visé et l'instant affiché.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>

namespace ajp::core {

struct VblankGrid {
  double phase_us = 0.0;   // date d'un vblank de la grille
  double period_us = 0.0;  // durée d'un vblank
  // La date du k-ième vblank après la phase.
  double at(std::int64_t k) const { return phase_us + period_us * static_cast<double>(k); }
  // Le numéro (fractionnaire) du vblank à la date t_us.
  double index_of(double t_us) const { return (t_us - phase_us) / period_us; }
};

class VblankClock {
 public:
  static constexpr std::size_t kWindow = 256;
  static constexpr std::size_t kMinSamples = 8;

  // visible_us : date où l'image est apparue ; target_us : date visée (0 : inconnue).
  void observe(std::int64_t visible_us, std::uint64_t id, std::int64_t target_us);

  // La grille ajustée ; rien tant que kMinSamples dates n'ont pas été vues.
  std::optional<VblankGrid> grid() const;
  // La fréquence mesurée, en Hz ; 0 sans grille.
  double hz() const;
  // Médiane de (affiché − visé) sur les images qui avaient une date visée ; 0 sans mesure.
  std::int64_t output_latency_us() const;
  std::size_t samples() const { return times_.size(); }
  void reset();

 private:
  struct Sample {
    std::int64_t visible_us;
    std::int64_t k;  // numéro de vblank depuis la première observation
  };
  std::deque<Sample> times_;
  std::deque<std::int64_t> latencies_;
  double period_guess_us_ = 0.0;
  std::int64_t first_us_ = 0;
  std::int64_t last_k_ = 0;
  std::int64_t last_us_ = 0;
  std::uint64_t last_id_ = 0;
  bool have_last_ = false;
};

}  // namespace ajp::core
