// L'horloge des vblanks de l'écran, apprise des dates où les images sont devenues visibles
// (retour de présentation Vulkan, ou wp_presentation). Pure : aucune E/S, aucune horloge propre.
//
// Chaque observation donne la date d'affichage d'une image ; on lui attribue un numéro de vblank
// entier par rapport au premier, et une régression linéaire sur les 256 dernières donne la phase
// et la période de la grille. De là : la fréquence réelle (164,78 Hz et non 165), et la latence
// de sortie, c'est-à-dire l'écart médian entre l'instant visé et l'instant affiché.
//
// Le pas qui numérote les vblanks est la période annoncée de l'écran (set_nominal_period : la durée de
// rafraîchissement de la swapchain ou du compositeur), jamais le plus petit écart vu : un lecteur qui ne
// présente qu'à l'image neuve (23,976 i/s sur 144 Hz, tous les 6 ou 7 vblanks) n'a aucun écart d'un seul
// vblank, et la grille apprenait alors la cadence du contenu (A12-01). Une date qui ne tombe pas près d'un
// multiple entier de la période est écartée. Sans période annoncée (VRR, fréquence inconnue), aucune grille :
// rien ne dirait si un écart vu compte un vblank ou plusieurs.
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

  // Une date plus loin que cela d'un multiple entier de la période (en périodes) n'est pas un vblank.
  static constexpr double kOffGrid = 0.3;
  // La régression ne remplace la période annoncée pour numéroter que si elle en reste à 2 % près.
  static constexpr double kNominalTolerance = 0.02;
  // Autant de dates écartées de suite : c'est la dernière gardée qui était fausse, la grille repart.
  static constexpr int kMaxRejects = 8;
  // Un trou plus long (pause, fenêtre cachée) : le numéro du vblank ne se compte plus sûrement, la grille
  // repart.
  static constexpr std::int64_t kMaxGapUs = 5'000'000;

  // La période annoncée de l'écran, en µs (0 : inconnue). Changée de plus de 0,5 % (autre mode, autre
  // écran) : la grille repart.
  void set_nominal_period(double period_us);
  double nominal_period_us() const { return nominal_us_; }

  // visible_us : date où l'image est apparue ; target_us : date visée (0 : inconnue).
  void observe(std::int64_t visible_us, std::uint64_t id, std::int64_t target_us);

  // La grille ajustée ; rien tant que kMinSamples dates n'ont pas été vues, ou sans période annoncée.
  std::optional<VblankGrid> grid() const { return fit_; }
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
  void restart(std::int64_t visible_us);  // la grille repart de cette date
  void push(std::int64_t visible_us, std::int64_t steps);
  void refit();  // la régression des dates gardées, dans fit_
  std::deque<Sample> times_;
  std::deque<std::int64_t> latencies_;
  std::optional<VblankGrid> fit_;
  double nominal_us_ = 0.0;
  std::int64_t last_k_ = 0;
  std::int64_t last_us_ = 0;
  int rejects_ = 0;
  bool have_last_ = false;
};

}  // namespace ajp::core
