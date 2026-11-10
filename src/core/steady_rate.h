// Rapide à pas constant par plan (chantier 7.4, docs/auto-libre.md) : le nombre N d'images de sortie par
// image source, gardé constant dans un plan, choisi par un modèle de coût. Pur : le lecteur donne des
// instants (µs, horloge hôte) et des mesures, jamais d'horloge à lire ici. Derrière --fast-steady ; le Rapide
// d'aujourd'hui (core/free_rate.h) reste le défaut.
//
// Coût d'une période source P = 1 / fps : t(N) + d·N, où t(n) = a + b·n est le coût GPU d'une paire de n
// images (core/cost_model.h) et d le coût de l'affichage d'une image. N est le plus grand candidat tel que
// (t(N) + d·N)·(1 + marge) ≤ P, sous le plafond de l'écran : en VRR, des demi-pas de 1 à 1 + (demi-pas − 1)/2
// et N·fps ≤ fréquence ; à fréquence fixe, les facteurs qui calent l'écran (lock_factor de hz / (k·fps)).
//
// - A priori : la droite donnée par le banc ou le profil (prior), corrigée par l'échelle s mesurée sur les
// plans
//   précédents (coût mesuré / coût a priori).
// - Un plan : N ← choix(a priori × s) à la coupe ; après `fit_pairs` paires mesurées, N ← choix(droite du
//   plan), une seule fois. Les quatre premières paires d'un plan qui dépassent le budget font le même choix
//   tout de suite.
// - Urgence : `urgent_losses` images perdues en `urgent_window_us` : N ← min(N − un demi-pas,
//   choix(a priori × s)), gelé jusqu'à la coupe suivante.
// - Repli : à N = 1 seulement, 5 % d'images perdues ou plus sur 5 s (et au moins 10).
// - Sans a priori (ni banc ni profil) : après `early_pairs` paires mesurées, la droite a = 0, b = coût moyen
//   par image. Elle surestime le coût au-dessus du nombre d'images mesuré (a/n + b ≥ b) : N monte prudemment,
//   puis l'échelle mesurée la corrige plan après plan.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "core/cost_model.h"
#include "core/free_rate.h"
#include "core/profile.h"

namespace ajp::core {

struct FreeParams {
  double margin = 0.15;
  int half_steps = 20;  // en VRR : N de 1 à 10,5 par demi-pas
  int urgent_losses = 2;
  std::int64_t urgent_window_us = 1'000'000;
  int fit_pairs = 24;   // paires mesurées d'un plan avant de le réajuster
  int early_pairs = 4;  // premières paires d'un plan jugées contre le budget
  int fallback_per_mille = 50;
  std::int64_t fallback_window_us = 5'000'000;
  int fallback_min_lost = 10;
  double display_smoothing = 0.1;      // lissage de d
  std::int64_t settle_us = 1'200'000;  // après seek ou reprise : mesures ignorées
  // Après la mise en route, un seek ou une reprise : les pertes ne comptent pas pour le repli (le démarrage
  // perd toujours quelques images, comme FreeRate::kStartUs).
  std::int64_t start_us = 5'000'000;
  int fallback_min_buckets = 4;  // pertes réparties sur autant de tranches de 250 ms
};

struct FreeDisplay {
  bool vrr = true;
  double hz = 60.0;     // VRR : l'écran ou le plafond ; fréquence fixe : l'écran, sur lequel on cale
  double cap_hz = 0.0;  // fréquence fixe : le plafond choisi, qui borne sans caler (A12-04) ; 0 : aucun
};

struct SteadyInfo {
  Ratio factor;
  double scale = 1.0;           // s : coût mesuré / a priori
  std::int64_t display_us = 0;  // d lissé
  std::int64_t budget_us = 0;   // P / (1 + marge)
  bool frozen = false;          // urgence : gelé jusqu'à la coupe
  std::uint32_t lost = 0;       // images perdues depuis le début
  std::uint32_t urgencies = 0;
};

class SteadyRate {
 public:
  void configure(const FreeParams& params, const FreeDisplay& display, double source_fps);
  // La droite a priori t(n) = a + b·n, en µs, n images par paire (source comprise).
  void prior(CostLine line);
  // Une paire de `images` images a coûté `gpu_us` au GPU (sans l'attente d'un emplacement).
  void pair_cost(int images, std::int64_t gpu_us, std::int64_t now_us);
  // Le coût de l'affichage d'une image (toutes les passes de l'écran), lissé.
  void display_cost(std::int64_t d_us);
  void scene_cut(std::int64_t now_us);
  // `count` images perdues à l'écran (sautées ou manquées).
  void lost(std::int64_t now_us, int count);
  // `images` images produites par une paire (compte du repli).
  void produced(int images, std::int64_t now_us);
  // Seek, reprise, (re)chargement : les mesures repartent, le facteur reste.
  void settle(std::int64_t now_us);

  // N en fraction réduite (num / den), au moins 1.
  Ratio factor() const { return factor_; }
  // Vrai une fois après chaque changement de N.
  bool take_changed();
  bool should_fall_back(std::int64_t now_us) const;
  SteadyInfo info() const;

  // Le plus grand candidat que la droite tient dans le budget (1 si aucun).
  Ratio choose(const CostLine& line) const;
  // Les candidats, du plus petit au plus grand (1 compris).
  const std::vector<Ratio>& candidates() const { return candidates_; }

 private:
  struct Bucket {
    std::int64_t index = -1;
    int produced = 0;
    int lost = 0;
  };
  Bucket& bucket(std::int64_t now_us);
  void set_factor(Ratio factor);
  CostLine scaled_prior() const;
  void rechoose_from_plan(std::int64_t now_us);
  double cost_at(const CostLine& line, double n) const;

  FreeParams params_;
  FreeDisplay display_;
  double fps_ = 24.0;
  std::vector<Ratio> candidates_{Ratio{1, 1}};
  CostLine prior_{};
  bool has_prior_ = false;
  double scale_ = 1.0;
  double display_us_ = 0.0;
  Ratio factor_{1, 1};
  bool changed_ = false;
  bool frozen_ = false;
  bool adjusted_ = false;  // le plan a eu son réajustement
  CostModel plan_;
  int plan_pairs_ = 0;
  double plan_ratio_sum_ = 0.0;  // coût mesuré / a priori, paire par paire, dans ce plan
  int plan_ratio_n_ = 0;
  int early_over_ = 0;
  double boot_per_image_sum_ = 0.0;  // sans a priori : coût par image des premières paires
  int boot_n_ = 0;
  std::int64_t settled_until_us_ = 0;
  std::int64_t quiet_until_us_ = -1;  // pertes hors du compte du repli jusque-là ; −1 : à la première image
  std::array<std::int64_t, 8> recent_losses_{};  // instants des dernières pertes
  std::size_t next_loss_ = 0;
  std::uint32_t lost_total_ = 0;
  std::uint32_t urgencies_ = 0;
  std::array<Bucket, 24> buckets_{};
};

}  // namespace ajp::core
