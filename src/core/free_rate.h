// Mode « Auto libre » : le temps de service mesuré par image, la contre-pression et le repli. Pur : le
// lecteur donne des instants (µs, horloge hôte) et des mesures, jamais d'horloge à lire ici.
//
// Aucun palier, aucun multiplicateur, aucune cadence visée : le producteur sort une image dès que le GPU
// peut la livrer à l'instant où elle sera vue (core/free_grid.h). Cette classe ne fixe que l'écart
// minimal entre deux images consécutives d'une paire, `step_us` : le temps de service mesuré d'une image
// × 1,10, élargi par la contre-pression quand des images sont perdues à l'écran.
//
// - Temps de service : la durée GPU propre à une image créée (jamais l'attente d'un slot du ring), moyenne
//   qui monte vite (une scène lourde) et descend lentement (elle ne s'emballe pas sur un coup de chance).
// - Contre-pression : une image sautée ou manquée à l'écran élargit l'écart tout de suite (× 1,12 par
//   image perdue, au plus × 2) ; elle ne se relâche qu'après 1 s sans perte, puis lentement (3 s).
// - Repli : 3 % d'images perdues ou plus sur 3 s (et au moins 6) rendent la main au mode Auto.
#pragma once

#include <array>
#include <cstdint>

namespace ajp::core {

struct FreeInfo {
  double fps = 0.0;             // images créées par seconde (fenêtre de 2 s) ; 0 : pas encore mesuré
  int load_pct = -1;            // temps de service × cadence en % d'une seconde de GPU ; -1 : inconnu
  std::int64_t step_us = 0;     // l'écart de service planifié ; 0 : inconnu
  std::int64_t service_us = 0;  // le temps de service d'une image mesuré ; 0 : inconnu
  double pressure = 1.0;        // contre-pression (1 : aucune)
  std::uint32_t lost = 0;       // images perdues depuis le début
};

class FreeRate {
 public:
  static constexpr double kMargin = 1.10;
  static constexpr double kPressureStep = 1.12;
  static constexpr double kPressureMax = 2.0;
  static constexpr std::int64_t kCalmUs = 1'000'000;     // sans perte avant que la pression se relâche
  static constexpr std::int64_t kReleaseUs = 3'000'000;  // constante de temps du relâchement
  static constexpr std::int64_t kFallbackWindowUs = 3'000'000;
  static constexpr std::int64_t kSettleUs = 1'200'000;  // après seek ou reprise : mesures ignorées
  static constexpr std::int64_t kStartUs = 3'000'000;   // après l'activation du mode
  static constexpr std::int64_t kMaxSampleUs = 300'000;
  static constexpr std::int64_t kBucketUs = 250'000;
  static constexpr int kFallbackMinLost = 6;
  static constexpr int kFallbackPerMille = 30;  // 3 %

  // L'écart minimal entre deux images (période de l'écran ou du plafond), en µs de temps réel.
  void set_min_gap(std::int64_t us) { min_gap_us_ = us; }
  std::int64_t min_gap_us() const { return min_gap_us_; }

  // Une paire de `images` (>= 2) images a occupé le GPU `busy_us` ; une image seule ne dit rien du coût
  // d'une image interpolée.
  void cost(int images, std::int64_t busy_us, std::int64_t now_us);
  // `images` images planifiées par la paire qui vient d'être préparée.
  void produced(int images, std::int64_t now_us);
  // `lost` images sautées ou manquées à l'écran depuis le dernier appel.
  void missed(std::int64_t now_us, std::int64_t lost);
  // Seek, reprise, (re)chargement.
  void settle(std::int64_t now_us);
  // Repart de zéro (le mode vient d'être activé).
  void reset(std::int64_t now_us);

  // Le temps de service d'une image, avec marge et pression, en µs de temps réel ; au moins l'écart
  // minimal. 0 tant qu'aucune mesure n'existe.
  std::int64_t step_us(std::int64_t now_us) const;
  double pressure(std::int64_t now_us) const;
  bool should_fall_back(std::int64_t now_us) const;
  FreeInfo info(std::int64_t now_us) const;

 private:
  struct Bucket {
    std::int64_t index = -1;  // now_us / kBucketUs ; -1 : vide
    int produced = 0;
    int lost = 0;
  };
  Bucket& bucket(std::int64_t now_us);
  void sum(std::int64_t now_us, std::int64_t span_us, int& produced, int& lost) const;

  std::int64_t min_gap_us_ = 0;
  double service_us_ = 0.0;
  double pressure_ = 1.0;
  std::int64_t last_loss_us_ = 0;
  std::int64_t settled_until_us_ = 0;
  std::uint32_t lost_total_ = 0;
  std::array<Bucket, 24> buckets_{};
};

}  // namespace ajp::core
