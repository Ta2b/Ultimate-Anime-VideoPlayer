// Mode « Rapide » (Auto libre) : le temps de service mesuré par image, la contre-pression et le repli. Pur :
// le lecteur donne des instants (µs, horloge hôte) et des mesures, jamais d'horloge à lire ici.
//
// Aucun palier, aucun multiplicateur, aucune cadence visée : le producteur sort une image dès que le GPU
// peut la livrer à l'instant où elle sera vue (core/free_grid.h). Cette classe ne fixe que l'écart
// minimal entre deux images consécutives d'une paire, `step_us` : le temps de service mesuré d'une image
// × 1,10, élargi par la contre-pression quand des images sont perdues à l'écran.
//
// - Temps de service : la durée GPU propre à une image créée (jamais l'attente d'un slot du ring), moyenne
//   qui monte vite (une scène lourde) et descend lentement (elle ne s'emballe pas sur un coup de chance).
//   Une paire réduite à une image interpolée se mesure aussi ; une paire que l'écart a réduite à sa seule
//   source ne mesure rien et fait redescendre l'estimation (`starved`), sinon un à-coup du GPU la figerait
//   au-dessus de l'écart des sources et plus aucune paire ne la corrigerait (A12-12).
// - Contre-pression : une image sautée ou manquée à l'écran élargit l'écart tout de suite (× 1,12 par
//   image perdue, au plus × 2) ; elle ne se relâche qu'après 1 s sans perte, puis lentement (3 s).
// - Repli : 5 % d'images perdues ou plus sur 5 s (et au moins 10) rendent la main au mode Auto.
// - Mouvement fort anticipé : une paire en mouvement fort coûte bien plus par image (RIFE à chaque instant au
//   lieu du milieu et des poses). Son temps de service se mesure à part, et le lecteur, qui sait une paire à
//   l'avance qu'elle sera lourde, planifie son écart sur ce temps-là, avant que la chaîne ne décroche. Sans
//   mesure lourde encore, l'écart est celui d'une paire calme × kHeavyGuess.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

namespace ajp::core {

struct FreeInfo {
  double fps = 0.0;                   // images créées par seconde (fenêtre de 2 s) ; 0 : pas encore mesuré
  int load_pct = -1;                  // temps de service × cadence en % d'une seconde de GPU ; -1 : inconnu
  std::int64_t step_us = 0;           // l'écart de service planifié ; 0 : inconnu
  std::int64_t service_us = 0;        // le temps de service d'une image mesuré ; 0 : inconnu
  std::int64_t heavy_service_us = 0;  // le même, en mouvement fort ; 0 : pas encore mesuré
  double pressure = 1.0;              // contre-pression (1 : aucune)
  std::uint32_t lost = 0;             // images perdues depuis le début
};

class FreeRate {
 public:
  static constexpr double kMargin = 1.10;
  static constexpr double kPressureStep = 1.12;
  static constexpr double kPressureMax = 2.0;
  static constexpr std::int64_t kCalmUs = 1'000'000;     // sans perte avant que la pression se relâche
  static constexpr std::int64_t kReleaseUs = 3'000'000;  // constante de temps du relâchement
  static constexpr std::int64_t kFallbackWindowUs = 5'000'000;
  // La cadence affichée (FreeInfo::fps) : les images des 2 dernières secondes, assez court pour suivre un
  // changement de plan, assez long pour ne pas sautiller.
  static constexpr std::int64_t kInfoWindowUs = 2'000'000;
  static constexpr std::int64_t kSettleUs = 1'200'000;  // après seek ou reprise : mesures ignorées
  static constexpr std::int64_t kStartUs =
      5'000'000;  // après l'activation du mode (démarrage, moteurs qui chauffent)
  static constexpr std::int64_t kMaxSampleUs = 300'000;
  static constexpr std::int64_t kBucketUs = 250'000;
  static constexpr int kFallbackMinLost = 10;
  static constexpr int kFallbackPerMille = 50;  // 5 %
  // Des pertes durables seulement : réparties sur au moins 4 tranches de 250 ms de la fenêtre (une seconde),
  // pas un à-coup isolé.
  static constexpr int kFallbackMinBuckets = 4;
  static constexpr double kHeavyGuess = 2.0;  // paire lourde jamais mesurée : coût d'une calme × 2

  // L'écart minimal entre deux images (période de l'écran ou du plafond), en µs de temps réel.
  void set_min_gap(std::int64_t us) { min_gap_us_ = us; }
  std::int64_t min_gap_us() const { return min_gap_us_; }

  // La part de l'estimation perdue par paire affamée : la pente d'une scène plus légère, vers zéro.
  static constexpr double kStarvedDecay = 0.12;

  // Une paire de `images` images calculées a occupé le GPU `busy_us`. Une image seule ne compte que si elle
  // est interpolée : une source seule (coupe, dessin tenu) ne dit rien du coût d'une image interpolée, à
  // l'appelant de ne pas la donner. `heavy` : une paire en mouvement fort, mesurée à part.
  void cost(int images, std::int64_t busy_us, std::int64_t now_us, bool heavy = false);
  // Une paire interpolée où l'écart n'a laissé que la source : rien n'est mesuré, donc rien ne confirme
  // l'estimation. Elle redescend de kStarvedDecay, jusqu'à ce qu'une paire porte de nouveau une image
  // interpolée et la mesure. `heavy` : la paire était planifiée sur le temps du mouvement fort.
  void starved(std::int64_t now_us, bool heavy = false);
  // `images` images planifiées par la paire qui vient d'être préparée.
  void produced(int images, std::int64_t now_us);
  // `lost` images sautées ou manquées à l'écran depuis le dernier appel.
  void missed(std::int64_t now_us, std::int64_t lost);
  // Seek, reprise, (re)chargement : les mesures du coût s'arrêtent kSettleUs ; les pertes ne comptent pas
  // pour le repli pendant kStartUs, comme au démarrage (la file se remplit, les moteurs repartent).
  void settle(std::int64_t now_us);
  // Repart de zéro (le mode vient d'être activé).
  void reset(std::int64_t now_us);

  // Le temps de service d'une image, avec marge et pression, en µs de temps réel ; au moins l'écart
  // minimal. 0 tant qu'aucune mesure n'existe. `heavy` : pour une paire annoncée en mouvement fort.
  std::int64_t step_us(std::int64_t now_us, bool heavy = false) const;
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
  double heavy_us_ = 0.0;  // temps de service d'une image en mouvement fort ; 0 : jamais mesuré
  double pressure_ = 1.0;
  std::int64_t last_loss_us_ = 0;
  std::int64_t settled_until_us_ = 0;
  std::int64_t quiet_until_us_ = 0;  // avant : les pertes ne comptent pas pour le repli
  std::uint32_t lost_total_ = 0;
  std::array<Bucket, 24> buckets_{};
};

// Le repli du Rapide vers Stable, pour la séance : il revient au Rapide après un calme, plus long à chaque
// repli (3 min, 6, 12… au plus 30 min, P3.38) ; un choix explicite du mode l'efface.
class FreeFallback {
 public:
  static constexpr std::int64_t kRetryUs = 180'000'000;
  static constexpr std::int64_t kRetryMaxUs = 1'800'000'000;

  // Vrai au premier repli seulement : le message ne se répète pas sur une carte limite (P3.38).
  bool fell_back(std::int64_t now_us) {
    active_ = true;
    retry_at_us_ = now_us + delay_us_;
    delay_us_ = std::min(delay_us_ * 2, kRetryMaxUs);
    return !std::exchange(told_, true);
  }
  // Vrai tant que le repli tient ; faux dès que l'heure de réessayer le Rapide est venue.
  bool active(std::int64_t now_us) {
    if (active_ && now_us >= retry_at_us_) active_ = false;
    return active_;
  }
  void clear() { *this = FreeFallback{}; }

 private:
  bool active_ = false;
  bool told_ = false;
  std::int64_t retry_at_us_ = 0;
  std::int64_t delay_us_ = kRetryUs;
};

}  // namespace ajp::core
