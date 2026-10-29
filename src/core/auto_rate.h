// Interpolation en mode auto : le multiplicateur ×k (de ×1 à ×6, par demi-paliers sur un écran VRR armé)
// pousse la fluidité au maximum que l'écran permet et que le GPU tient, et ne descend que le temps
// d'une scène difficile. Pur : le lecteur donne des instants (µs, horloge hôte) et des mesures, jamais
// d'horloge à lire ici, donc tout se teste sans GPU.
//
// - Prédire avant de sonder. Chaque paire terminée donne son coût de service (µs) pour n images
//   créées ; le modèle t(n) = a + b·n (a : mouvement, une fois par paire ; b : warp, SPAN, composition,
//   HDR, par image) se fait sur ces mesures (moyenne glissante, p90 de la scène en cours, poids qui
//   s'estompent avec le temps). Un palier n'est tenté que si t(palier) tient dans la période de la source
//   moins une marge de 15 %.
// - Descendre vite : deux paires en retard en 1 s, trois images manquées en ~1 s, ou un coût mesuré
//   au-dessus de la période : un palier de moins tout de suite, sans fenêtre de 4 s. Les paires faites
//   au palier d'avant ne comptent pas (aucune cascade).
// - Remonter vite et fin : dès que la prédiction le permet après 16 paires propres (~0,7 s), d'un palier.
// - Plafond appris par scène : un essai raté pose un plafond qui perd la moitié de sa force toutes les
//   6 s et que la coupe de scène divise ; il ne fait qu'élargir la marge, il ne bloque jamais.
// - Chien de garde : aucune paire depuis 2 s en lecture, le palier descend (jusqu'à ×1) ; aucune montée
//   avant que les paires reviennent.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

namespace ajp::core {

inline constexpr int kAutoMaxStep =
    8;  // ×8 : de quoi suivre un écran de 165 Hz jusqu'au bout (×6,88 pour 23,976)
// Granularité « au fps près » d'un écran VRR armé : un palier = 1/24 de multiplicateur (~1 i/s pour une
// source de 24 i/s). Le facteur passe par Factor{palier, 24} ; l'écran ne vibre pas, la cadence n'étant
// plus liée au rafraîchissement.
inline constexpr int kFineGranularity = 24;
// 1, 2 ou kFineGranularity : toute autre valeur y est ramenée.
int auto_granularity(int wanted);

// Le plus grand palier dont la cadence tient sous la fréquence de l'écran (à 1 % près) pour une source
// de `source_fps` i/s ; 1 si l'écran est inconnu. Sans VRR, les paliers qui ne tiennent pas un nombre
// entier de rafraîchissements font du judder : le menu l'avertit, il ne les interdit pas.
// `granularity` : kFineGranularity, un palier par 1/24 de multiplicateur (VRR armé) ; 1, des paliers entiers
// ; 2, des demi-paliers (×1,5, ×2,5…), que seul un écran VRR armé affiche sans vibrer (la cadence n'est plus
// liée au rafraîchissement). Les paliers se comptent alors en moitiés : 5 est ×2,5 ; le facteur est palier /
// granularity.
int auto_ceiling(double display_hz, double source_fps, int granularity = 1);

// Images manquées à l'écran sur une fenêtre (missing) pour shown images neuves : le GPU ne suit pas
// même quand aucune paire n'est « en retard » (mesuré sur 4070 Ti : DistilDRBA ×6 a 707 manquées pour
// 1 retard, le coût par image créée vient de SPAN et de RTX HDR). Au moins 6 manquées et plus de 1,2 %
// des images attendues. Garde-fou lent ; la descente rapide passe par AutoRate::missed.
inline bool auto_overloaded(std::int64_t missing, std::int64_t shown) {
  return missing >= 6 && missing * 1000 > (missing + shown) * 12;
}

// Le coût de SERVICE d'une paire : de l'instant où le GPU a pu la commencer (le plus tardif de sa première
// soumission et de la fin de la paire d'avant : derrière elle, la paire attend sa file, ce qui n'est pas son
// coût) à sa fin vue, plus l'attente de ses mesures avant soumission. Sans cela, un palier trop haut qui
// remplit la file gonfle le coût mesuré et fait descendre trop bas (mesuré sur 4070 Ti : ×1 pour un GPU à
// 45 % quelques secondes plus tard).
inline std::int64_t service_cost_us(std::int64_t first_submitted_us, std::int64_t previous_done_us,
                                    std::int64_t done_us, std::int64_t lead_us) {
  const std::int64_t start = std::max(first_submitted_us, previous_done_us);
  return std::max<std::int64_t>(done_us - start, 0) + lead_us;
}

// Pourquoi le palier a bougé en dernier.
enum class AutoReason : std::uint8_t {
  Start,      // départ
  Memory,     // départ au dernier palier tenu pour cette vidéo
  Rise,       // la prédiction tient avec marge : un palier de plus
  Late,       // paires en retard sur l'horloge
  Missed,     // images manquées à l'écran
  Predicted,  // le coût mesuré dépasse la période de la source
  Watchdog,   // plus aucune paire
  Ceiling     // le plafond (écran, réglage) a bougé
};
std::string_view auto_reason_text(AutoReason reason);

// Une paire de l'interpolation, au moment où elle est préparée.
struct PairOutcome {
  std::int64_t now_us = 0;
  int made_step = 0;           // le palier avec lequel elle a été faite (en 1 / granularity)
  int late = 0;                // instants sautés pour rattraper l'horloge
  std::int64_t period_us = 0;  // période de la source à la vitesse de lecture (0 : inconnue)
};

struct AutoInfo {
  int step = 1;
  int granularity = 1;
  int ceiling = 1;
  int predicted = 0;   // le palier que la prédiction tiendrait ; 0 : pas assez de mesures
  int load_pct = -1;   // coût mesuré du palier courant en % de la période ; -1 : inconnu
  int cap = 0;         // plafond appris encore vif ; 0 : aucun
  bool heavy = false;  // la scène charge le GPU (coût ≥ 85 % de la période)
  AutoReason reason = AutoReason::Start;
  std::uint32_t changes = 0;
};

class AutoRate {
 public:
  static constexpr std::int64_t kHoldUs = 700'000;          // après un changement : mesures ignorées
  static constexpr std::int64_t kSettleUs = 1'000'000;      // après seek, reprise, chargement
  static constexpr std::int64_t kLateWindowUs = 1'000'000;  // deux retards dans cette fenêtre : descente
  static constexpr int kLatePairs = 2;
  static constexpr std::int64_t kMissWindowUs = 1'200'000;  // trois manquées dans cette fenêtre : descente
  static constexpr int kMissLimit = 3;
  static constexpr std::int64_t kWatchdogUs = 2'000'000;
  static constexpr std::int64_t kMinRiseGapUs =
      1'000'000;                                          // jamais plus d'un changement par seconde en montée
  static constexpr std::uint32_t kRisePairs = 16;         // paires propres avant de monter
  static constexpr std::uint32_t kMinPairs = 12;          // paires livrées depuis le (re)chargement
  static constexpr std::uint32_t kBlindRisePairs = 72;    // sans mesure de coût : montée comme avant
  static constexpr double kMargin = 0.15;                 // marge de la prédiction
  static constexpr double kCapMargin = 0.30;              // marge au-dessus d'un plafond appris vif
  static constexpr std::int64_t kHalfLifeUs = 6'000'000;  // poids, plafond et pénalités s'estompent
  static constexpr std::int64_t kProbeUs = 2'000'000;     // un palier tenté est « tenu » après ce temps
  static constexpr int kMaxImages = kAutoMaxStep + 1;
  static constexpr double kDeadBand = 0.06;  // au pas fin : pas de montée sous 6 % d'écart
  // Au pas fin, en régime (écart modeste au palier prédit) un changement toutes les 5 s au plus : le temps de
  // trame d'un écran VRR OLED reste constant entre deux changements. Loin sous le palier prédit (scène
  // allégée), la remontée est rapide : un saut jusqu'à 40 % du palier, une seconde d'écart.
  static constexpr std::int64_t kSteadyGapUs = 5'000'000;
  static constexpr double kFarRatio = 0.5;  // l'écart au palier prédit qui compte pour « loin »

  // `ceiling` : le palier le plus haut permis (granularity à kAutoMaxStep × granularity). Le départ est ×2
  // (ou le plafond). Les paliers se comptent en 1 / granularity (1 ou 2, voir auto_ceiling).
  explicit AutoRate(int ceiling = kAutoMaxStep, int granularity = 1);

  int step() const { return step_; }
  int granularity() const { return granularity_; }
  int learned_cap(std::int64_t now_us) const;  // 0 : aucun (ou plus assez vif pour compter)
  int ceiling() const { return ceiling_; }
  AutoReason reason() const { return reason_; }
  std::uint32_t changes() const { return changes_; }
  // Plafond changé (écran déplacé, réglage) : le palier courant le suit.
  void set_ceiling(int ceiling);
  // Plafond et granularité changés ensemble (le VRR s'arme ou se coupe) : le palier courant, le plafond
  // appris et les pénalités passent dans la nouvelle unité, sans dépasser le plafond.
  void set_ceiling(int ceiling, int granularity);

  // Départ à un palier retenu (en centièmes de multiplicateur) : un cran en dessous, la montée rapide fait
  // le reste.
  void seed(int remembered_centi, std::int64_t now_us);
  // Le palier courant en centièmes de multiplicateur (250 = ×2,5), indépendant de la granularité.
  int centi() const { return step_ * 100 / granularity_; }

  // Une paire préparée. Rend le palier à utiliser pour la suivante.
  int pair(const PairOutcome& outcome);
  // Une paire dont le palier n'a rien à dire (coupe de scène, dessin tenu) : le moteur vit.
  void alive(std::int64_t now_us) { beat_us_ = now_us; }
  // Une paire terminée sur le GPU : `images` créées (source comprise), `cost_us` de la soumission de sa
  // première image à la fin de sa dernière, plus l'attente de ses mesures.
  void cost(int images, std::int64_t cost_us, std::int64_t now_us);
  // Images manquées à l'écran depuis le dernier appel (delta ≥ 0).
  int missed(std::int64_t now_us, std::int64_t delta);
  // Le pas de l'horloge de main : chien de garde ; playing faux (pause, fenêtre cachée) l'endort.
  int tick(std::int64_t now_us, bool playing);
  // Coupe de scène détectée : la scène d'avant ne dit plus rien du coût d'après.
  void scene_cut(std::int64_t now_us);
  // Seek, pause, reprise, (re)chargement : la mesure repart.
  void settle(std::int64_t now_us);

  AutoInfo info(std::int64_t now_us) const;
  // Le palier que la prédiction tient (0 : pas assez de mesures).
  int predicted_step(std::int64_t now_us) const;

 private:
  struct ImageCost {
    double ewma_us = 0.0;
    double weight = 0.0;
    std::int64_t at_us = 0;
    std::array<float, 16> ring{};
    std::size_t filled = 0;
    std::size_t next = 0;
  };
  struct Line {
    double a = 0.0;
    double b = 0.0;
  };

  void descend(AutoReason reason, std::int64_t now_us, int steps = 1);
  int drop_for(int predicted) const;  // de combien descendre
  bool fine() const { return granularity_ >= kFineGranularity; }
  void changed(AutoReason reason, std::int64_t now_us);
  void try_rise(std::int64_t now_us);
  double weight_of(const ImageCost& c, std::int64_t now_us) const;
  std::optional<Line> fit(std::int64_t now_us) const;
  double spike(std::int64_t now_us) const;
  double model_us(int step, std::int64_t now_us) const;  // < 0 : pas de modèle
  double penalty(int step, std::int64_t now_us) const;
  double cap_strength(std::int64_t now_us) const;
  int late_in_window(std::int64_t now_us) const;
  int missed_in_window(std::int64_t now_us) const;

  int granularity_;
  int ceiling_;
  int step_;
  int hint_ = 0;  // dernier palier tenu, retenu d'une lecture d'avant
  AutoReason reason_ = AutoReason::Start;
  std::uint32_t changes_ = 0;
  std::int64_t last_change_us_ = -kMinRiseGapUs;
  std::int64_t hold_until_us_ = 0;
  std::int64_t beat_us_ = 0;
  std::int64_t period_us_ = 0;
  std::int64_t probe_until_us_ = 0;
  bool probing_ = false;
  std::uint32_t clean_ = 0;
  std::uint32_t pairs_ = 0;  // paires livrées depuis le dernier settle
  std::array<std::int64_t, 4> late_at_{};
  std::size_t late_next_ = 0;
  std::array<std::pair<std::int64_t, int>, 6> miss_{};
  std::size_t miss_next_ = 0;
  int cap_ = 0;
  std::int64_t cap_at_us_ = 0;
  struct Fail {
    double extra = 0.0;
    std::int64_t at_us = 0;
  };
  std::array<Fail, kAutoMaxStep * kFineGranularity + 2> fails_{};
  std::array<ImageCost, kMaxImages + 1> costs_{};
};

}  // namespace ajp::core
