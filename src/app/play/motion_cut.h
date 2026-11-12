// « Couper l'interpolation si la carte ne suit pas » (PlaySettings::auto_off). Pur : le lecteur donne des
// instants (µs, horloge hôte) et les compteurs du fil du GPU, jamais d'horloge lue ici.
//
// En mode Rapide, RIFE en retard laisse déjà l'image d'origine à l'écran : l'écart de service ne laisse à la
// paire que sa source (core/free_grid.h, paire affamée), et l'image interpolée revient dès que RIFE suit de
// nouveau. Ce que la lecture ne doit pas faire, c'est garder RIFE en route quand il ne produit plus rien : la
// carte travaille pour rien et SPAN en pâtit. MotionWatch le voit : quand RIFE ne produit plus au moins une
// image interpolée par paire de la source pendant kCutWindows secondes de suite, l'interpolation est à
// couper.
//
// - Le seuil suit la cadence de la source : une paire due par paire de sources qui bouge. Une coupe de scène
//   ou un dessin tenu ne s'interpolent pas à dessein ; ils ne sont pas dus (UpscaleStage::count_owed).
// - Les seeks, (re)chargements, pauses et fenêtres cachées ne comptent pas : ni retard ni rattrapage, et
//   kSettleUs de plus après eux, le temps que la file se remplisse et que les moteurs repartent.
// - Une seconde trop pauvre en paires dues (moins de kMinOwed, un plan presque fixe) ne compte pas non plus.
//
// La coupure vaut pour la chaîne ouverte seulement (MotionGuard, une par run_chain) : la vidéo suivante,
// « Réessayer » ou un autre choix d'interpolation dans les réglages la rendent. Rien n'est verrouillé.
#pragma once

#include <cstdint>
#include <optional>

#include "app/play/settings.h"

namespace ajp::app::play {

// Les compteurs du fil du GPU (LiveSettings::motion_tally), cumulés depuis l'ouverture de la chaîne.
struct MotionTally {
  std::uint64_t owed = 0;      // paires du Rapide qui devaient porter au moins une image interpolée
  std::uint64_t made = 0;      // images interpolées planifiées par ces paires
  std::uint64_t restarts = 0;  // seeks, sauts d'époque, (re)chargements vus par le fil du GPU
};

class MotionWatch {
 public:
  static constexpr std::int64_t kWindowUs = 1'000'000;
  static constexpr int kCutWindows = 10;  // environ 10 s de suite
  static constexpr std::int64_t kSettleUs = 2'000'000;
  static constexpr std::uint64_t kMinOwed = 3;

  // Un relevé, à chaque tour de la boucle qui relit l'écran. `counting` : le Rapide tourne, la lecture
  // avance et la fenêtre se voit. Vrai au relevé qui complète la dernière seconde en retard : l'interpolation
  // est à couper.
  bool note(std::int64_t now_us, const MotionTally& tally, bool counting) {
    const MotionTally before = seen_;
    seen_ = tally;
    if (!started_ || !counting || tally.restarts != before.restarts) {
      started_ = true;
      settle(now_us);
      return false;
    }
    if (now_us < settled_until_us_) {
      open_window(now_us);
      return false;
    }
    owed_ += tally.owed - before.owed;
    made_ += tally.made - before.made;
    if (now_us - window_us_ < kWindowUs) return false;
    if (owed_ >= kMinOwed) behind_ = made_ < owed_ ? behind_ + 1 : 0;  // trop peu de paires : sans verdict
    open_window(now_us);
    return behind_ >= kCutWindows;
  }
  // Les secondes de suite où RIFE n'a pas tenu une image interpolée par paire.
  int behind() const { return behind_; }
  void reset() { *this = MotionWatch{}; }

 private:
  void settle(std::int64_t now_us) {
    settled_until_us_ = now_us + kSettleUs;
    open_window(now_us);
  }
  void open_window(std::int64_t now_us) {
    window_us_ = now_us;
    owed_ = 0;
    made_ = 0;
  }

  bool started_ = false;
  MotionTally seen_;
  std::int64_t window_us_ = 0;
  std::int64_t settled_until_us_ = 0;
  std::uint64_t owed_ = 0;
  std::uint64_t made_ = 0;
  int behind_ = 0;
};

// Ce qui, dans les réglages, décide de l'interpolation en lecture : la coupure tient tant qu'il ne change
// pas.
struct InterpolationChoice {
  int rife_factor = 0;
  int rife_factor_den = 1;
  bool rife_auto = false;
  bool auto_free = false;
  bool auto_off = false;
  friend bool operator==(const InterpolationChoice&, const InterpolationChoice&) = default;
};

inline InterpolationChoice interpolation_choice(const PlaySettings& s) {
  return {.rife_factor = s.rife_factor,
          .rife_factor_den = s.rife_factor_den,
          .rife_auto = s.rife_auto,
          .auto_free = s.auto_free,
          .auto_off = s.auto_off};
}

struct Chain;

// La garde de run_chain (motion_cut.cpp) : MotionWatch sur les compteurs de la chaîne, la coupure et son
// retour.
class MotionGuard {
 public:
  // Vrai tant que l'interpolation est coupée ; un autre choix d'interpolation dans les réglages la rend
  // (le multiplicateur de la chaîne revient alors à celui des réglages). Appelé par apply_auto.
  bool held(Chain& chain);
  // Un tour de la boucle qui relit l'écran. `room_hz` : la cadence que l'écran (ou le plafond choisi)
  // permet ; sous deux fois la source, aucune image interpolée ne peut tenir à chaque paire et rien ne se
  // compte. Vrai quand l'interpolation vient d'être coupée : à run_chain de rappeler apply_auto.
  bool watch(Chain& chain, double room_hz, double source_fps, std::int64_t now_us);

 private:
  MotionWatch watch_;
  std::optional<InterpolationChoice> cut_;
};

}  // namespace ajp::app::play
