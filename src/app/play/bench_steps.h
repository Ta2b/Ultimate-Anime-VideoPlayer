// « ajp bench » (docs/ux-parametres.md, 4.7) : ce que l'utilisateur voit. Trois étapes lisibles, une barre
// de progression globale et des échecs d'essai toujours expliqués en une phrase. Rien ici ne joue ni ne
// touche l'interface : ce sont des fonctions pures et deux petits objets, que l'évaluation
// (auto_bench_run.h) et son affichage se partagent.
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <expected>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "app/play/auto_bench.h"
#include "app/play/settings.h"
#include "core/types.h"
#include "prepare/process.h"

namespace ajp::app::play {

// Les étapes, dans l'ordre. L'utilisateur lit « Étape 2 sur 3 : … », jamais « niveau » ni « tranche ».
enum class BenchStep : int {
  Prepare = 1,  // le premier essai : charge (et au besoin construit) les moteurs, estime la machine
  Levels,       // les réglages recommandés, du plus léger au plus lourd qui tient
  Explore,      // la résolution et le modèle d'interpolation au-dessus, si la machine a de la marge
};
inline constexpr int kBenchSteps = 3;

// « Étape 2 sur 3 : test des réglages « Rapide » ».
std::string step_line(BenchStep step, std::string_view what);
// La barre globale : le poids de chaque étape, puis la part faite de l'étape en cours (0 à 1).
float overall_fraction(BenchStep step, float within);

// L'avancement, écrit par le fil de l'évaluation et lu par l'interface à chaque image.
class BenchProgress {
 public:
  struct View {
    BenchStep step = BenchStep::Prepare;
    std::string what;       // « test des réglages « Rapide » »
    float fraction = 0.0F;  // globale, 0 à 1
    std::string note;  // un mot de plus (« première fois : la construction peut durer plusieurs minutes »)
  };
  void set(BenchStep step, std::string what, float within = 0.0F, std::string note = {});
  void set_note(std::string note);
  View get() const;

 private:
  mutable std::mutex mutex_;
  BenchStep step_ = BenchStep::Prepare;
  std::string what_;
  float within_ = 0.0F;
  std::string note_;
};

// Les dernières lignes qu'un enfant a écrites, pour dire pourquoi son essai n'a rien rendu et pour le
// journal.
class ChildTail {
 public:
  explicit ChildTail(std::size_t keep = 12) : keep_(keep) {}
  void add(std::string_view line);
  // La dernière ligne qui ressemble à une erreur (erreur, refus, échec, failed, error), sinon la dernière
  // ligne ; sans l'horodatage ni le canal du journal, coupée à 200 caractères. Vide sans ligne.
  std::string problem() const;
  std::string text() const;  // les lignes gardées, une par ligne (le journal)

 private:
  std::size_t keep_;
  std::deque<std::string> lines_;
};

// « [2026-10-29 12:53:57.048] [app] [error] refus : x » devient « refus : x ».
std::string strip_log_prefix(std::string_view line);

// Pourquoi un essai n'a rien rendu, en une phrase : « L'essai des réglages « Rapide » n'a rien rendu : … ».
// what : « des réglages « Rapide » » ; problem : la dernière ligne d'erreur de l'enfant (peut être vide) ;
// limit : le délai qui lui était donné. Cancelled et une fin normale n'ont pas de phrase (vide).
std::string explain_trial(std::string_view what, const prepare::ProcessExit& exit, std::string_view problem,
                          std::chrono::seconds limit);
// Fin normale sans fichier de mesures lisible.
std::string explain_no_reading(std::string_view what, std::string_view detail);

}  // namespace ajp::app::play
