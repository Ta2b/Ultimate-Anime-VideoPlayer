// « ajp bench » et le bouton « Optimiser pour ma machine » : le lancement des essais de auto_bench.h. Chaque
// essai est un processus enfant court, « ajp play » avec son coupé, un dossier de réglages jetable
// (--data-dir : ni play.toml ni user.toml ni journal de l'utilisateur n'y sont touchés) et le niveau forcé
// (--bench-level) ; il écrit ses mesures dans un fichier (--bench-out) que le parent relit. Un enfant qui
// plante ou qui dépasse son délai est un niveau qui ne tient pas : le parent ne tombe jamais avec lui, et dit
// pourquoi en une phrase (bench_steps.h : explain_trial) tout en journalisant le code de sortie et les
// dernières lignes de l'enfant.
#pragma once

#include <chrono>
#include <expected>
#include <filesystem>
#include <functional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/auto_bench.h"
#include "app/play/bench_steps.h"

namespace ajp::app::play {

inline constexpr int kBenchSeconds = 10;  // durée d'un essai (docs/ux-parametres.md : 10 à 15 s)
// La séquence du benchmark est en 1080p : l'exploration ne monte pas la hauteur de travail au-dessus.
inline constexpr int kExplorationSourceHeight = 1080;

struct BenchPlan {
  std::filesystem::path executable;  // ajp lui-même
  std::filesystem::path video;       // la courte séquence jouée à chaque niveau
  std::filesystem::path work_dir;    // où poser les dossiers jetables des essais
  std::filesystem::path
      data;  // les données de l'utilisateur : son settings.toml (moteurs choisis) passe aux essais, sans
             // quoi chaque essai rouvrirait l'assistant ; vide : « déjà vu » seulement
  int seconds = kBenchSeconds;
  bool fullscreen = true;  // l'écran réel, comme la lecture
  bool silent = true;      // fenêtre masquée autant que le bureau le permet (--hidden ; hide_is_reliable)
  // Le délai d'un essai s'ajoute à sa durée. Le premier essai charge et, la première fois, construit les
  // moteurs de la machine : sur un portable cela dépasse largement 3 minutes (un SPAN seul en prend 2,5 sur
  // une RTX 4070 Ti), et un essai tué en pleine construction ne garde rien : le suivant recommence au même
  // point. Il a donc un délai long ; les essais d'après, quand un niveau a déjà tenu, un délai court : un
  // niveau dont les moteurs sont longs à construire est dit « non testé », il ne gèle pas l'évaluation.
  std::chrono::seconds first_allowance{20 * 60};
  std::chrono::seconds build_allowance{4 * 60};
  std::chrono::seconds explore_allowance{25};  // l'exploration ne construit pas : trop long, elle s'arrête
  BenchProgress* shown = nullptr;              // l'avancement montré à l'utilisateur (étapes, barre)
};

// La ligne de commande de l'enfant d'un niveau (level absent : « auto », le niveau estimé de la machine).
// extra : des options de plus (l'exploration : --work-height, --rife-model).
std::vector<std::string> bench_child_args(const BenchPlan& plan, const std::optional<Tier>& level,
                                          const std::filesystem::path& data_dir,
                                          const std::filesystem::path& out_file,
                                          const std::vector<std::string>& extra = {});

// Le comparateur A/B (« ajp compare ») : deux variantes de la chaîne, jouées sur la même vidéo dans les mêmes
// conditions, et leurs mesures côte à côte. Une variante, ce sont des options de « ajp play » (--rife=3,
// --work-height=540…) et, pour ce qu'aucune option ne règle (les poses DRBA, le moteur d'interpolation), un
// play.toml de départ copié dans son dossier jetable.
struct Variant {
  std::string name;
  std::vector<std::string> args;
  std::filesystem::path settings;  // vide : les réglages par défaut du lecteur
};

std::vector<std::string> variant_child_args(const BenchPlan& plan, const Variant& variant,
                                            const std::filesystem::path& data_dir,
                                            const std::filesystem::path& out_file);

struct CompareOutcome {
  Variant a;
  Variant b;
  BenchReading reading_a;
  BenchReading reading_b;
};

// Joue A puis B (puis A et B une seconde fois dans l'ordre inverse : un PC qui chauffe ne favorise personne)
// ; la mesure gardée de chacun est la plus mauvaise des deux passages. Refus comme run_bench.
std::expected<CompareOutcome, Refusal> run_compare(
    const BenchPlan& plan, const Variant& a, const Variant& b, const std::stop_token& stop,
    const std::function<void(const std::string&)>& progress = {});

// Le rapport côte à côte : images perdues, retard p95, temps par image, composition, qui tient, et le plus
// léger.
std::string format_compare_report(const CompareOutcome& outcome);

struct BenchOutcome {
  std::vector<std::string>
      notes;                   // un essai qui n'a rien rendu ou n'a pas été fait : pourquoi, en une phrase
  std::vector<BenchRun> runs;  // dans l'ordre joué
  Tier estimated = Tier::Normal;
  BenchResult result;  // la carte, le pilote, le niveau gardé ; la date est à poser par l'appelant
};

// Joue les essais : l'estimé d'abord, puis next_level jusqu'à fin, puis l'exploration. progress reçoit la
// phrase de chaque étape (« Étape 2 sur 4 : … ») ; plan.shown reçoit l'avancement structuré. Refus :
// Cancelled sur demande d'arrêt, DecodeFailed si le premier essai ne rend rien (le détail dit pourquoi, avec
// la dernière ligne d'erreur de l'enfant).
std::expected<BenchOutcome, Refusal> run_bench(const BenchPlan& plan, const std::stop_token& stop,
                                               const std::function<void(const std::string&)>& progress = {});

// Le résultat de l'évaluation de bout en bout : les essais, le niveau gardé (daté, déjà écrit dans
// bench.toml) et le rapport technique, pour le journal et « ajp bench » : l'interface ne le montre pas.
struct Evaluation {
  BenchOutcome outcome;
  BenchResult result;
  std::string report;
};

// L'évaluation de bout en bout : run_bench, puis le niveau gardé dans data_dir/bench.toml (daté du jour). Le
// rapport a une ligne en plus si bench.toml n'a pas pu s'écrire.
std::expected<Evaluation, Refusal> evaluate_machine(
    const BenchPlan& plan, const std::filesystem::path& data_dir, const std::stop_token& stop,
    const std::function<void(const std::string&)>& progress = {});

// Le masquage de la fenêtre des essais (--hidden) est-il fiable sur ce bureau ? GNOME sous Wayland ne rend
// pas la main à une fenêtre minuscule sans bordure comme les autres bureaux : la mesure y serait faussée, la
// fenêtre des essais y reste donc visible (petite). Pur : les variables d'environnement sont données.
bool hide_is_reliable(std::string_view session_type, std::string_view current_desktop);
// La même, lue dans l'environnement du processus.
bool hide_is_reliable_here();

// Le rapport technique : une ligne par essai (niveau, images perdues, retard, tient ou non), le niveau gardé.
std::string format_bench_report(const BenchOutcome& outcome);

}  // namespace ajp::app::play
