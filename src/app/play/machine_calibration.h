// La calibration de la machine (l'assistant, puis Vidéo > Post-traitement > « Résultats du bench ») : à la
// place des préréglages, le lecteur essaie des chaînes sur un même extrait très dur pour l'interpolation
// (media/bench_clip.h) et recommande la meilleure que la machine tient. L'utilisateur ne voit qu'une barre de
// chargement.
//
// Les hauteurs de travail essayées montent jusqu'à celle de la séquence (1080p), sur tout écran : une source
// 1080p que la carte tient n'est plus réduite à la moitié de l'écran (A49-02). Deux familles, dans l'ordre
// (choix jugé à l'œil) :
//   1. SPAN seul, depuis le palier du niveau estimé : en montant jusqu'au premier qui ne tient pas, ou en
//      descendant jusqu'au premier qui tient (B07-14) ;
//   2. SPAN + RIFE en mode auto Instantané, du plus haut palier que SPAN seul a tenu vers le bas, jusqu'au
//      premier qui tient : RIFE ne fait qu'ajouter du travail, un palier que SPAN seul ne tient pas ne se
//      tient pas avec RIFE ;
//   3. si SPAN + RIFE tient au plus haut palier avec des modèles plus légers que ceux de Qualité, ce palier
//      rejoué avec ceux de Qualité (C07-03) : une carte que la table sous-estime monte de modèle.
// Les modèles sont ceux du niveau estimé de la carte (auto_config.h : SPAN Balanced Sharp b8f64 et RIFE 4.26
// en Qualité et Ultra, SPAN Performance Sharp et RIFE 4.22 lite en dessous), SPAN toujours en FP16 : le FP8
// n'est plus essayé tant que sa porte n'est pas rejugée (A49-03, ADR 0007 suspendue). Sans palier SPAN qui
// tienne, RIFE seul (SPAN coupé) est le recours. DistilDRBA n'est pas essayé (décision 8 des audits). Ici la
// recherche, pure, et le fichier des résultats ; les essais (des « ajp play » enfants, auto_bench_run.h) dans
// le .cpp.
#pragma once

#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/auto_bench_run.h"
#include "app/play/auto_config.h"
#include "app/play/settings.h"

namespace ajp::app::play {

// La version de calibration.toml. 2 : hauteurs jusqu'à 1080p sur tout écran, modèles du niveau en FP16. Un
// fichier plus ancien (essais bornés à la moitié de l'écran, SPAN FP8, RIFE 4.22 lite) se relit pour le
// tableau mais l'Auto ne le suit plus : il borne la chaîne plus bas que la carte ne tient.
// 3 : le moteur et le pilote de la mesure sont notés, et un essai avec RIFE ne tient que si RIFE double
// vraiment la cadence (A07-02) : une version 2 a pu retenir une chaîne où le Rapide avait coupé RIFE.
// 4 : une chaîne ne tient qu'avec au plus kMaxLossPerImage de pertes par image visée (B07-02) ; une version
// 3 a pu retenir une chaîne qui perd une image par seconde.
inline constexpr int kCalibrationVersion = 4;
// Un essai avec RIFE tient si les images neuves obtenues, créditées des instants laissés sans image à dessein
// (coupes et dessins tenus de la séquence), font au moins cette part de la cadence doublée. En dessous, le
// mode Rapide a baissé le facteur faute de temps : aucune image n'est perdue, mais RIFE ne tourne pas.
inline constexpr double kMinRifeShare = 0.95;
// Les pertes qu'une chaîne recommandée peut avoir, par image visée (target_fps), pas par présentation : à
// 24 i/s sur un écran à 60 Hz, les 2 % par présentation de holds (auto_bench.h) font 5 % des images, une
// saccade par seconde. 0,5 % : au plus une image sur 200, une par dix secondes de séquence à 24 i/s.
inline constexpr double kMaxLossPerImage = 0.005;

// Les modèles joués par les essais ; vides : ceux par défaut (SPAN Performance Sharp, RIFE 4.22 lite).
struct CalibrationModels {
  std::string span;
  std::string rife;
  bool operator==(const CalibrationModels&) const = default;
};
// Ceux du niveau estimé de la carte (level_span_model, level_rife_model).
CalibrationModels calibration_models(Tier tier);
// Ceux de Qualité, que l'essai de la famille 3 joue (ChainCandidate::finer).
CalibrationModels finer_models();

struct ChainCandidate {
  bool span = true;
  bool rife = true;
  int height = 0;      // la hauteur de travail de SPAN ; sans SPAN, 0 (la source telle quelle)
  bool finer = false;  // joué avec finer_models() au lieu des modèles du niveau (famille 3, C07-03)
  bool operator==(const ChainCandidate&) const = default;
};

struct CandidateTrial {
  ChainCandidate chain;
  bool holds = false;
  double lost_percent = -1.0;  // images perdues ou manquées, en % des présentations ; < 0 : rien rendu
  double fps = 0.0;            // images neuves par seconde obtenues, en moyenne
  double low_fps = 0.0;        // le « 1 % low » : d'après les 1 % d'écarts les plus longs
  double target_fps = 0.0;     // celles visées : la séquence (24 i/s), doublée par RIFE
  // Images perdues ou manquées par image visée, l'échelle du verdict (kMaxLossPerImage) ; < 0 : inconnu (un
  // fichier plus ancien), needs_second_run reprend alors lost_percent.
  double lost_per_image = -1.0;
};

// Les images neuves par seconde d'une lecture, créditées de ses instants sans image à dessein ; 0 :
// rien de mesuré. Une lecture d'un lecteur plus ancien (sans décompte) rend la cadence brute.
double credited_fps(const BenchReading& reading);
// La cadence visée par un essai : celle de la séquence (24 i/s), doublée par RIFE sans dépasser celle de
// l'écran (period_us, 0 : inconnue), que le mode Rapide ne dépasse pas non plus.
double trial_target_fps(const ChainCandidate& chain, std::int64_t period_us);
// Le verdict d'un essai : pertes et retard sous leurs seuils (holds, auto_bench.h), pertes sous
// kMaxLossPerImage des images visées, et avec RIFE la cadence créditée à kMinRifeShare de target_fps au
// moins.
bool trial_holds(const ChainCandidate& chain, const BenchReading& reading, double target_fps);

// Les hauteurs essayées, de la plus lourde à la plus légère : toutes jusqu'à celle de la séquence.
std::vector<int> calibration_heights();

// Le prochain essai après ceux déjà faits ; rien quand la recherche est finie.
// La famille SPAN seul part de start_height (le palier au plus égal, le plus bas si 0), monte tant qu'elle
// tient ou descend tant qu'elle ne tient pas : une carte rapide ne construit pas tous les paliers d'en bas.
// try_finer : les modèles joués sont plus légers que finer_models(), la famille 3 peut les monter.
std::optional<ChainCandidate> next_candidate(const std::vector<CandidateTrial>& tried, int start_height = 0,
                                             bool try_finer = false);
// Le palier où la calibration commence pour ce niveau estimé : celui que l'Auto estimé y joue.
int calibration_start_height(Tier tier);
// La chaîne recommandée, la fluidité d'abord (P3.11) : le plus haut palier où SPAN et RIFE tiennent
// ensemble, avec les modèles de Qualité s'ils y tiennent aussi ; sinon le plus haut SPAN seul ; RIFE seul si
// SPAN ne tient nulle part ; ni SPAN ni RIFE si rien ne tient. Les autres paliers restent au tableau, au
// choix de l'utilisateur.
ChainCandidate best_candidate(const std::vector<CandidateTrial>& tried);
// Le plus haut palier où SPAN seul a tenu ; 0 : aucun (CalibratedChain::span_height).
int best_span_alone(const std::vector<CandidateTrial>& tried);
// Les réglages d'une chaîne, sur ceux de l'utilisateur (le HDR choisi, les langues… restent) : préréglage
// Personnalisé, et pour le reste ce que l'Auto calibré écrira (apply_preset, A07-01) : réduction Hermite,
// SPAN et RIFE de models en FP16, RIFE en mode auto Instantané, ordre de SPAN en Auto, poses gardées,
// « Restaurer les traits » actif (sans lui un essai SPAN sous 1080p jouerait la source sans SPAN, A07-06).
PlaySettings candidate_settings(const ChainCandidate& chain, PlaySettings base,
                                const CalibrationModels& models = {});
// « 720p · SPAN · RIFE (auto Instant) » ; pour le tableau et le journal.
std::string describe(const ChainCandidate& chain);

struct MachineCalibration;
// Ce que la calibration a retenu, dit à l'utilisateur à sa fin : la chaîne, la cadence mesurée, ce qui a été
// abaissé contre la vitesse, et la perte d'images du réglage juste au-dessus qui ne tenait pas. Le gain est
// celui de ce banc seulement : rien d'autre n'est annoncé.
std::string calibration_notice(const MachineCalibration& calibration);

struct MachineCalibration {
  ChainCandidate chosen;
  std::vector<CandidateTrial> tried;  // dans l'ordre joué
  std::string gpu;
  std::string date;  // AAAA-MM-JJ
  int display_height = 0;
  int version = kCalibrationVersion;  // celle du fichier relu ; sous kCalibrationVersion : pas suivie
  // Les modèles de la chaîne recommandée : ceux du niveau, ou finer_models() quand chosen.finer. Les essais
  // de la famille 3 (finer) ont joué finer_models(), les autres ceux du niveau.
  CalibrationModels models;
  std::string engine;       // le moteur qui a joué les essais (« tensorrt », « vulkan »…) ; vide : inconnu
  std::string driver;       // le pilote de la carte pendant la mesure ; vide : inconnu
  bool on_battery = false;  // mesurée sur batterie (P3.12) : gardée au tableau, pas suivie par l'Auto
  double display_hz = 0.0;  // la fréquence de l'écran pendant la mesure (C37-08) ; 0 : inconnue
};

// La barre de la calibration (P3.13) : l'avancement dit par calibrate_machine, plus une avance lente pendant
// un essai qui dure (une construction de moteur), au plus 8 points : elle ne reste plus figée des minutes.
float creeping_fraction(float reported, double seconds_since_change);
// La durée annoncée avant la calibration (P3.13), en minutes : les essais, rejeux compris, et les
// constructions de TensorRT (une par taille essayée) quand ses moteurs ne sont pas encore construits, plus
// longues sur un portable.
int calibration_minutes(bool builds_engines, bool laptop);

// Un essai se rejoue une fois (P3.12) : quand il ne tient pas, ou qu'il tient de justesse (plus de la
// moitié des pertes permises par image visée, ou une cadence sous 97 % de la visée).
bool needs_second_run(const CandidateTrial& trial);

// Le fichier des résultats, à côté de play.toml, que calibrate_machine écrit quand plan.data est donné : un
// par hauteur d'écran (calibration-1440p.toml, A37-10), pour qu'un écran 1440p et une télé 4K gardent
// chacun le leur. display_height ≤ 0 : calibration.toml, l'ancien fichier unique.
std::filesystem::path calibration_file(const std::filesystem::path& data_dir, int display_height = 0);
// La calibration de cet écran : son fichier, sinon l'ancien calibration.toml (relu tel quel ; calibration_use
// dit s'il est d'une autre hauteur).
std::expected<MachineCalibration, Refusal> load_calibration_at(const std::filesystem::path& data_dir,
                                                               int display_height);
// Le dossier jetable des essais (BenchPlan::work_dir) : propre à ce lancement, sous le dossier de données de
// l'utilisateur (A25-11 : un nom fixe dans /tmp appartenait au premier compte qui calibrait, et un dossier
// temporaire introuvable donnait un chemin relatif au dossier courant). data_dir vide : chemin vide, et
// calibrate_machine refuse.
std::filesystem::path calibration_work_dir(const std::filesystem::path& data_dir);
// Le clip de l'évaluation, dans le dossier des données : une version de plus change ce nom, et
// purge_bench_leftovers retire les clips des versions d'avant (B36-09).
inline constexpr std::string_view kBenchClipName = "bench-clip-v4.mkv";
std::string format_calibration(const MachineCalibration& calibration);
std::expected<MachineCalibration, Refusal> parse_calibration(std::string_view text);
std::expected<MachineCalibration, Refusal> load_calibration(const std::filesystem::path& file);
// Ce que l'Auto fait de cette calibration sur cette machine : suivie si elle est de cette version, mesurée
// sur secteur, sur cette carte, à cette hauteur d'écran et par le moteur en service.
CalibrationUse calibration_use(const MachineCalibration& calibration, const CalibrationHost& host);
// Suivie, mais la branche du pilote a changé depuis la mesure : le tableau propose de refaire le banc
// (A36-02), même règle que bench.toml (driver_branch_changed, A37-09).
bool calibration_driver_changed(const MachineCalibration& calibration, const CalibrationHost& host);
// La chaîne de calibration.toml (dossier data_dir) si l'Auto la suit (calibration_use) : les bornes de l'Auto
// calibré. std::nullopt sinon.
std::optional<CalibratedChain> calibrated_chain(const std::filesystem::path& data_dir,
                                                const CalibrationHost& host);
// La même, d'une calibration déjà lue (celle que le panneau vient de refaire).
std::optional<CalibratedChain> followed_chain(const MachineCalibration& calibration,
                                              const CalibrationHost& host);
std::expected<void, Refusal> save_calibration(const std::filesystem::path& file,
                                              const MachineCalibration& calibration);

// Un essai, tel que la recherche le demande : la chaîne, son rang (son dossier jetable) et s'il a droit au
// long délai d'une première construction de moteur. calibrate_machine lance un « ajp play » enfant ; les
// tests en donnent un faux (A07-11).
using TrialPlayer =
    std::function<std::expected<BenchReading, Refusal>(const ChainCandidate& chain, int index, bool first)>;
// Ce que la recherche rend : les essais gardés, dans l'ordre joué, et la carte, le pilote et le moteur que
// les lectures ont dits.
struct CalibrationTrials {
  std::vector<CandidateTrial> tried;
  std::string gpu;
  std::string driver;
  std::string engine;
};
// La recherche : next_candidate jusqu'à la fin, chaque essai rejoué une fois quand il ne tient pas ou tient
// de justesse, ou quand il a dépassé son délai. Refus : Cancelled sur demande d'arrêt, celui du premier essai
// s'il ne rend rien.
std::expected<CalibrationTrials, Refusal> run_calibration_trials(const TrialPlayer& play,
                                                                 const std::stop_token& stop,
                                                                 const std::function<void(float)>& fraction,
                                                                 int start_height = 0,
                                                                 bool try_finer = false);

// Les essais, l'un après l'autre (au plus une dizaine) : base porte les réglages de l'utilisateur, tier le
// niveau estimé de la carte (ses modèles : calibration_models), fraction reçoit l'avancement (0 à 1). La
// séquence du bench s'écrit dans plan.video si elle manque. Refus : Cancelled sur demande d'arrêt,
// DecodeFailed si le premier essai ne rend rien.
std::expected<MachineCalibration, Refusal> calibrate_machine(const BenchPlan& plan, const PlaySettings& base,
                                                             int display_height, Tier tier,
                                                             const std::stop_token& stop,
                                                             const std::function<void(float)>& fraction);

}  // namespace ajp::app::play
