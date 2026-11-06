// La calibration de la machine (l'assistant, puis Vidéo > Post-traitement > « Résultats du bench ») : à la
// place des préréglages, le lecteur essaie des chaînes sur un même extrait très dur pour l'interpolation
// (media/bench_clip.h) et recommande la meilleure que la machine tient. L'utilisateur ne voit qu'une barre de
// chargement.
//
// Les hauteurs de travail essayées s'arrêtent à la moitié de l'écran : SPAN double l'image, au-delà sa sortie
// dépasserait l'écran (1080p : 540p au plus, 1440p : 720p, 4K : 1080p). Deux familles, dans l'ordre (choix de
// Tatoub, 04/11) :
//   1. SPAN seul, du palier le plus léger au plus lourd, jusqu'au premier qui ne tient pas ;
//   2. SPAN + RIFE 4.22 lite en mode auto Rapide, du plus haut palier que SPAN seul a tenu vers le bas,
//      jusqu'au premier qui tient : RIFE ne fait qu'ajouter du travail, un palier que SPAN seul ne tient pas
//      ne se tient pas avec RIFE.
// SPAN est le Balanced Sharp en FP8 quand la carte le prend (FP8 CUTLASS), sinon le Performance Sharp en
// FP16. Sans palier SPAN qui tienne, RIFE seul (SPAN coupé) est le recours. GMFSS et DistilDRBA ne sont pas
// essayés. Ici la recherche, pure, et le fichier des résultats ; les essais (des « ajp play » enfants,
// auto_bench_run.h) dans le .cpp.
#pragma once

#include <array>
#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/auto_bench_run.h"
#include "app/play/settings.h"

namespace ajp::app::play {

inline constexpr std::string_view kCalibrationRife = "rife_v4.22_lite.onnx";
// SPAN des essais : le Balanced Sharp en FP8 (précision Performance), sinon le modèle par défaut du lecteur,
// le Performance Sharp, en FP16 (précision Équilibre).
inline constexpr std::string_view kCalibrationSpanFp8 =
    "2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx";
// Ce que l'essai d'un SPAN FP8 doit avoir joué (BenchReading::span_variant) pour que la recherche le garde.
inline constexpr std::string_view kCalibrationFp8Variant = "FP8 CUTLASS";
// Les hauteurs de travail possibles, de la plus légère à la plus lourde.
inline constexpr std::array<int, 6> kCalibrationHeights{360, 480, 540, 720, 900, 1080};

// La hauteur de travail à partir de laquelle un essai SPAN dit si la carte prend le FP8.
inline constexpr int kFp8JudgeHeight = 720;

struct ChainCandidate {
  bool span = true;
  bool rife = true;
  int height = 0;  // la hauteur de travail de SPAN ; sans SPAN, 0 (la source telle quelle)
  bool operator==(const ChainCandidate&) const = default;
};

struct CandidateTrial {
  ChainCandidate chain;
  bool holds = false;
  double lost_percent = -1.0;  // images perdues ou manquées, en % des présentations ; < 0 : rien rendu
  double fps = 0.0;            // images neuves par seconde obtenues, en moyenne
  double low_fps = 0.0;        // le « 1 % low » : d'après les 1 % d'écarts les plus longs
  double target_fps = 0.0;     // celles visées : la séquence (24 i/s), doublée par RIFE
};

// Les hauteurs essayées pour un écran de cette hauteur, de la plus lourde à la plus légère : au plus la
// moitié de l'écran, au moins la plus légère (360p).
std::vector<int> calibration_heights(int display_height);

// Le prochain essai après ceux déjà faits ; rien quand la recherche est finie.
std::optional<ChainCandidate> next_candidate(const std::vector<CandidateTrial>& tried, int display_height);
// La chaîne recommandée : SPAN passe avant l'interpolation, donc le plus haut palier SPAN qui tient, avec
// RIFE s'il tient aussi à ce palier (sinon SPAN seul) ; RIFE seul si SPAN ne tient nulle part ; ni SPAN ni
// RIFE si rien ne tient. Les paliers plus bas avec RIFE restent au tableau, au choix de l'utilisateur.
ChainCandidate best_candidate(const std::vector<CandidateTrial>& tried);
// Les réglages d'une chaîne, sur ceux de l'utilisateur (le HDR choisi, les langues… restent) : préréglage
// Personnalisé, réduction Hermite, RIFE 4.22 lite en mode auto Rapide ; span_fp8 : SPAN Balanced Sharp en
// précision Performance (FP8), sinon le SPAN par défaut en FP16.
PlaySettings candidate_settings(const ChainCandidate& chain, PlaySettings base, bool span_fp8 = false);
// « 720p · SPAN · RIFE 4.22 lite (auto Rapide) » ; pour le tableau et le journal.
std::string describe(const ChainCandidate& chain);

struct MachineCalibration {
  ChainCandidate chosen;
  std::vector<CandidateTrial> tried;  // dans l'ordre joué
  std::string gpu;
  std::string date;  // AAAA-MM-JJ
  int display_height = 0;
  bool span_fp8 = false;  // les essais SPAN ont joué le Balanced Sharp en FP8
};

// Le fichier des résultats (calibration.toml, à côté de play.toml), que calibrate_machine écrit quand
// plan.data est donné.
std::filesystem::path calibration_file(const std::filesystem::path& data_dir);
std::string format_calibration(const MachineCalibration& calibration);
std::expected<MachineCalibration, Refusal> parse_calibration(std::string_view text);
std::expected<MachineCalibration, Refusal> load_calibration(const std::filesystem::path& file);
std::expected<void, Refusal> save_calibration(const std::filesystem::path& file,
                                              const MachineCalibration& calibration);

// Les essais, l'un après l'autre (au plus une dizaine) : base porte les réglages de l'utilisateur, fraction
// reçoit l'avancement (0 à 1). La séquence du bench s'écrit dans plan.video si elle manque. Refus : Cancelled
// sur demande d'arrêt, DecodeFailed si le premier essai ne rend rien.
std::expected<MachineCalibration, Refusal> calibrate_machine(const BenchPlan& plan, const PlaySettings& base,
                                                             int display_height, const std::stop_token& stop,
                                                             const std::function<void(float)>& fraction);

}  // namespace ajp::app::play
