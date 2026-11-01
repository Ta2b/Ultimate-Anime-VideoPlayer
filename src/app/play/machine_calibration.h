// La calibration de la machine de l'assistant (wizard.h, écran « Calibration de la machine ») : à la place
// des préréglages, le lecteur essaie des chaînes sur un même extrait très dur pour l'interpolation (media/
// bench_clip.h) et garde la meilleure que la machine tient. L'utilisateur ne voit qu'une barre de chargement.
//
// La recherche part du centre, 720p + SPAN + RIFE 4.22 lite (mode auto Rapide) : si la machine tient, elle
// monte (900p, 1080p) jusqu'au premier palier qui ne tient pas ; sinon elle descend (540p, 480p). Sans
// palier qui tienne, l'interpolation sort de l'équation (SPAN seul, de 720p vers le haut ou le bas), puis
// SPAN (RIFE seul), puis tout : entre SPAN et RIFE, SPAN passe toujours avant, c'est lui qui change le plus
// l'image. Ici la recherche, pure ; les essais (des « ajp play » enfants, auto_bench_run.h) dans le .cpp.
#pragma once

#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

#include "app/play/auto_bench_run.h"
#include "app/play/settings.h"

namespace ajp::app::play {

inline constexpr std::string_view kCalibrationRife = "rife_v4.22_lite.onnx";
// Les hauteurs de travail essayées, de la plus légère à la plus lourde ; la recherche part de kCenterHeight.
inline constexpr std::array<int, 5> kCalibrationHeights{480, 540, 720, 900, 1080};
inline constexpr int kCenterHeight = 720;

struct ChainCandidate {
  bool span = true;
  bool rife = true;
  int height = kCenterHeight;  // la hauteur de travail de SPAN ; sans SPAN, sans objet (0)
  bool operator==(const ChainCandidate&) const = default;
};

struct CandidateTrial {
  ChainCandidate chain;
  bool holds = false;
};

// Le prochain essai après ceux déjà faits ; rien quand la recherche est finie (best_candidate dit alors la
// chaîne gardée). Une famille (SPAN + RIFE, SPAN seul, RIFE seul) est finie dès qu'on connaît son plus haut
// palier qui tient, ou qu'aucun ne tient.
std::optional<ChainCandidate> next_candidate(const std::vector<CandidateTrial>& tried);
// La chaîne gardée : le plus haut palier qui tient de la meilleure famille ; ni SPAN ni RIFE si rien ne
// tient.
ChainCandidate best_candidate(const std::vector<CandidateTrial>& tried);
// Les réglages d'une chaîne, sur ceux de l'utilisateur (le HDR choisi, les langues… restent) : préréglage
// Personnalisé, réduction Hermite, RIFE 4.22 lite en mode auto Rapide.
PlaySettings candidate_settings(const ChainCandidate& chain, PlaySettings base);
// « 720p · SPAN · RIFE » ; pour le journal et le résumé.
std::string describe(const ChainCandidate& chain);

struct MachineCalibration {
  ChainCandidate chosen;
  std::vector<CandidateTrial> tried;
};

// Les essais, l'un après l'autre (au plus une dizaine) : base porte les réglages de l'utilisateur, fraction
// reçoit l'avancement (0 à 1). Refus : Cancelled sur demande d'arrêt, DecodeFailed si le premier essai ne
// rend rien.
std::expected<MachineCalibration, Refusal> calibrate_machine(const BenchPlan& plan, const PlaySettings& base,
                                                             const std::stop_token& stop,
                                                             const std::function<void(float)>& fraction);

}  // namespace ajp::app::play
