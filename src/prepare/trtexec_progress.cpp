#include "prepare/trtexec_progress.h"

#include <array>

namespace ajp::prepare {
namespace {

struct Marker {
  std::string_view text;  // sous-chaîne cherchée dans la ligne
  float fraction;
  std::string_view stage;
  float after = 0.0F;  // le repère ne compte qu'une fois cette fraction atteinte
};

// Dans l'ordre d'apparition, relevés sur trtexec 11.3.0.99 (build logs du lecteur
// actuel et de ajp_trt_build, ~/ajp-reports/tensorrt.md). Le choix des tactiques
// occupe l'essentiel du temps. Pour SPAN, TensorRT 11 enchaîne trois passes du
// constructeur, chacune close par « Engine generation completed » : la principale
// (45 s sur 58), une seconde par le backend compilateur (12 s), puis la reprise de
// la meilleure depuis le timing cache (0,4 s). RIFE n'en fait qu'une. Une passe
// qui recommence après la première (« Profiling results in this builder pass »)
// est donc la seconde, et seule la fin de celle-ci mène à la sérialisation.
constexpr std::array kMarkers = {
    Marker{"&&&& RUNNING", 0.02F, "Démarrage de trtexec"},
    Marker{"Start parsing network model", 0.05F, "Lecture du modèle"},
    Marker{"Finished parsing network model", 0.10F, "Choix des tactiques"},
    Marker{"Total Activation Memory", 0.75F, "Plan mémoire"},
    Marker{"Engine generation completed", 0.78F, "Première passe du constructeur terminée"},
    Marker{"Profiling results in this builder pass", 0.80F, "Seconde passe du constructeur", 0.78F},
    Marker{"Engine generation completed", 0.95F, "Sérialisation du moteur", 0.80F},
    Marker{"Engine built in", 0.97F, "Moteur construit"},
    Marker{"&&&& PASSED", 1.0F, "Terminé"},
};

std::string_view trim(std::string_view text) {
  const auto first = text.find_first_not_of(" \t");
  if (first == std::string_view::npos) return {};
  return text.substr(first, text.find_last_not_of(" \t") - first + 1);
}

}  // namespace

std::optional<BuildStage> TrtexecProgress::feed(std::string_view line) {
  if (line.find("&&&& FAILED") != std::string_view::npos) failed_ = true;
  if (line.find("&&&& PASSED") != std::string_view::npos) passed_ = true;
  if (line.find("[E]") != std::string_view::npos) last_error_ = trim(line);
  for (const Marker& marker : kMarkers) {
    if (line.find(marker.text) == std::string_view::npos || marker.fraction <= fraction_ ||
        fraction_ < marker.after) {
      continue;
    }
    fraction_ = marker.fraction;
    return BuildStage{.fraction = marker.fraction, .stage = std::string(marker.stage)};
  }
  return std::nullopt;
}

}  // namespace ajp::prepare
