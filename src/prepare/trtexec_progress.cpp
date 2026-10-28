#include "prepare/trtexec_progress.h"

#include <array>

namespace ajp::prepare {
namespace {

struct Marker {
  std::string_view text;  // sous-chaîne cherchée dans la ligne
  float fraction;
  std::string_view stage;
};

// Dans l'ordre d'apparition. Le choix des tactiques occupe l'essentiel du temps
// (22 s pour RIFE 4.26, 53 à 64 s pour SPAN sur la 4070 Ti).
constexpr std::array kMarkers = {
    Marker{"&&&& RUNNING", 0.02F, "Démarrage de trtexec"},
    Marker{"Start parsing network model", 0.05F, "Lecture du modèle"},
    Marker{"Finished parsing network model", 0.10F, "Choix des tactiques"},
    Marker{"Total Activation Memory", 0.85F, "Plan mémoire"},
    Marker{"Engine generation completed", 0.90F, "Sérialisation du moteur"},
    Marker{"Engine built in", 0.95F, "Moteur construit"},
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
    if (line.find(marker.text) == std::string_view::npos || marker.fraction <= fraction_) continue;
    fraction_ = marker.fraction;
    return BuildStage{.fraction = marker.fraction, .stage = std::string(marker.stage)};
  }
  return std::nullopt;
}

}  // namespace ajp::prepare
