// Les images d'exemple de la calibration du grade adaptatif (Paramètres > HDR > « Calibrer le grade
// adaptatif ») : quatre plans fixes sombres et quatre clairs, tirés de Re:Zero (saison 4, épisode 18) et de
// Slime (saison 4, épisode 24), embarqués en JPEG (data/calibration, SOURCES.md). Chacune dit ce qu'il faut
// voir à l'écran quand les réglages sont justes.
#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace ajp::app::play {

inline constexpr int kCalibrationPerStep = 4;  // images par étape (sombre, puis claire)

struct CalibrationImage {
  int id = 0;                 // 0 à 3 : scènes sombres ; 4 à 7 : scènes claires
  std::string_view title;     // clé de traduction (i18n.h)
  std::string_view guidance;  // ce qu'il faut voir, clé de traduction
  int width = 0;              // 0 : l'image ne s'est pas décodée (rgba vide)
  int height = 0;
  std::span<const std::uint16_t> rgba;  // width × height × 4, RGBA16, codes BT.1886
};

// L'image index (0 à kCalibrationPerStep - 1) de l'étape : sombre (dark) ou claire. Décodée à la demande ;
// seule la dernière demandée reste en mémoire (rgba vaut jusqu'à l'appel suivant pour une autre image).
CalibrationImage calibration_image(bool dark, int index);

}  // namespace ajp::app::play
