// Les images d'exemple de la calibration du grade adaptatif (Paramètres > HDR > « Calibrer le grade
// adaptatif ») : quatre scènes sombres, quatre claires, dessinées par le lecteur (aucun fichier à livrer).
// Chacune dit ce qu'il faut voir à l'écran quand les réglages sont justes.
#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace ajp::app::play {

inline constexpr int kCalibrationWidth = 960;
inline constexpr int kCalibrationHeight = 540;
inline constexpr int kCalibrationPerStep = 4;  // images par étape (sombre, puis claire)

struct CalibrationImage {
  int id = 0;                           // 0 à 3 : scènes sombres ; 4 à 7 : scènes claires
  std::string_view title;               // clé de traduction (i18n.h)
  std::string_view guidance;            // ce qu'il faut voir, clé de traduction
  std::span<const std::uint16_t> rgba;  // kCalibrationWidth × kCalibrationHeight, RGBA16, codes BT.1886
};

// L'image index (0 à kCalibrationPerStep - 1) de l'étape : sombre (dark) ou claire. Dessinée au premier
// appel, gardée ensuite.
CalibrationImage calibration_image(bool dark, int index);

}  // namespace ajp::app::play
