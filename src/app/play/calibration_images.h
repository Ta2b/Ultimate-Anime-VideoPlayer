// Les images d'exemple de la calibration du grade adaptatif (Paramètres > HDR > « Calibrer le grade
// adaptatif ») : quatre plans fixes sombres et quatre clairs, tirés de Re:Zero (saison 4, épisode 18) et de
// Slime (saison 4, épisode 24), livrés en une vidéo (data/calibration/calibration.mkv, SOURCES.md) que le
// lecteur lit en pause, plan par plan, dans la même chaîne que les épisodes : ce que la calibration montre
// est ce que la lecture rendra. Chaque plan dit ce qu'il faut voir à l'écran quand les réglages sont justes.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string_view>

namespace ajp::app::play {

inline constexpr int kCalibrationPerStep = 4;                     // plans par étape (sombre, puis claire)
inline constexpr std::int64_t kCalibrationSegmentUs = 6'000'000;  // chaque plan tient 6 s de la vidéo

struct CalibrationImage {
  int id = 0;                 // 0 à 3 : scènes sombres ; 4 à 7 : scènes claires
  std::string_view title;     // clé de traduction (i18n.h)
  std::string_view guidance;  // ce qu'il faut voir, clé de traduction
  std::int64_t at_us = 0;     // l'instant montré dans la vidéo, au milieu de son plan
};

// Le plan index (0 à kCalibrationPerStep - 1) de l'étape : sombre (dark) ou claire.
CalibrationImage calibration_image(bool dark, int index);

// La vidéo livrée : calibration/calibration.mkv à côté de SDL_GetBasePath (Contents/Resources dans AJP.app),
// comme luts/.
std::filesystem::path calibration_clip();

}  // namespace ajp::app::play
