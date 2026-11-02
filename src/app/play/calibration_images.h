// Les images d'exemple de la calibration du grade adaptatif (Paramètres > HDR > « Calibrer le grade
// adaptatif ») : deux plans fixes sombres, deux sombres avec un éclat et quatre clairs, tirés de Re:Zero
// (saison 4, épisode 18) et de Slime (saison 4, épisode 24), livrés en une vidéo
// (data/calibration/calibration.mkv, SOURCES.md) que le lecteur lit en pause, plan par plan, dans la même
// chaîne que les épisodes : ce que la calibration montre est ce que la lecture rendra. Chaque plan dit ce
// qu'il faut voir à l'écran quand les réglages sont justes.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string_view>

namespace ajp::app::play {

inline constexpr int kCalibrationImages = 8;                      // plans de la vidéo, toutes étapes
inline constexpr std::int64_t kCalibrationSegmentUs = 6'000'000;  // chaque plan tient 6 s de la vidéo

// Les étapes de la calibration, dans leur ordre : scène sombre, scène sombre avec éclat, scène claire.
enum class CalibrationStep : std::uint8_t { Dark, Glow, Bright };
inline constexpr int kCalibrationSteps = 3;

struct CalibrationImage {
  int id = 0;                 // 0 et 1 : scènes sombres ; 2 et 3 : sombres avec éclat ; 4 à 7 : claires
  std::string_view title;     // clé de traduction (i18n.h)
  std::string_view guidance;  // ce qu'il faut voir, clé de traduction
  std::int64_t at_us = 0;     // l'instant montré dans la vidéo, au milieu de son plan
};

// Le nombre de plans de l'étape, et son plan index (0 à calibration_count(step) - 1).
int calibration_count(CalibrationStep step);
CalibrationImage calibration_image(CalibrationStep step, int index);
// Le plan id de la vidéo (0 à kCalibrationImages - 1), toutes étapes confondues : la calibration les montre
// dans l'ordre, chacun réglé à sa vraie place dans la courbe.
CalibrationImage calibration_shot(int id);

// La vidéo livrée : calibration/calibration.mkv à côté de SDL_GetBasePath (Contents/Resources dans AJP.app),
// comme luts/.
std::filesystem::path calibration_clip();

}  // namespace ajp::app::play
