// Les images d'exemple de la calibration du grade adaptatif (Paramètres > HDR > « Calibrer le grade
// adaptatif ») : dix-huit plans fixes tirés d'Akudama Drive, des Enfants du Temps, de Violet Evergarden, de
// Frieren, de Your Name, de Re:Zero, de Slime et de
// Black Clover, réunis en une vidéo (calibration.mkv, décrite par data/calibration/SOURCES.md ; tirée
// d'œuvres sous droits, elle n'est ni dans le dépôt ni dans les paquets) que le lecteur lit en pause, plan
// par plan, dans la même chaîne que les épisodes : ce que la calibration montre est ce que la lecture rendra.
// Chaque plan dit ce qu'il faut voir à l'écran quand les réglages sont justes.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string_view>

namespace ajp::app::play {

inline constexpr int kCalibrationImages = 18;                     // plans de la vidéo
inline constexpr std::int64_t kCalibrationSegmentUs = 6'000'000;  // chaque plan tient 6 s de la vidéo

struct CalibrationImage {
  int id = 0;                 // sa place dans la vidéo
  std::string_view title;     // clé de traduction (i18n.h)
  std::string_view guidance;  // ce qu'il faut voir, clé de traduction
  std::int64_t at_us = 0;     // l'instant montré dans la vidéo, au milieu de son plan
};

// L'ordre de la calibration : d'abord les plans obligatoires, qui tiennent les deux bouts de la courbe, son
// milieu, la peau et les éclats d'une scène claire (la cellule et son rai de lumière, Hina dans la pénombre,
// Hina sous la pluie, la ville sous le soleil, la salle aux lampes) ; puis les plans bonus, qui la précisent
// (projecteurs dans la brume, forêt, blancs, feu d'artifice, ombre chaude, crépuscule, bleus profonds du
// dragon, escalier sombre, corne lumineuse, ciel rose, vallée dans la lumière, mur de
// feu, chaîne de lumière).
inline constexpr std::array<int, kCalibrationImages> kCalibrationOrder{0, 1,  2,  3,  4,  5,  6,  7,  8,
                                                                       9, 10, 11, 12, 13, 14, 15, 16, 17};
inline constexpr int kCalibrationRequired = 5;

// Le plan id de la vidéo (0 à kCalibrationImages - 1) : la calibration les montre dans l'ordre, chacun réglé
// à sa vraie place dans la courbe.
CalibrationImage calibration_shot(int id);

// La vidéo : calibration/calibration.mkv dans les données de l'utilisateur (SDL_GetPrefPath), sinon à côté de
// SDL_GetBasePath ; le premier de ces chemins quand elle n'est nulle part.
std::filesystem::path calibration_clip();

}  // namespace ajp::app::play
