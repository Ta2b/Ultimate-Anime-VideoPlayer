// La place de la vidéo pendant la calibration du grade adaptatif (calibration_images.h) : à gauche du panneau
// des paramètres, jamais dessous. La vidéo de calibration y passe par la même chaîne et le même grade que les
// épisodes ; les sous-titres et l'OSD ne s'y montrent pas, l'interface reste par-dessus.
#pragma once

namespace ajp::app::play {

struct CalibrationShot {
  int x0 = 0;  // la place à l'écran, en pixels
  int y0 = 0;
  int x1 = 0;
  int y1 = 0;
};

}  // namespace ajp::app::play
