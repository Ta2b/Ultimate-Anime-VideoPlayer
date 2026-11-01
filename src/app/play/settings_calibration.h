// La calibration du grade adaptatif (Paramètres > HDR), en deux temps : la scène tenue pour sombre, puis pour
// claire (GradeSettings::adaptive_force), chacune avec ses curseurs et quatre images d'exemple
// (calibration_images.h) montrées à gauche du panneau, à la place de la vidéo. « Terminer » garde les deux
// bouts pour cet écran (with_calibrated_grade les reprend à chaque ouverture).
#pragma once

#include "app/play/calibration_shot.h"
#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

struct GradeWizard {
  int step = 0;          // 0 : fermée ; 1 : scène sombre ; 2 : scène claire
  int image = 0;         // l'image d'exemple de l'étape (0 à kCalibrationPerStep - 1)
  PlaySettings before;   // les réglages d'avant, que « Annuler » remet
  CalibrationShot shot;  // l'image montrée, sa place et son grade (step > 0)
};
// panel_left : le bord gauche du panneau des paramètres, en pixels ; l'image se place à sa gauche.
void grade_calibration(PlaySettings& d, const SettingsChoices& choices, GradeWizard& wizard, float scale,
                       float panel_left);

}  // namespace ajp::app::play
