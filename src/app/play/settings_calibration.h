// La calibration du grade adaptatif (Paramètres > HDR), en deux temps : la scène tenue pour sombre, puis pour
// claire (GradeSettings::adaptive_force), chacune avec ses curseurs et quatre plans d'exemple
// (calibration_images.h). Le lecteur ouvre la vidéo de calibration en pause, dans la même chaîne que les
// épisodes, et la montre à gauche du panneau ; à la fin il rend la vidéo d'avant, à sa position, ou
// l'accueil. « Terminer » garde les deux bouts pour cet écran (with_calibrated_grade les reprend à chaque
// ouverture).
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "app/play/calibration_shot.h"
#include "app/play/input.h"
#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

struct GradeWizard {
  int step = 0;          // 0 : fermée ; 1 : scène sombre ; 2 : scène claire
  int image = 0;         // le plan d'exemple de l'étape (0 à kCalibrationPerStep - 1)
  PlaySettings before;   // les réglages d'avant, que « Annuler » remet
  CalibrationShot shot;  // la place de la vidéo de calibration (step > 0)
  bool showing = false;  // la vidéo de calibration est celle qui se lit : elle se montre à sa place
  // La vidéo de calibration demandée (PlayFile) et pas encore rendue : à la fin, la vidéo d'avant (vide :
  // l'accueil), à sa position et dans sa pause.
  bool on_clip = false;
  std::filesystem::path return_file;
  std::int64_t return_us = 0;
  bool return_paused = false;
  int sought = -1;         // le plan (CalibrationImage::id) déjà cherché dans la vidéo
  double sought_at = 0.0;  // quand (ImGui::GetTime), pour ne pas relancer la recherche à chaque image
};
// panel_left : le bord gauche du panneau des paramètres, en pixels ; la vidéo se place à sa gauche. actions
// reçoit l'ouverture de la vidéo de calibration, puis la recherche de chaque plan et sa pause.
void grade_calibration(PlaySettings& d, const SettingsChoices& choices, GradeWizard& wizard, float scale,
                       float panel_left, std::vector<Input>& actions);
// À chaque image, panneau ouvert ou non : la calibration finie (ou quittée), la vidéo d'avant se rouvre.
void settle_calibration(GradeWizard& wizard, std::vector<Input>& actions);

}  // namespace ajp::app::play
