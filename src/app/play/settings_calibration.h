// La calibration du grade adaptatif (Paramètres > HDR) : les huit plans de la vidéo de calibration
// (calibration_images.h), chacun à sa vraie place dans la courbe (sa luminosité, son éclat, son ciel, mesurés
// comme en lecture), réglé avec les curseurs « cette scène ». « Terminer » ajuste la courbe sur les huit à la
// fois, comme l'apprentissage (grade_learning.h : learn_grade). Le lecteur ouvre la vidéo de calibration en
// pause, dans la même chaîne que les épisodes, et la montre à gauche du panneau ; à la fin il rend la vidéo
// d'avant, à sa position, ou l'accueil. « Terminer » garde les deux bouts pour cet écran
// (with_calibrated_grade les reprend à chaque ouverture).
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "app/play/calibration_images.h"
#include "app/play/calibration_shot.h"
#include "app/play/grade_learning.h"
#include "app/play/input.h"
#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

struct GradeWizard {
  int step = 0;   // 0 : fermée ; 1 : en cours
  int image = 0;  // le plan montré (0 à kCalibrationImages - 1)
  // Ce que l'utilisateur veut sur chaque plan : sa scène et les valeurs qu'il y voit, retenues tant qu'il le
  // regarde.
  std::array<std::optional<LearnSample>, kCalibrationImages> samples;
  PlaySettings before;          // les réglages d'avant, que « Annuler » remet
  std::string profile;          // le profil de grade que « Terminer » garde (nouveau, ou celui recalibré)
  std::array<char, 64> name{};  // le nom tapé pour un nouveau profil
  CalibrationShot shot;         // la place de la vidéo de calibration (step > 0)
  bool showing = false;         // la vidéo de calibration est celle qui se lit : elle se montre à sa place
  // La vidéo de calibration demandée (PlayFile) et pas encore rendue : à la fin, la vidéo d'avant (vide :
  // l'accueil), à sa position et dans sa pause.
  bool on_clip = false;
  std::filesystem::path return_file;
  std::int64_t return_us = 0;
  bool return_paused = false;
  int sought = -1;         // le plan (CalibrationImage::id) déjà cherché dans la vidéo
  double sought_at = 0.0;  // quand (ImGui::GetTime), pour ne pas relancer la recherche à chaque image
  bool autostart = false;  // l'assistant de premier démarrage la demande : elle part seule, nouveau profil
};
// panel_left : le bord gauche du panneau des paramètres, en pixels ; la vidéo se place à sa gauche. actions
// reçoit l'ouverture de la vidéo de calibration, puis la recherche de chaque plan et sa pause.
void grade_calibration(PlaySettings& d, const SettingsChoices& choices, GradeWizard& wizard, float scale,
                       float panel_left, std::vector<Input>& actions);
// À chaque image, panneau ouvert ou non : la calibration finie (ou quittée), la vidéo d'avant se rouvre.
void settle_calibration(GradeWizard& wizard, std::vector<Input>& actions);
// Grade adaptatif, « Cette scène » : noirs, couleurs et gris moyen de la scène à l'écran, réglés directement
// (la correction se répartit sur les bouts de la courbe, grade_learning.h : correct_scene), et
// l'apprentissage tiré des corrections retenues.
void scene_correction(PlaySettings& d, const SettingsChoices& choices, std::vector<Input>& actions);

}  // namespace ajp::app::play
