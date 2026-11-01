// La calibration du pic de l'écran (Paramètres > HDR) : une mire au PQ exact (hdr_pattern.h), un curseur de
// nits qu'on monte jusqu'à ce que le carré se fonde dans son cadre, et le résultat gardé pour cet écran.
#pragma once

#include <optional>

#include "app/play/hdr_pattern.h"
#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

// calibration : la mire montrée (absente : la calibration est fermée). Le pic retenu va à l'écran
// (d.display_peaks, sous choices.display_key) et le pic repasse en automatique ; sans clé d'écran, il devient
// le pic manuel. Vrai : le pic automatique a changé, la chaîne se rouvre pour le reprendre.
bool peak_calibration(PlaySettings& d, const SettingsChoices& choices,
                      std::optional<HdrPattern>& calibration);

// La calibration du grade adaptatif, en deux temps sur la vidéo en cours : la scène tenue pour sombre, puis
// pour claire (GradeSettings::adaptive_force), chacune avec ses curseurs. « Terminer » garde les deux bouts
// pour cet écran (with_calibrated_grade les reprend à chaque ouverture).
struct GradeWizard {
  int step = 0;         // 0 : fermée ; 1 : scène sombre ; 2 : scène claire
  PlaySettings before;  // les réglages d'avant, que « Annuler » remet
};
void grade_calibration(PlaySettings& d, const SettingsChoices& choices, GradeWizard& wizard);

}  // namespace ajp::app::play
