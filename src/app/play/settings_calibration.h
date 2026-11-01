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

}  // namespace ajp::app::play
