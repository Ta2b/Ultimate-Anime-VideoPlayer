// La calibration du pic de l'écran (Paramètres > HDR) : une mire au PQ exact (hdr_pattern.h), un curseur de
// nits qu'on monte jusqu'à ce que le carré se fonde dans son cadre, et le résultat posé comme pic du HDR.
#pragma once

#include <optional>

#include "app/play/hdr_pattern.h"
#include "app/play/settings.h"
#include "app/play/settings_panel.h"

namespace ajp::app::play {

// calibration : la mire montrée (absente : la calibration est fermée). « Appliquer » pose d.true_hdr_peak.
void peak_calibration(PlaySettings& d, const SettingsChoices& choices,
                      std::optional<HdrPattern>& calibration);

}  // namespace ajp::app::play
